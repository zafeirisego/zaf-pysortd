
#include "rashomon/tracker.h"
#include "rashomon/split_tracker.h"
#include "rashomon/branch_tracker.h"
#include "rashomon/terminal_tracker.h"
#include "rashomon/leaf_tracker.h"

namespace SORTD {

    template <class OT>
    std::shared_ptr<AbstractTracker<OT>> AbstractTracker<OT>::CreateTracker(Solver<OT>* solver, Cache<OT>* cache, const ADataView& data, const typename Solver<OT>::Context& context, int max_depth, int max_num_nodes, typename OT::SolType UB) {
        max_depth = std::min(max_depth, max_num_nodes);

        bool leaf_node = max_depth == 0 || max_num_nodes == 0
            || !solver->SatisfiesMinimumLeafNodeSize(data, 2); // Not enough data to split into two
        if constexpr (Solver<OT>::sparse_objective) {
            auto branching_costs = solver->GetBranchingCosts(data, context, 0);
            leaf_node &= (UB < branching_costs); // UB too small to add another split
        }
        if (leaf_node) {
            return std::make_shared<LeafTracker<OT>>(solver, data, context, UB);
        }

        if constexpr (OT::use_terminal) {
            if (max_depth == 1 || max_depth == 2) {
                return std::make_shared<TerminalTracker<OT>>(solver, cache, data, context, max_depth, max_num_nodes, UB);
            }
        }

        return std::make_shared<BranchTracker<OT>>(solver, cache, data, context, max_depth, max_num_nodes, UB);
    }

    template <class OT>
    SplitTracker<OT>::SplitTracker(Solver<OT>* solver, Cache<OT>* cache, ADataView& data, const typename Solver<OT>::Context& context, int max_depth, int max_num_nodes, int num_features, typename Solver<OT>::SolType UB, int feature) :
        solver(solver), feature(feature), max_depth(max_depth), max_num_nodes(max_num_nodes), data(data), context(context), num_features(num_features), UB(UB), cache(cache) {
        runtime_assert(!context.GetBranch().HasBranchedOnFeature(feature));
        next_solution_value = FindInitialSolutionValue();
        if (next_solution_value > UB + SOLUTION_PRECISION) {
            Erase();
            return;
        }
    }

    template <class OT>
    void SplitTracker<OT>::Initialize() {
        using SolType = typename SplitTracker<OT>::SolType;
        using SolContainer = typename SplitTracker<OT>::SolContainer;
        using Context = typename SplitTracker<OT>::Context;

        initialized = true;
        Context left_context, right_context;
        solver->GetTask()->GetLeftContext(data, context, feature, left_context);
        solver->GetTask()->GetRightContext(data, context, feature, right_context);

        ADataView left_data, right_data;
        solver->GetSplitter().Split(data, context.GetBranch(), feature, left_data, right_data);
        runtime_assert(solver->SatisfiesMinimumLeafNodeSize(left_data) && solver->SatisfiesMinimumLeafNodeSize(right_data));
        auto branching_costs = solver->GetBranchingCosts(data, context, feature);
        const int max_size_subtree = std::min((1 << (max_depth - 1)) - 1, max_num_nodes - 1);
        const int min_size_subtree = max_num_nodes - 1 - max_size_subtree;
        const int left_subtree_size = min_size_subtree;
        const int right_subtree_size = max_num_nodes - left_subtree_size - 1;

        // If there are left and right trackers in the cache, use them directly.
        SolContainer left_sol, right_sol;
        if (cache->IsOptimalAssignmentCached(left_data, left_context.GetBranch(), max_depth - 1, left_subtree_size)) {
            left_sol = cache->RetrieveOptimalAssignment(left_data, left_context.GetBranch(), max_depth - 1, left_subtree_size);
        } else {
            SolContainer right_lb = cache->RetrieveLowerBound(left_data, left_context.GetBranch(), max_depth - 1, left_subtree_size);
            left_sol = solver->SolveSubTreeIncrementalUB(left_data, left_context, UB - right_lb.solution - branching_costs, max_depth - 1, left_subtree_size);
        }
        if (cache->IsOptimalAssignmentCached(right_data, right_context.GetBranch(), max_depth - 1, right_subtree_size)) {
            right_sol = cache->RetrieveOptimalAssignment(right_data, right_context.GetBranch(), max_depth - 1, right_subtree_size);
        } else {
            SolContainer left_lb = cache->RetrieveLowerBound(right_data, right_context.GetBranch(), max_depth - 1, right_subtree_size);
            right_sol = solver->SolveSubTreeIncrementalUB(right_data, right_context, UB - left_lb.solution - branching_costs, max_depth - 1, right_subtree_size);
        }

        left_tracker = CreateChildTracker(solver, cache,  left_data,  left_context,  max_depth - 1, left_subtree_size,  UB - right_sol.solution - branching_costs);
        right_tracker = CreateChildTracker(solver, cache, right_data, right_context, max_depth - 1, right_subtree_size, UB - left_sol.solution - branching_costs);
        runtime_assert(std::abs(left_tracker->GetFirstSolutionValue() - left_sol.solution) <= SOLUTION_PRECISION);
        runtime_assert(std::abs(right_tracker->GetFirstSolutionValue() - right_sol.solution) <= SOLUTION_PRECISION);

        // If the left and right solutions are completely enumerated using the depth-two solver,
        // combine them to obtain split tracker's full set of solutions
        if (left_tracker->IsSolutionListComplete() && right_tracker->IsSolutionListComplete()) {
            CombineSolutions();
            return;
        }

        SolType sol = OT::worst;
        // Check if the first solution is a trivial extension (two leaf nodes with the same label), and the labels cannot be switched
        if (solver->GetSolverParameters().ignore_trivial_extentions
                && left_sol.label != INT32_MAX && right_sol.label != INT32_MAX
                && std::abs(left_sol.label - right_sol.label) < SOLUTION_PRECISION 
                && !CanSwitchTrivialExtensionLabel(left_data, right_data)) {
            runtime_assert(left_tracker->HasNext() && right_tracker->HasNext());
            // Pop the first solution for both left and right (should be available, otherwise this SplitTracker is not correctly initialized)
            if (left_tracker->HasNext()) left_tracker->Pop();
            if (right_tracker->HasNext()) right_tracker->Pop();
            visited_heap_indices.Insert(0, 0);
            // Since the first solutions are leafs, they should not have other solutions than the trivial extension
            runtime_assert(left_tracker->GetSolutionN(0)->GetNumSolutions() == 1 && right_tracker->GetSolutionN(0)->GetNumSolutions() == 1);
            // We skip the (0,0) combination and add both (1,0) and (0,1)
            if (left_tracker->HasNext()) {
                CombineSols<OT>(left_tracker->GetNextSolutionValue(), right_sol.solution, branching_costs, sol);
                if (sol < UB + SOLUTION_PRECISION ) {
                    heap.emplace(HeapEntry(sol, 1, 0));
                    max_heap_value = sol;
                }
                visited_heap_indices.Insert(1, 0);
            }
            if (right_tracker->HasNext()) {
                CombineSols<OT>(left_sol.solution, right_tracker->GetNextSolutionValue(), branching_costs, sol);
                if (sol < UB + SOLUTION_PRECISION) {
                    heap.emplace(HeapEntry(sol, 0, 1));
                    max_heap_value = std::max(max_heap_value, sol);
                }
                visited_heap_indices.Insert(0, 1);
            }            
            next_solution_value = heap.empty() ? OT::worst : heap.top().solution;
            return;
        }

        CombineSols<OT>(left_sol.solution, right_sol.solution, branching_costs, sol);
        heap.emplace(HeapEntry(sol, 0, 0));
        max_heap_value = sol;
        visited_heap_indices.Insert(0, 0);
    }

    template <class OT>
    typename SplitTracker<OT>::SolType SplitTracker<OT>::FindInitialSolutionValue() {
        using SolType = typename SplitTracker<OT>::SolType;
        using SolContainer = typename SplitTracker<OT>::SolContainer;
        using Context = typename SplitTracker<OT>::Context;

        Context left_context, right_context;
        solver->GetTask()->GetLeftContext(data, context, feature, left_context);
        solver->GetTask()->GetRightContext(data, context, feature, right_context);

        ADataView left_data, right_data;
        solver->GetSplitter().Split(data, context.GetBranch(), feature, left_data, right_data);
        if (!solver->SatisfiesMinimumLeafNodeSize(left_data) || !solver->SatisfiesMinimumLeafNodeSize(right_data)) {
            return OT::worst;
        }

        const int max_size_subtree = std::min((1 << (max_depth - 1)) - 1, max_num_nodes - 1);
        const int min_size_subtree = max_num_nodes - 1 - max_size_subtree;
        const int left_subtree_size = min_size_subtree;
        const int right_subtree_size = max_num_nodes - left_subtree_size - 1;

        SolType branching_costs = solver->GetBranchingCosts(data, context, feature);
        SolContainer left_sol, right_sol;
        if (cache->IsOptimalAssignmentCached(left_data, left_context.GetBranch(), max_depth - 1, left_subtree_size)) {
            left_sol = cache->RetrieveOptimalAssignment(left_data, left_context.GetBranch(), max_depth - 1, left_subtree_size);
        } else {
            SolContainer right_lb = cache->RetrieveLowerBound(left_data, left_context.GetBranch(), max_depth - 1, left_subtree_size);
            left_sol = solver->SolveSubTreeIncrementalUB(left_data, left_context, UB - right_lb.solution - branching_costs, max_depth - 1, left_subtree_size);
        }

        if (cache->IsOptimalAssignmentCached(right_data, right_context.GetBranch(), max_depth - 1, right_subtree_size)) {
            right_sol = cache->RetrieveOptimalAssignment(right_data, right_context.GetBranch(), max_depth - 1, right_subtree_size);
        } else {
            right_sol = solver->SolveSubTreeIncrementalUB(right_data, right_context, UB - left_sol.solution - branching_costs, max_depth - 1, right_subtree_size);
        }
        
        // Check if no solution found
        if (CheckEmptySol<OT>(left_sol) || CheckEmptySol<OT>(right_sol)) return OT::worst;
        
        // Check if within UB
        SolType sol = OT::worst;
        CombineSols<OT>(left_sol.solution, right_sol.solution, branching_costs, sol);
        if (sol > UB + SOLUTION_PRECISION) return sol;

        // When ignoring trivial extensions, we run into a problem if the two optimal solutions are leaf nodes with the same label
        if (solver->GetSolverParameters().ignore_trivial_extentions
            && left_sol.label != INT32_MAX && right_sol.label != INT32_MAX
            && std::abs(left_sol.label - right_sol.label) < SOLUTION_PRECISION) {
            //std::cout << "Warning: matching labels " << feature << std::endl;

            if (!CanSwitchTrivialExtensionLabel(left_data, right_data)) {

                //int depth = context.GetBranch().Depth();
                //for (int i = 0; i < depth; i++) std::cout << "  ";
                //std::cout << context.GetBranch() << " > " << feature;

                // We need to skip the first solution. Therefore, we initialize the split tracker, and pop both the left and right first solution
                Initialize();
                //std::cout << " " << (next_solution_value > UB + SOLUTION_PRECISION ? " Alt > UB " : "Alt < UB") 
                //    << " " << (std::round(next_solution_value * 1e4) / 10000.0) << " | " << (std::round(UB * 1e4) / 10000.0) << std::endl;
                return next_solution_value;
            }
        }

        
        return sol;
    }

    template <class OT>
    bool SplitTracker<OT>::CanSwitchTrivialExtensionLabel(ADataView& left_data, ADataView& right_data) {
        using SolType = typename SplitTracker<OT>::SolType;
        using SolContainer = typename SplitTracker<OT>::SolContainer;
        using Context = typename SplitTracker<OT>::Context;

        const int num_labels = data.NumLabels();
        // First search for two equal-valued leaf solutions with different labels
        if (num_labels > 1) { // Don't try this if we are doing regression
            Context left_context, right_context;
            solver->GetTask()->GetLeftContext(data, context, feature, left_context);
            solver->GetTask()->GetRightContext(data, context, feature, right_context);
            double left_opt = DBL_MAX, right_opt = DBL_MAX;
            std::vector<int> left_opt_labels, right_opt_labels;
            for (int label = 0; label < num_labels; label++) {
                auto left_cost = solver->GetTask()->GetLeafCosts(left_data, left_context, label);
                auto right_cost = solver->GetTask()->GetLeafCosts(right_data, right_context, label);
                if (std::abs(left_opt - left_cost) < SOLUTION_PRECISION) left_opt_labels.push_back(label);
                else if (left_cost < left_opt) {
                    left_opt = left_cost;
                    left_opt_labels = { label };
                }
                if (std::abs(right_opt - right_cost) < SOLUTION_PRECISION) right_opt_labels.push_back(label);
                else if (right_cost < right_opt) {
                    right_opt = right_cost;
                    right_opt_labels = { label };
                }
            }
            if (left_opt_labels.size() > 1 || right_opt_labels.size() > 1) {
                // There is at least one leaf where we can switch the label
                return true;
            }
        }
        return false;
    }

    template <class OT>
    typename SplitTracker<OT>::SolutionTrackerP SplitTracker<OT>::GetLastUnusedSolution() {
        using SolutionTracker = typename SplitTracker<OT>::SolutionTracker;
        if (IsSolutionListComplete()) {
            if (this->HasNSolution(solution_index)) {
                return this->GetSolutionN(solution_index);
            }
            throw std::runtime_error("GetLastUnusedSolution expects to have a solution at solution index. Check first with HasNext or Pop first.");
        }
        runtime_assert(!solutions.empty());
        return solutions[solutions.size() - 1];
    }

    template <class OT>
    bool SplitTracker<OT>::CreateCandidateSolution(std::shared_ptr<RecursiveSolutionTracker<OT>> solution, size_t left, size_t right) {
        using SolType = typename SplitTracker<OT>::SolType;
        //clock_t clock_start = clock();
        SolType comb_sol;
        auto branching_costs = solver->GetBranchingCosts(data, context, feature);
        auto& left_sol = left_tracker->GetSolutionN(left);
        auto& right_sol = right_tracker->GetSolutionN(right);
        CombineSols<OT>(left_sol->obj, right_sol->obj, branching_costs, comb_sol);
        runtime_assert(solution->obj <= comb_sol + 2 * SOLUTION_PRECISION);
        
        if (IsTrivialExtension(left_sol,right_sol)) {
            bool alternate_exists = SwitchToAlternativeSolution(left_sol,right_sol);
            if (!alternate_exists) {
                visited_heap_indices.Insert(left, right);
                return true;
            }
        }
                
        if (SOL_EQUAL(typename OT::SolType, comb_sol, solution->obj)) {
            solution->solutions.push_back(std::make_pair(left_sol, right_sol));

            visited_heap_indices.Insert(left, right);
            //solver->GetStatistics().time_rashomon_split_heap += double(clock() - clock_start) / CLOCKS_PER_SEC;
            return true;
        } else if (comb_sol <= UB + SOLUTION_PRECISION) {
            auto new_entry = HeapEntry(comb_sol, left, right);
            if ((heap.empty() && heap_buffer.empty()) || (!heap.empty() && comb_sol <= max_heap_value + SOLUTION_PRECISION)) {
                heap.emplace(new_entry);
                if (heap.size() == 1) max_heap_value = new_entry.solution;
            } else {
                heap_buffer.push_back(new_entry);
                if (new_entry.solution > max_heap_buffer_value) max_heap_buffer_value = new_entry.solution;
            }
        }
        // Update visited indexes
        visited_heap_indices.Insert(left, right);
        //solver->GetStatistics().time_rashomon_split_heap += double(clock() - clock_start) / CLOCKS_PER_SEC;
        return false;
    }

    template <class OT>
    void SplitTracker<OT>::MergeHeapBuffer() {
        //clock_t clock_start = clock();
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, CompareHeapEntry> temp_heap(heap_buffer.begin(), heap_buffer.end());
        heap = temp_heap;
        max_heap_value = max_heap_buffer_value;
        max_heap_buffer_value = 0.0;
        heap_buffer.clear();
        //solver->GetStatistics().time_rashomon_split_heap += double(clock() - clock_start) / CLOCKS_PER_SEC;
    }

    template <class OT>
    void SplitTracker<OT>::CombineSolutions() {
        using SolType = typename SplitTracker<OT>::SolType;
        using SolutionTracker = typename SplitTracker<OT>::SolutionTracker;
        using SolutionTrackerP = std::shared_ptr<SolutionTracker>;

        //clock_t clock_start = clock();
        solution_list_complete = true;
        solution_index = static_cast<size_t>(-1); // that is, we have not used the first (0) solution yet
        SolType worst = OT::worst;
        SolType branching_costs = solver->GetBranchingCosts(data, context, feature);
        std::unordered_map<int64_t, std::shared_ptr<RecursiveSolutionTracker<OT>>> solution_map;
        for (auto& left_sol : left_tracker->GetSolutions()) {
            for (auto& right_sol : right_tracker->GetSolutions()) {
                if (IsTrivialExtension(left_sol,right_sol)) {
                    bool alternate_exists = SwitchToAlternativeSolution(left_sol,right_sol);
                    if (!alternate_exists) continue;
                }
                SolType obj = OT::worst;
                // It is possible that incrementing left or right index of the solution yields one with the same value.
                // Therefore, additionally investigate these cases.
                CombineSols<OT>(left_sol->obj, right_sol->obj, branching_costs, obj);
                if (obj > UB + SOLUTION_PRECISION) break;
                int64_t i_obj = static_cast<int64_t>(std::round(obj / SOLUTION_PRECISION));
                auto sol = solution_map[i_obj];
                if (sol == nullptr) {
                    sol = std::make_shared<RecursiveSolutionTracker<OT>>(obj, feature, num_features);
                    solutions.push_back(sol);
                    solution_map[i_obj] = sol;
                }
                sol->UpdateNumSolutions(left_sol->GetNumSolutions() * right_sol->GetNumSolutions());
                sol->solutions.push_back(std::make_pair(left_sol, right_sol));

                if (sol->cumulative_sol_count.empty()) {
                    sol->cumulative_sol_count.push_back(left_sol->GetNumSolutions() * right_sol->GetNumSolutions() - 1);
                } else {
                    sol->cumulative_sol_count.push_back(left_sol->GetNumSolutions() * right_sol->GetNumSolutions() + sol->cumulative_sol_count.back());
                }
            }
        }
        std::sort(solutions.begin(), solutions.end(), CompareSolutionTrackers<OT>());
        next_solution_value = solutions.empty() ? OT::worst : solutions[0]->obj;
        for (auto& sol : solutions) {
            sol->Shrink();
        }
        if (solver->GetParameters().GetBooleanParameter("track-tree-statistics")) {
            for (auto& sol : solutions) {
                sol->CalculateFeatureStats();
                sol->CalculateNodeStats();
            }
        }
        //solver->GetStatistics().time_rashomon_combine_terminal += double(clock() - clock_start) / CLOCKS_PER_SEC;
    }

    template <class OT>
    void SplitTracker<OT>::AccumulateSameSolutions(std::shared_ptr<RecursiveSolutionTracker<OT>> solution, std::vector<AccumulateIndex>& popped_indices) {

        // One by one pop the indices and check recursively whether incrementing one of the left or right indices by one results in the same solution value.
        for (size_t i = 0; i < popped_indices.size(); i++) {
            auto& popped_index = popped_indices[i];
            
            size_t left_index = popped_index.lindex;
            size_t right_index = popped_index.rindex;
            bool expand_left = popped_index.expand_left;
            bool expand_right = popped_index.expand_right;
            size_t iterated_left_index = left_index + 1;
            size_t iterated_right_index = right_index + 1;
            
            ++solver->sol_count;

            if (expand_left) {
                if (left_tracker->HasNSolution(iterated_left_index) && !visited_heap_indices.IsVisited(iterated_left_index, right_index)) {
                    size_t right_index_low = visited_heap_indices.GetFirstRightNotVisited(iterated_left_index);
                    for (size_t ri = right_index_low; ri <= right_index; ri++) {
                        bool solution_found = CreateCandidateSolution(solution, iterated_left_index, ri);
                        if (solution_found) {
                            popped_indices.emplace_back(iterated_left_index, ri, true, true);
                            break;
                        }
                    }
                } else if (left_tracker->HasNSolution(iterated_left_index) &&
                    SOL_EQUAL(typename OT::SolType, left_tracker->GetSolutionN(iterated_left_index)->obj, left_tracker->GetSolutionN(left_index)->obj)) {
                    popped_indices.emplace_back(iterated_left_index, right_index, true, false);
                }
            }

            if (expand_right) {
                if (right_tracker->HasNSolution(iterated_right_index) && !visited_heap_indices.IsVisited(left_index, iterated_right_index)) {
                    size_t left_index_low = visited_heap_indices.GetFirstLeftNotVisited(iterated_right_index);
                    for (size_t li = left_index_low; li <= left_index; li++) {
                        bool solution_found = CreateCandidateSolution(solution, li, iterated_right_index);
                        if (solution_found) {
                            popped_indices.emplace_back(li, iterated_right_index, true, true);
                            break;
                        }
                    }
                } else if (right_tracker->HasNSolution(iterated_right_index) &&
                    SOL_EQUAL(typename OT::SolType, right_tracker->GetSolutionN(iterated_right_index)->obj, right_tracker->GetSolutionN(right_index)->obj)) {
                    popped_indices.emplace_back(left_index, iterated_right_index, false, true);
                }
            }
        }
    }

    template <class OT>
    void SplitTracker<OT>::PopSameHeapSolutions(std::shared_ptr<RecursiveSolutionTracker<OT>> solution, std::vector<AccumulateIndex>& popped_indices) {
        if (heap.empty()) return;
        HeapEntry entry = heap.top();
        auto obj = entry.solution;
        auto current_obj = solution->obj;
        while (SOL_EQUAL(typename OT::SolType, obj, current_obj)) {
            //clock_t clock_start = clock();
            heap.pop();
            //solver->GetStatistics().time_rashomon_split_heap += double(clock() - clock_start) / CLOCKS_PER_SEC;

            auto left_sol = left_tracker->GetSolutionN(entry.lindex);
            auto right_sol = right_tracker->GetSolutionN(entry.rindex);
            if (IsTrivialExtension(left_sol,right_sol)) {
                bool alternate_exists = SwitchToAlternativeSolution(left_sol,right_sol);
                if (!alternate_exists) {
                    entry = heap.top();
                    obj = entry.solution;
                    continue;
                }
            };
            solution->solutions.push_back(std::make_pair(left_sol, right_sol));
            popped_indices.emplace_back(entry.lindex, entry.rindex, true, true);
            
            if (heap.empty()) {
                if (!heap_buffer.empty()) {
                    MergeHeapBuffer();
                } else {
                    break;
                }
            }

            entry = heap.top();
            obj = entry.solution;
        }
        auto next_value = heap.empty() ? OT::worst : heap.top().solution;
        runtime_assert(obj <= next_value + 2 * SOLUTION_PRECISION);
    }

    template <class OT>
    void SplitTracker<OT>::Pop() {
        using SolType = typename SplitTracker<OT>::SolType;
        using SolutionTracker = typename SplitTracker<OT>::SolutionTracker;
        using SolutionTrackerP = std::shared_ptr<SolutionTracker>;

        if (!initialized) {
            Initialize();
        }
        // When the split tracker solutions completely enumerated as a result of its depth-two left and right solutions pop using indexing.
        if (IsSolutionListComplete()) {
            solution_index++;
            if (solution_index + 1 < solutions.size()) {
                next_solution_value = solutions[solution_index + 1]->obj;
            } else {
                next_solution_value = OT::worst;
            }
            return;
        }

        if (heap.empty()) {
            if (!heap_buffer.empty()) {
                MergeHeapBuffer();
            } else {
                Erase();
                return;
            }
        }
        auto promised_solution_value = GetNextSolutionValue();

        HeapEntry entry = heap.top();
        //clock_t clock_start = clock();
        heap.pop();
        //solver->GetStatistics().time_rashomon_split_heap += double(clock() - clock_start) / CLOCKS_PER_SEC;

        auto solution = std::make_shared<RecursiveSolutionTracker<OT>>(entry.solution, feature, num_features);

        auto left_sol = left_tracker->GetSolutionN(entry.lindex);
        auto right_sol = right_tracker->GetSolutionN(entry.rindex);
        
        if (IsTrivialExtension(left_sol, right_sol)) {
            bool alternate_exists = SwitchToAlternativeSolution(left_sol, right_sol);
            // We should not have added this solution to the heap, unless one label could be flipped
            runtime_assert(alternate_exists);
        }
        solution->solutions.push_back(std::make_pair(left_sol, right_sol));
        std::vector<AccumulateIndex> popped_indices = { {entry.lindex,entry.rindex, true, true} };
        // Store all same valued solutions under one solution tracker.
        PopSameHeapSolutions(solution, popped_indices);
        AccumulateSameSolutions(solution, popped_indices);

        for (auto& sol : solution->solutions) {
            solution->UpdateNumSolutions(sol.first->GetNumSolutions() * sol.second->GetNumSolutions()); // to handle multiple optimum solutions for the subtree this calculation needs to happen here.
            if (solution->cumulative_sol_count.empty()) {
                solution->cumulative_sol_count.push_back(sol.first->GetNumSolutions() * sol.second->GetNumSolutions() - 1);
            } else {
                solution->cumulative_sol_count.push_back(sol.first->GetNumSolutions() * sol.second->GetNumSolutions() + solution->cumulative_sol_count.back());
            }
        }
        if (solver->GetParameters().GetBooleanParameter("track-tree-statistics")) {
            solution->CalculateFeatureStats();
            solution->CalculateNodeStats();
        }

#ifdef DEBUG
        runtime_assert(std::abs(promised_solution_value - solution->obj) <= 2 * SOLUTION_PRECISION);
        if (!solutions.empty()) {
            auto last_solution = solutions[solutions.size() - 1];
            runtime_assert(last_solution->obj <= solution->obj + 2 * SOLUTION_PRECISION);
        } else {
            runtime_assert(std::abs(next_solution_value - solution->obj) <= 2 * SOLUTION_PRECISION);
        }
#endif
        solution->Shrink();
        solutions.push_back(solution);

        if (heap.empty() && !heap_buffer.empty()) MergeHeapBuffer();

        if (!heap.empty()) {
            entry = heap.top();
            next_solution_value = entry.solution;
        } else {
            next_solution_value = OT::worst;
        }

#ifdef DEBUG
        runtime_assert(solution->obj <= GetNextSolutionValue() + 2 * SOLUTION_PRECISION);
        runtime_assert(promised_solution_value <= GetNextSolutionValue() + 2 * SOLUTION_PRECISION);
#endif
    }

    template <class OT>
    std::shared_ptr<AbstractTracker<OT>> SplitTracker<OT>::CreateChildTracker(Solver<OT>* solver, Cache<OT>* cache, const ADataView& data,
        const typename Solver<OT>::Context& context, int max_depth, int max_num_nodes, typename Solver<OT>::SolType UB) {
        UB = std::max(OT::best, UB);
        return AbstractTracker<OT>::CreateTracker(solver, cache, data, context, max_depth, max_num_nodes, UB);
    }

    template <class OT>
    void SplitTracker<OT>::Erase() {
        using SolutionTrackerP = typename SplitTracker<OT>::SolutionTrackerP;
        //for (int i = 0; i < context.GetBranch().Depth(); i++) std::cout << " ";
        //std::cout << " Erasing ST " << context.GetBranch() << " - " << feature << std::endl;
        data.Clear();
        heap = std::priority_queue<HeapEntry, std::vector<HeapEntry>, CompareHeapEntry>();
        heap_buffer = std::vector<HeapEntry>();
        visited_heap_indices.Clear();
        solver->AddToQueueStats(context.GetBranch().Depth(),feature,solutions.size());
        solutions = std::vector<SolutionTrackerP>();
        next_solution_value = OT::worst;
        left_tracker = nullptr;
        right_tracker = nullptr;
    }

    template <class OT>
    bool SplitTracker<OT>::SwitchToAlternativeSolution(SolutionTrackerP left_sol, SolutionTrackerP right_sol) {
        if (left_sol->GetAltLabel() != OT::worst_label) {
            left_sol->SwitchLabel();
            return true;
        } else if (right_sol->GetAltLabel() != OT::worst_label) {
            right_sol->SwitchLabel();
            return true;
        } else {
            return false;
        }
    }

    template struct SplitTracker<CostComplexAccuracy>;
    template struct SplitTracker<CostComplexRegression>;


    template std::shared_ptr<AbstractTracker<CostComplexAccuracy>> AbstractTracker<CostComplexAccuracy>::CreateTracker(
        Solver<CostComplexAccuracy>*, Cache<CostComplexAccuracy>*, const ADataView&, const Solver<CostComplexAccuracy>::Context&, int, int, CostComplexAccuracy::SolType);

    template std::shared_ptr<AbstractTracker<CostComplexRegression>> AbstractTracker<CostComplexRegression>::CreateTracker(
        Solver<CostComplexRegression>*, Cache<CostComplexRegression>*, const ADataView&, const Solver<CostComplexRegression>::Context&, int, int, CostComplexRegression::SolType);
}
