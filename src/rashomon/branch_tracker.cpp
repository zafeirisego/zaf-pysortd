
#include "rashomon/branch_tracker.h"
#include "rashomon/split_tracker.h"
#include "rashomon/rashomon_terminal_solver.h"
#include "rashomon/rashomon_utils.h"

namespace SORTD {

    template<class OT>
    BranchTracker<OT>::BranchTracker(Solver<OT>* solver, Cache<OT>* cache, const ADataView& _data, const typename Solver<OT>::Context& context, int _max_depth, int _max_num_nodes, typename BranchTracker<OT>::SolType _UB) :
    solver(solver), cache(cache), data(_data), context(context), max_depth(_max_depth), max_num_nodes(_max_num_nodes), UB(_UB), num_features(_data.NumFeatures()),
    leaf_tracker(solver, _data, context, _UB) {
      // Initialize only the optimal value 
        optimal_solution_value = FindInitialSolutionValue();
        solutions = std::make_shared<std::vector<std::shared_ptr<SolutionTracker>>>();
        // Fix a rounding issue if the UB and optimal solution are close
        if (_UB < optimal_solution_value) {
            runtime_assert(std::abs(_UB - optimal_solution_value) <= SOLUTION_PRECISION);
            UB = optimal_solution_value;
        }
    }

    template <class OT>
    typename BranchTracker<OT>::SolType BranchTracker<OT>::FindInitialSolutionValue() {
        using SolContainer = typename Solver<OT>::SolContainer;
        using SolType = typename Solver<OT>::SolType;
        SolContainer opt_solution;
        if (cache->IsOptimalAssignmentCached(data, context.GetBranch(), max_depth, max_num_nodes)) {
            opt_solution = cache->RetrieveOptimalAssignment(data, context.GetBranch(), max_depth, max_num_nodes);
        } else {
            opt_solution = solver->SolveSubTreeIncrementalUB(data, context, UB, max_depth, max_num_nodes);
        }
        
        return opt_solution.solution;
    }

    template <class OT>
    void BranchTracker<OT>::Initialize() {
        initialized = true;
        auto& branch = context.GetBranch();
        // Check if we already encountered this subproblem before, and if so, retrieve it from the cache
        auto cached_solutions = cache->RetrieveBranchTracker(data, branch, max_depth, max_num_nodes);
        if (!cached_solutions.IsEmpty() && UB + SOLUTION_PRECISION <= cached_solutions.UB) {
            solutions = cached_solutions.solutions;
            split_trackers = cached_solutions.split_trackers;
            best_next_split = cached_solutions.best_next_split;
            return;
        }

        solutions = std::make_shared<std::vector<std::shared_ptr<SolutionTracker>>>();
        split_trackers = std::make_shared<std::vector<SplitTracker<OT>>>();
        split_trackers->reserve(num_features);
        best_next_split = std::make_shared<std::priority_queue<AbstractTracker<OT>*, std::vector<AbstractTracker<OT>*>, TrackerCompare<OT>>>();
        
        // Construct a vector of SplitTrackers, one for each feature
        for (int f = 0; f < num_features; ++f) {
            // Skip features that have already been branched on. Check max_depth for debugging without the terminal solver
            if (max_depth == 0 || branch.HasBranchedOnFeature(f))
                split_trackers->push_back(SplitTracker<OT>());
            else
                split_trackers->push_back(SplitTracker<OT>(solver, cache, data, context, max_depth, max_num_nodes, num_features, UB, f));
        }

        // Add the split trackers to the priority queue, provided they have at least one solution within the bound
        for (auto& split_tracker : *split_trackers) {
            if (!split_tracker.HasNext()) continue;
            if (branch.HasBranchedOnFeature(split_tracker.feature)) continue;
            if (split_tracker.GetNextSolutionValue() > UB + SOLUTION_PRECISION) continue; 
            best_next_split->push(&split_tracker);
        }
        // Add the leaf tracker to the priority queue, provided it has a solution within the bound
        if (leaf_tracker.GetNextSolutionValue() <= UB + SOLUTION_PRECISION) { 
            best_next_split->push(&leaf_tracker);
        }
        runtime_assert(!best_next_split->empty());

        // Store the attributes of this tracker in the cache
        auto entry = BranchTrackerCacheEntry<OT>(solutions, split_trackers, best_next_split, UB);
        cache->UpdateBranchTracker(data, branch, max_depth, max_num_nodes, entry);
    }

    template<class OT>
    AbstractTracker<OT>* BranchTracker<OT>::GetBestNextTracker() {
        if (AreSplitsExhausted()) return nullptr;
        return best_next_split->top();
    }

    template <class OT>
    typename BranchTracker<OT>::SolType BranchTracker<OT>::GetNextSolutionValue() {
        // If no solutions are constructed yet, the optimal solution is the next solution value
        if (solutions->empty()) return optimal_solution_value;
        // If solutions are constructed, and the splits are exhausted, then return worst
        if (AreSplitsExhausted()) return OT::worst;
        // Otherwise, return the value of the best next tracker
        return GetBestNextTracker()->GetNextSolutionValue();
    }

    template<class OT>
    void BranchTracker<OT>::Pop() {
        // If not initialized, initialize the split trackers (or retrieve them from the cache
        if (!initialized) {
            Initialize();
            // If we retrieve a set of solutions from the cache, then there is no need to pop a new solution
            if (!solutions->empty()) return;
        }
        
        // Pop should only be called if this tracker has a next solution
        runtime_assert(HasNext() && !AreSplitsExhausted());
        if (!(HasNext() && !AreSplitsExhausted())) {
            std::cout << "Error: Branch tracker Pop can only be called if it has a next solution." << std::endl;
            std::exit(1);
        }

        // Find the split/leaf tracker with the best next solution
        AbstractTracker<OT>* best_next_tracker = GetBestNextTracker();        
#ifdef DEBUG
        auto promised_solution_value = best_next_tracker->GetNextSolutionValue();
#endif

        // Pop it from that tracker and add it to the constructed solution list
        best_next_tracker->Pop();
#ifdef DEBUG
        auto actual_value = best_next_tracker->GetLastUnusedSolution()->obj;
        runtime_assert(std::abs(promised_solution_value - actual_value) <= SOLUTION_PRECISION);
#endif
        EmplaceSolution(best_next_tracker->GetLastUnusedSolution());

        // Pop the tracker from the priority-queue and re-insert it if it 
        // still has solutions left, otherwise, erase it
        best_next_split->pop();
        if (best_next_tracker->HasNext()) {
            best_next_split->push(best_next_tracker);
        } else {
            best_next_tracker->Erase();
        }

#ifdef DEBUG
        // Check if the next best solution is worse than the currently popped solution
        if (!AreSplitsExhausted()) {
            auto new_best_next_value = GetBestNextTracker()->GetNextSolutionValue();
            runtime_assert(new_best_next_value + 2 * SOLUTION_PRECISION > promised_solution_value);
        }
#endif
    }


    template struct BranchTracker<CostComplexAccuracy>;
    template struct BranchTracker<CostComplexRegression>;
    template struct BranchTracker<AverageDepthAccuracy>;

}
