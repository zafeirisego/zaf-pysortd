#include "rashomon/terminal_tracker.h"

namespace SORTD {

    template<class OT>
    TerminalTracker<OT>::TerminalTracker(Solver<OT> *solver, Cache<OT> *cache, const ADataView &_data,
                                         const typename Solver<OT>::Context &context, int _max_depth,
                                         int _max_num_nodes, typename Solver<OT>::SolType _UB)
            : solver(solver), cache(cache), data(_data), context(context), max_depth(_max_depth),
              max_num_nodes(_max_num_nodes), UB(_UB) {
        using SolutionTracker = typename Solver<OT>::SolutionTracker;
        using SolContainer = typename Solver<OT>::SolContainer;
        using SolType = typename Solver<OT>::SolType;

        // Obtain the solutions from the cache, if existing, and the upper bound of the cached solutions is at least as high as the current UB
        auto &branch = context.GetBranch();
        auto cached_solutions = cache->RetrieveBranchTracker(data, branch, max_depth, max_num_nodes);
        if (!cached_solutions.IsEmpty()) {
            solutions = cached_solutions.solutions;
            if (UB <= cached_solutions.UB) {
                runtime_assert(!solutions->empty());
                current_UB = cached_solutions.UB;
                UB_min_update = current_UB;
                next_solution_value = solutions->empty() ? OT::worst : solutions->at(0)->obj;
                return;
            } else if (!solutions->empty()) {
                current_UB = cached_solutions.UB;
                UB_min_update = current_UB;
                next_solution_value = solutions->at(0)->obj;
                return;
            }
        }

        // If using the incremental Rashomon bound, set the minimum Rashomon bound value based on the 
        // leaf solution + the branching costs, otherwise, set to the actual Rashomon bound
        if (solver->UseIncrementalRashomonBound()) {
            SolContainer tempUB = InitializeSol<OT>();
            auto leaf_sol = solver->SolveLeafNode(data, context, tempUB);
            SolType branching_costs = solver->GetBranchingCosts(data, context, 0);
            // Set the minimum UB at least to the cost of a leaf solution + the branching costs. This means at least all depth-one solutions are direclty computed
            // (This minimum UB is heuristically set based on some tests)
            current_UB = std::min(UB + SOLUTION_PRECISION, double(leaf_sol.solution + branching_costs));
            UB_min_update = current_UB;
        } else {
            current_UB = UB + SOLUTION_PRECISION;
        }

        // Find at least one solution up to current_UB using the Rashomon Terminal solver
        // If no solution found, increment the current_UB till one is found
        TerminalCallIterativeUB(0);

        // This TerminalTracker should only be constructed, if the parent SplitTracker claims there is at least one solution
        runtime_assert(!solutions->empty());

        next_solution_value = solutions->empty() ? OT::worst : solutions->at(0)->obj;

        // Store the solutions (based on the UB) in the cache
        auto entry = BranchTrackerCacheEntry<OT>(solutions, current_UB);
        cache->UpdateBranchTracker(data, branch, max_depth, max_num_nodes, entry);
    }

    template<class OT>
    void TerminalTracker<OT>::TerminalCallIterativeUB(size_t required_n) {

        // Check if the cache already has a list of solutions, if the required amount: done
        auto cached_solutions = cache->RetrieveBranchTracker(data, context.GetBranch(), max_depth, max_num_nodes);
        if (!cached_solutions.IsEmpty()) {
            if (current_UB < cached_solutions.UB) {
                solutions = cached_solutions.solutions;
                current_UB = cached_solutions.UB;
                if (solutions->size() >= required_n + 1) return;
            }
        }

        // Repeatedly call the RashomonSolverTerminalNode as long as the current_UB does not provide the 
        // required amount of solutions. Increment the currentUB with the base step size, or with half the distance 
        // to the actual Rashomon bound. The reason for the decreasing step size, is that the number of solutions increases exponentially,
        // as the bound is increased.
        while (!solutions || current_UB < UB) {
            double delta = std::min(double(UB_min_update), double((UB + SOLUTION_PRECISION) - current_UB));
            if (delta < 0.2 * UB) {
                current_UB = (UB + SOLUTION_PRECISION);
            } else if (solutions) {
                current_UB += 0.5 * delta;
            }
            auto UB_container = SolContainer(current_UB);
            auto results = solver->SolveRashomonTerminalNode(data, context, UB_container, max_depth, max_num_nodes);
            solutions = results.first;
            if (solutions->size() > required_n) break;
        }

        // Sort the solutions, and set the next_solution_value to the start
        std::sort(solutions->begin(), solutions->end(), CompareSolutionTrackers<OT>());

        auto entry = BranchTrackerCacheEntry<OT>(solutions, current_UB);
        cache->UpdateBranchTracker(data, context.GetBranch(), max_depth, max_num_nodes, entry);
        }

    template<class OT>
    void TerminalTracker<OT>::Pop() {
        if (!initialized) {
            current_index = 0;
            initialized = true;
        } else {
            current_index++;
        }
        if (current_index + 1 < solutions->size()) {
            next_solution_value = GetSolutions()[current_index + 1]->obj;
        } else if (current_UB < UB) {
            TerminalCallIterativeUB(current_index + 1);

            next_solution_value = current_index + 1 < solutions->size()
                                  ? GetSolutions()[current_index + 1]->obj
                                  : OT::worst;
        } else {
            next_solution_value = OT::worst;
        }
    }

    template
    struct TerminalTracker<CostComplexAccuracy>;
    template
    struct TerminalTracker<CostComplexRegression>;
}