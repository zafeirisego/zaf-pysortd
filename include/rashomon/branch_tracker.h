

#ifndef SORTD_RASHOMON_BRANCH_TRACKER_H
#define SORTD_RASHOMON_BRANCH_TRACKER_H

#include "base.h"
#include <bitset>
#include "solver/solver.h"
#include "tasks/tasks.h"
#include "rashomon/tracker.h"
#include "rashomon/leaf_tracker.h"
#include "rashomon/solution_tracker.h"
#include "rashomon/solution_value_heap.h"

namespace SORTD {

    template <class OT>
    struct TrackerCompare;

    template <class OT>
    struct SplitTracker;

    /*
	* BranchTracker maintains a sorted set of solutions for the subtree
	*/
    template <class OT>
    struct BranchTracker : public AbstractTracker<OT> {
        using SolType = typename Solver<OT>::SolType;
        using SolContainer = typename Solver<OT>::SolContainer;
        using Context = typename Solver<OT>::Context;
        using SolutionTracker = typename Solver<OT>::SolutionTracker;
        using SolutionTrackerP = std::shared_ptr<SolutionTracker>;
        using SolutionList = std::shared_ptr<std::vector<std::shared_ptr<SolutionTracker>>>;
        using SplitTrackerList = std::shared_ptr<std::vector<SplitTracker<OT>>>;
        using TrackerPriorityQueue = std::shared_ptr<std::priority_queue<AbstractTracker<OT>*, std::vector<AbstractTracker<OT>*>, TrackerCompare<OT>>>;
        SolType worst = OT::worst;

        BranchTracker() = delete;
        BranchTracker(Solver<OT>* solver, Cache<OT>* cache, const ADataView& _data, const Context& context, int _max_depth, int _max_num_nodes, SolType _UB);

        // Retrieve the optimal solution of the search node using the cache.
        SolType FindInitialSolutionValue();

        // Return the best solution value of its split trackers and leaf tracker.
        SolType GetNextSolutionValue();

        inline const std::vector<SolutionTrackerP>& GetSolutions() const { return *solutions; }

        // Initialize the split trackers
        void Initialize();

        inline bool HasNext() { return GetNextSolutionValue() < OT::worst; }

        inline void EmplaceSolution(SolutionTrackerP solution) {
#ifdef DEBUG
            if (!solutions->empty()) {
                auto last_solution = solutions->at(solutions->size() - 1);
                runtime_assert(last_solution->obj <= solution->obj + 2 * SOLUTION_PRECISION);
            }
#endif
            solution->Shrink();
            solutions->push_back(solution);
        }

        inline SolutionTrackerP GetLastUnusedSolution() {
            // TODO implement
            return std::make_shared<LeafSolutionTracker<OT>>(worst, OT::worst_label);
        }

        inline bool AreSplitsExhausted() const {
            return best_next_split->empty();
        }

        AbstractTracker<OT>* GetBestNextTracker();

        void Pop();

        int num_features;
        Solver<OT>* solver;
        Cache<OT>* cache;
        ADataView data;
        Context context;
        int max_depth;
        int max_num_nodes;
        SolType UB;
        bool initialized = false;

        SolType optimal_solution_value;
        
        SolutionList solutions = nullptr;
        SplitTrackerList split_trackers = nullptr;
        LeafTracker<OT> leaf_tracker;
        
        TrackerPriorityQueue best_next_split;
    };

    

}


#endif //SORTD_RASHOMON_BRANCH_TRACKER_H
