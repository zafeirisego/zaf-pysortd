
#ifndef SORTD_RASHOMON_TERMINAL_TRACKER_H
#define SORTD_RASHOMON_TERMINAL_TRACKER_H

#include "rashomon/tracker.h"
#include "solver/solver.h"

namespace SORTD {

    /*
	* TerminalTracker maintains a sorted set of solutions for a depth-two subtree.
	*/
    template<class OT>
	struct TerminalTracker : AbstractTracker<OT> {
        using SolType = typename Solver<OT>::SolType;
        using SolContainer = typename Solver<OT>::SolContainer;
        using Context = typename Solver<OT>::Context;
        using SolutionTracker = typename Solver<OT>::SolutionTracker;
        using SolutionTrackerP = std::shared_ptr<SolutionTracker>;
        SolType worst = OT::worst;

        TerminalTracker() = delete;
        TerminalTracker(Solver<OT>* solver, Cache<OT>* cache, const ADataView& _data, const Context& context, int _max_depth, int _max_num_nodes, SolType _UB);

        // Prepare the next best solution value
        void Pop();

        bool HasNext() { return next_solution_value < OT::worst; }

        inline SolType GetNextSolutionValue() {
            return next_solution_value;
        }

        inline SolutionTrackerP GetLastUnusedSolution() {
            if (this->HasNSolution(current_index)) {
                return this->GetSolutionN(current_index);
            }
            return std::make_shared<LeafSolutionTracker<OT>>(worst, OT::worst_label);
        }

        inline const std::vector<SolutionTrackerP>& GetSolutions() const { return *solutions; }

        bool IsSolutionListComplete() const { return current_UB >= UB; }

        // Gradually increase the UB and retrieve the depth-two solutions within the bound.
        void TerminalCallIterativeUB(size_t required_n);

        Solver<OT>* solver;
        Cache<OT>* cache;
        ADataView data;
        Context context;
        int max_depth;
        int max_num_nodes;
        SolType UB;
        SolType current_UB;
        SolType UB_min_update{ 0 };

        size_t current_index = 0;
        
        bool initialized = false;
        
        SolType next_solution_value;
        std::shared_ptr<std::vector<SolutionTrackerP>> solutions = nullptr;
	};

}

#endif