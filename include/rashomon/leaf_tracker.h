#ifndef SORTD_RASHOMON_LEAF_TRACKER_H
#define SORTD_RASHOMON_LEAF_TRACKER_H

#include "rashomon/tracker.h"
#include "solver/solver.h"


namespace SORTD {

    /*
	* LeafTracker stores the solution of the subtree's leaf solution
	*/
    template<class OT>
    struct LeafTracker : public AbstractTracker<OT> {
        using SolType = typename Solver<OT>::SolType;
        using SolContainer = typename Solver<OT>::SolContainer;
        using Context = typename Solver<OT>::Context;
        using SolutionTracker = typename Solver<OT>::SolutionTracker;
        using SolutionTrackerP = std::shared_ptr<SolutionTracker>;
        using SolLabelType = typename OT::SolLabelType;

        LeafTracker() = delete;
        LeafTracker(Solver<OT>* solver, const ADataView& data, const Context& context, SolType UB) : solver(solver) {
            auto leaf_sol = InitializeSol<OT>();
            SolLabelType alt_label = OT::worst_label;
            if constexpr (OT::custom_leaf) {
                leaf_sol = solver->GetTask()->SolveLeafNode(data, context);
            } else {
                for (int label = 0; label < data.NumLabels(); label++) {
                    auto sol = Node<OT>(label, solver->GetTask()->GetLeafCosts(data, context, label));
                    if (!solver->SatisfiesConstraint(sol, context)) continue;
                    if (sol.solution + SOLUTION_PRECISION < leaf_sol.solution){
                        leaf_sol = sol;
                        alt_label = OT::worst_label;
                    } else if (std::abs(leaf_sol.solution - sol.solution) < SOLUTION_PRECISION) {
                        alt_label = sol.label;
                    }
                }
            }

            if (!LeftStrictDominatesRight<OT>(UB + SOLUTION_PRECISION, leaf_sol.solution)) {
                auto sol = std::make_shared<LeafSolutionTracker<OT>>(leaf_sol.solution, leaf_sol.label);
                if (alt_label != OT::worst_label) {
                    sol->SetAltLabel(alt_label);
                }
                solutions.emplace_back(sol);

                next_solution_value = leaf_sol.solution;
            } else {
                next_solution_value = OT::worst;
            }
        }

        void Pop() {
            next_solution_value = OT::worst;
        }

        inline bool HasNext() { return next_solution_value < OT::worst; }

        inline SolType GetNextSolutionValue() {
            return next_solution_value;
        }

        inline SolutionTrackerP GetLastUnusedSolution() { 
            return solutions[0];
        }

        inline SolutionTrackerP& GetSolutionN(size_t n) {
            runtime_assert(this->HasNSolution(n));
            return solutions[n];
        }

        inline const std::vector<SolutionTrackerP>& GetSolutions() const { return solutions; }

        bool IsSolutionListComplete() const { return true; }

        Solver<OT>* solver;

        SolType next_solution_value;
        std::vector<SolutionTrackerP> solutions;
    };

}

#endif