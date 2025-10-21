/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree 
Partly from Jacobus G.M. van der Linden "DPF"
https://gitlab.tudelft.nl/jgmvanderlinde/dpf
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/
#pragma once
#include "base.h"
#include "solver/cost_storage.h"
#include "solver/cost_combiner.h"
#include "model/branch.h"
#include "solver/tree.h"
#include "solver/counter.h"
#include "solver/optimization_utils.h"
#include "tasks/tasks.h"
#include "rashomon/small_solution_cache.h"

namespace SORTD {

    template <class OT>
    struct RashomonTerminalResults {
        using SolType = typename OT::SolType;
        using SolContainer = Node<OT>;

        RashomonTerminalResults() { Clear(); }
        void Clear() {
            solutions = std::make_shared <std::vector<std::shared_ptr<AbstractSolutionTracker<OT>>>>();
            solutions_map = std::make_shared<std::vector<std::unordered_map<int64_t, std::shared_ptr<AbstractSolutionTracker<OT>>>>>();
            num_solutions = 0;
        }

        std::shared_ptr<std::vector<std::shared_ptr<AbstractSolutionTracker<OT>>>> solutions;
        std::shared_ptr<std::vector<std::unordered_map<int64_t, std::shared_ptr<AbstractSolutionTracker<OT>>>>> solutions_map;
        int num_solutions = 0;
    };

    template <class OT>
    class Solver;

    struct SolverParameters;

    struct RashomonLabelAssignment {
        int left_label{ 0 };
        int right_label{ 0 };
    };

    /*
	* RashomonTerminalSolver computes the depth-two solutions.
	*/

    template <class OT>
    class RashomonTerminalSolver {
    public:
        using SolType = typename OT::SolType;
        using SolContainer = Node<OT>;
        using Context = typename OT::ContextType;
        using SolLabelType = typename OT::SolLabelType;

        RashomonTerminalSolver(Solver<OT>* solver);

        RashomonTerminalResults<OT>& Solve(const ADataView& data, const Context& context, SolContainer& UB, int num_nodes);
        inline int ProbeDifference(const ADataView& data) const { return cost_calculator.ProbeDifference(data); }

    private:

        struct ChildrenInformation {
            ChildrenInformation() {
                Clear();
            }
            inline void Clear() {
                left_child_assignments = std::make_shared <Container<OT>>();
                right_child_assignments = std::make_shared <Container<OT>>();
                left_child_d1feature_assignments.clear();
                right_child_d1feature_assignments.clear();
                left_child_d1_labels.clear();
                right_child_d1_labels.clear();
            }
            std::shared_ptr<Container<OT>> left_child_assignments, right_child_assignments;
            std::vector<int> left_child_d1feature_assignments, right_child_d1feature_assignments = {};
            std::vector<std::pair<SolLabelType, SolLabelType>> left_child_d1_labels, right_child_d1_labels = {};
            Context left_context, right_context;
        };

        void SolveOneNode(const ADataView& data, const Context& context, bool initialized);
        void InitialiseChildrenInfo(const Context& context, const ADataView& data);
        void UpdateBestLeftChild(int root_feature, int feature, const SolType& solution, std::pair<SolLabelType, SolLabelType>& label);
        void UpdateBestRightChild(int root_feature, int feature, const SolType& solution, std::pair<SolLabelType, SolLabelType>& label);
        void UpdateBestTwoNodeAssignment(const Context& context, int root_feature);
        void UpdateBestThreeNodeAssignment(const Context& context, int root_feature);

        void Merge(int feature, const Context& context, std::shared_ptr<Container<OT>> left_solutions, std::shared_ptr<Container<OT>> right_solutions, std::vector<int>& left_d1_features, std::vector<int>& right_d1_features, std::vector<std::pair<SolLabelType, SolLabelType>>& left_d1_labels, std::vector<std::pair<SolLabelType, SolLabelType>>& right_d1_labels);
        bool SatisfiesConstraint(const Node<OT>& sol, const Context& context) const;

        SolutionTrackerP<OT> CreateLeafSolution(SolType solution_value, SolLabelType& label, SolLabelType alt_label);
        SolutionTrackerP<OT> CreateD1Solution(SolType solution_value, SolLabelType& label_left, SolLabelType& label_right, int feature);

        inline bool IsTrivialExtension(SolLabelType label_left, SolLabelType label_right);

        std::vector<ChildrenInformation> children_info;
        CostCalculator<OT> cost_calculator;
        RashomonTerminalResults<OT> results;
        OT* task;
        const SolverParameters* solver_parameters;
        SmallSolutionCache<OT>* small_solution_cache;
        int num_features, num_labels;
        SolContainer UB;

        Node<OT> temp_branch_node, temp_leaf_node;
        std::vector<Sols<OT>> sols;
        std::vector<Labels<OT>> labels;
        std::vector<RashomonLabelAssignment> label_assignments;
    };
}
