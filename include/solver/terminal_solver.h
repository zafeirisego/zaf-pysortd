/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/
#pragma once
#include "base.h"
#include "model/container.h"
#include "solver/cost_storage.h"
#include "solver/cost_combiner.h"
#include "model/branch.h"
#include "solver/tree.h"
#include "solver/counter.h"
#include "solver/optimization_utils.h"
#include "tasks/tasks.h"

namespace SORTD {

	template <class OT>
	struct TerminalResults {
		using SolType = typename OT::SolType;
		using SolContainer = Node<OT>;

		TerminalResults() { Clear(); }
		void Clear() {
			one_node_solutions = InitializeSol<OT>();
			two_nodes_solutions = InitializeSol<OT>();
			three_nodes_solutions = InitializeSol<OT>();
		}
		
		SolContainer one_node_solutions, two_nodes_solutions, three_nodes_solutions;
	};

	template <class OT>
	class Solver;

	struct SolverParameters;
	
	struct LabelAssignment {
		int left_label{ 0 };
		int right_label{ 0 };
	};

	template <class OT>
	class TerminalSolver {
	public:
		using SolType = typename OT::SolType;
		using SolContainer = Node<OT>;
		using Context = typename OT::ContextType;
		
		TerminalSolver(Solver<OT>* solver);

		TerminalResults<OT>& Solve(const ADataView& data, const Context& context, SolContainer& UB, SolType rash_bound_delta, int num_nodes);
		std::shared_ptr<Tree<OT>> ConstructOptimalTree(const Node<OT>& node, const ADataView& data, const Context& context, int max_depth, int num_nodes);
		inline int ProbeDifference(const ADataView& data) const { return cost_calculator.ProbeDifference(data); }
	
	private:

		struct ChildrenInformation {
			ChildrenInformation() {
				Clear();
			}
			inline void Clear() {
				left_child_assignments = InitializeSol<OT>();
				right_child_assignments = InitializeSol<OT>();
			}
			SolContainer left_child_assignments, right_child_assignments;
			Context left_context, right_context;
		};

		void SolveOneNode(const ADataView& data, const Context& context, bool initialized);
		void InitialiseChildrenInfo(const Context& context, const ADataView& data);
		void UpdateBestLeftChild(ChildrenInformation& child_info, const SolType& solution);
		void UpdateBestRightChild(ChildrenInformation& child_info, const SolType& solution);
		void UpdateBestTwoNodeAssignment(const Context& context, int root_feature);
		void UpdateBestThreeNodeAssignment(const Context& context, int root_feature);

		bool SatisfiesConstraint(const Node<OT>& sol, const Context& context) const;

		std::vector<ChildrenInformation> children_info;
		CostCalculator<OT> cost_calculator;
		TerminalResults<OT> results;
		OT* task;
		const SolverParameters* solver_parameters;
		int num_features, num_labels;
		SolContainer UB;
		SolType rash_bound_delta;

		Node<OT> temp_branch_node, temp_leaf_node;
		std::vector<Sols<OT>> sols;
		std::vector<LabelAssignment> label_assignments;
	};
}
