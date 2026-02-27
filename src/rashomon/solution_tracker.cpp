#include "rashomon/solution_tracker.h"
#include "solver/solver.h"

#ifdef WITH_PYBIND
#include <pybind11/pybind11.h>
#endif

namespace SORTD {

	template <class OT, class SVT>
	void RecursiveSolutionTracker<OT, SVT>::CalculateFeatureStats() {
		feature_count.assign(feature_count.size(), 0);
		for (int i = 0; i < feature_count.size(); ++i) {
			if (i == feature) {
				feature_count[i] = num_solutions;
				continue;
			}
			for (auto& sol : solutions) {
				size_t n_left_features = sol.first->GetFeatureCount(i);
				size_t n_right_features = sol.second->GetFeatureCount(i);
				feature_count[i] += (n_left_features * sol.second->GetNumSolutions()) + ((sol.first->GetNumSolutions() - n_left_features) * n_right_features);
			}
		}
	}

	template <class OT, class SVT>
	void RecursiveSolutionTracker<OT, SVT>::CalculateNodeStats() {

		//            if (obj == 52.830000000000005 && feature == 4) {
		//                int a = 1;
		//            }
		num_nodes_count.clear();
		for (auto& sol : solutions) {

			if (sol.first->IsLeaf() && sol.second->IsLeaf()) {
				UpdateNodeCount(1, 1);
			} else if (sol.first->IsLeaf()) {
				auto num_nodes_count = sol.second->GetNodeCounts();
				for (int i = 0; i < num_nodes_count.size(); i++) {
					UpdateNodeCount(i + 1, num_nodes_count[i]);
				}
			} else if (sol.second->IsLeaf()) {
				auto num_nodes_count = sol.first->GetNodeCounts();
				for (int i = 0; i < num_nodes_count.size(); i++) {
					UpdateNodeCount(i + 1, num_nodes_count[i]);
				}
			} else {
				auto num_nodes_count1 = sol.first->GetNodeCounts();
				auto num_nodes_count2 = sol.second->GetNodeCounts();
				for (int i = 0; i < num_nodes_count1.size(); i++) {
					for (int j = 0; j < num_nodes_count2.size(); j++) {
						size_t num_sols_left = num_nodes_count1[i];
						size_t num_sols_right = num_nodes_count2[j];
						UpdateNodeCount(i + j + 1, num_sols_left * num_sols_right);
					}
				}
			}


		}
	}

	template <class OT, class SVT>
	std::shared_ptr<Tree<OT>> RecursiveSolutionTracker<OT, SVT>::CreateTreeWithIndexN(const Solver<OT>* solver, size_t n) const {
		//            runtime_assert((n == 0 && cumulative_sol_count.empty()) || cumulative_sol_count.back() >= n);

		//            std::cout << "feature: " << feature << std::endl;
		//            std::cout << "num_solutions: " << num_solutions << std::endl;

		//            for (int i = 0; i <cumulative_sol_count.size(); ++i){
		//                std::cout << "cumulative_sol_count[i]: " << cumulative_sol_count[i] << std::endl;
		//            }
		runtime_assert(feature >= 0);
		runtime_assert(!solutions.empty());
		/*if (feature == -1) {
			return Tree<OT>::CreateLabelNode(label);
		} else if (solutions.empty()) {
			auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature);
			tree->left_child = Tree<OT>::CreateLabelNode(left_label);
			tree->right_child = Tree<OT>::CreateLabelNode(right_label);
			return tree;
		}*/

		size_t solution_index;
		if (cumulative_sol_count.empty()) { //When the solutions are terminal solver solutions
//                std::cout << "cumulative_sol_count is empty" << std::endl;
			solution_index = n;
			n = 0;
		} else {
			auto it = std::lower_bound(cumulative_sol_count.begin(), cumulative_sol_count.end(), size_t(n));
			solution_index = std::distance(cumulative_sol_count.begin(), it);
			if (solution_index != 0)  n -= cumulative_sol_count[solution_index - 1] + 1;
		}

		//            std::cout << "solution_index: " << solution_index << std::endl;
		//            std::cout << "n: " << n << std::endl;

		auto solution_pair = solutions[solution_index];
		size_t n_left_sols = solution_pair.first->GetNumSolutions();
		size_t left_index = n % n_left_sols;
		size_t right_index = n / n_left_sols;

		//            std::cout << "left_index: " << left_index << std::endl;
		//            std::cout << "right_index: " << right_index << std::endl;

		auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature);
		auto left_tree = solution_pair.first->CreateTreeWithIndexN(solver, left_index);
		auto right_tree = solution_pair.second->CreateTreeWithIndexN(solver, right_index);
		tree->left_child = left_tree;
		tree->right_child = right_tree;
		if (solver->IsFeatureFlipped(feature)) {
			std::swap(tree->left_child, tree->right_child);
		}
		return tree;
	}

	template <class OT, class SVT>
	size_t RecursiveSolutionTracker<OT, SVT>::NumberOfQueryFeatureF(int f) const {
		if (feature_count.empty() && feature == f) {
			return num_solutions;
		} else if (!feature_count.empty() && feature_count[f] > 0) {
			return feature_count[f];
		} else {
			return 0;
		}
	}

	template <class OT, class SVT>
	std::vector<std::shared_ptr<Tree<OT>>> RecursiveSolutionTracker<OT, SVT>::CreateQueryTreesWithFeature(const Solver<OT>* solver, int query_feature) {
		std::vector<std::shared_ptr<Tree<OT>>> trees;
		runtime_assert(feature >= 0);
		//if (feature == -1) return trees;
		/*if (feature == query_feature && left_label != OT::worst_label) {
			auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature);
			tree->left_child = Tree<OT>::CreateLabelNode(left_label);
			tree->right_child = Tree<OT>::CreateLabelNode(right_label);
			trees.push_back(tree);
			return trees;
		} else if (feature_count.empty() || (!feature_count.empty() && feature_count[feature] == 0)) return trees;
		else */
		if (feature == query_feature) {
			for (size_t i = 0; i < num_solutions; ++i) {
				trees.push_back(CreateTreeWithIndexN(solver, i));
			}
			return trees;
		}
		for (auto& sol : solutions) {
			if (!sol.first->HasQueryFeatureF(query_feature) && !sol.second->HasQueryFeatureF(query_feature)) continue;

			std::vector<std::shared_ptr<Tree<OT>>> left_trees_with_feature, right_trees_with_feature, left_trees_without_feature, all_right_trees;
			if (sol.first->GetNumSolutions() == 1 && sol.first->HasQueryFeatureF(query_feature)) left_trees_with_feature.push_back(sol.first->CreateTreeWithIndexN(solver, 0));
			else left_trees_with_feature = sol.first->CreateQueryTreesWithFeature(solver, query_feature);

			if (!left_trees_with_feature.empty()) {
				for (size_t i = 0; i < sol.second->GetNumSolutions(); ++i) all_right_trees.push_back(sol.second->CreateTreeWithIndexN(solver, i));
			}


			for (auto& left : left_trees_with_feature) {
				for (auto& right : all_right_trees) {
					auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature);
					tree->left_child = left;
					tree->right_child = right;
                    if (solver->IsFeatureFlipped(feature)) {
                        std::swap(tree->left_child, tree->right_child);
                    }
					trees.push_back(tree);
				}
			}

			left_trees_without_feature = sol.first->CreateQueryTreesWithoutFeature(solver, query_feature);
			if (!left_trees_without_feature.empty()) {
				if (sol.second->GetNumSolutions() == 1 && sol.second->HasQueryFeatureF(query_feature)) right_trees_with_feature.push_back(sol.second->CreateTreeWithIndexN(solver, 0));
				else right_trees_with_feature = sol.second->CreateQueryTreesWithFeature(solver, query_feature);
			}

			for (auto& left : left_trees_without_feature) {
				for (auto& right : right_trees_with_feature) {
					auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature);
					tree->left_child = left;
					tree->right_child = right;
                    if (solver->IsFeatureFlipped(feature)) {
                        std::swap(tree->left_child, tree->right_child);
                    }
					trees.push_back(tree);
				}
			}
		}
		return trees;
	}

	template <class OT, class SVT>
	std::vector<std::shared_ptr<Tree<OT>>> RecursiveSolutionTracker<OT, SVT>::CreateQueryTreesWithoutFeature(const Solver<OT>* solver, int query_feature) {
		std::vector<std::shared_ptr<Tree<OT>>> trees;
		runtime_assert(feature >= 0);
		//if (feature == -1) {
		//	trees.push_back(Tree<OT>::CreateLabelNode(label));
		//	return trees;
		//}
		if (feature == query_feature) return trees;
		/*if (feature != query_feature && left_label != OT::worst_label) {
			auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature);
			tree->left_child = Tree<OT>::CreateLabelNode(left_label);
			tree->right_child = Tree<OT>::CreateLabelNode(right_label);
			trees.push_back(tree);
			return trees;
		}*/
		for (auto& sol : solutions) {
			if (sol.first->NumberOfQueryFeatureF(query_feature) == sol.first->GetNumSolutions() ||
				sol.second->NumberOfQueryFeatureF(query_feature) == sol.second->GetNumSolutions()) continue;

			std::vector<std::shared_ptr<Tree<OT>>> left_trees, right_trees;
			if (sol.first->GetNumSolutions() == 1) left_trees.push_back(sol.first->CreateTreeWithIndexN(solver, 0));
			else left_trees = sol.first->CreateQueryTreesWithoutFeature(solver, query_feature);

			if (sol.second->GetNumSolutions() == 1) right_trees.push_back(sol.second->CreateTreeWithIndexN(solver, 0));
			else right_trees = sol.second->CreateQueryTreesWithoutFeature(solver, query_feature);

			for (auto& left : left_trees) {
				for (auto& right : right_trees) {
					auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature);
					tree->left_child = left;
					tree->right_child = right;
                    if (solver->IsFeatureFlipped(feature)) {
                        std::swap(tree->left_child, tree->right_child);
                    }
					trees.push_back(tree);
				}
			}
		}
		return trees;
	}

	template <class OT, class SVT>
	std::vector<std::shared_ptr<Tree<OT>>> RecursiveSolutionTracker<OT, SVT>::CreateNodeBudgetQueryTrees(const Solver<OT>* solver, int node_budget) {
		std::vector<std::shared_ptr<Tree<OT>>> trees;

		runtime_assert(feature >= 0);
		/*if (feature == -1 && node_budget == 0) {
			trees.push_back(Tree<OT>::CreateLabelNode(label));
			return trees;
		}*/
		/*if (feature != -1 && left_label != OT::worst_label && node_budget == 1) {
			auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature);
			tree->left_child = Tree<OT>::CreateLabelNode(left_label);
			tree->right_child = Tree<OT>::CreateLabelNode(right_label);
			trees.push_back(tree);
			return trees;
		}*/
		for (int partial_node_budget1 = 0; partial_node_budget1 < (node_budget / 2) + 1; ++partial_node_budget1) {
			int partial_node_budget2 = node_budget - partial_node_budget1 - 1;

			std::vector<std::pair<int, int>> budget_dists({ std::make_pair(partial_node_budget1,partial_node_budget2) });
			if (partial_node_budget1 != partial_node_budget2) budget_dists.push_back(std::make_pair(partial_node_budget2, partial_node_budget1));

			for (auto& sol : solutions) {

				for (auto budget_dist : budget_dists) {
					int left_budget = budget_dist.first;
					int right_budget = budget_dist.second;

					bool left_key_exists = sol.first->GetNodeCount(left_budget) > 0 || (sol.first->GetFeature() == -1 && left_budget == 0)
						|| (sol.first->GetFeature() != -1 && sol.first->IsD1() && left_budget == 1);
					bool right_key_exists = sol.second->GetNodeCount(right_budget) > 0 || (sol.second->GetFeature() == -1 && right_budget == 0)
						|| (sol.second->GetFeature() != -1 && sol.second->IsD1() && right_budget == 1);

					if (left_key_exists && right_key_exists) {

						std::vector<std::shared_ptr<Tree<OT>>> left_trees, right_trees;

						if (sol.first->GetNumSolutions() == 1) left_trees.push_back(sol.first->CreateTreeWithIndexN(solver, 0));
						else left_trees = sol.first->CreateNodeBudgetQueryTrees(solver, left_budget);

						if (sol.second->GetNumSolutions() == 1) right_trees.push_back(sol.second->CreateTreeWithIndexN(solver, 0));
						else right_trees = sol.second->CreateNodeBudgetQueryTrees(solver, right_budget);

						for (auto& left : left_trees) {
							for (auto& right : right_trees) {
								auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature);
								tree->left_child = left;
								tree->right_child = right;
                                if (solver->IsFeatureFlipped(feature)) {
                                    std::swap(tree->left_child, tree->right_child);
                                }
								trees.push_back(tree);
							}
						}
					}
				}
				if (trees.size() == GetNodeCount(node_budget)) return trees;
			}
			if (trees.size() == GetNodeCount(node_budget)) return trees;
		}
		return trees;
	}

	template struct RecursiveSolutionTracker<CostComplexAccuracy>;
	template struct RecursiveSolutionTracker<CostComplexRegression>;
	template struct RecursiveSolutionTracker<AverageDepthAccuracy>;

#ifdef WITH_PYBIND

	namespace py = pybind11;
	template struct RecursiveSolutionTracker<CostComplexAccuracy, py::object>;
	template struct RecursiveSolutionTracker<CostComplexRegression, py::object>;
	template struct RecursiveSolutionTracker<AverageDepthAccuracy, py::object>;

#endif

}
