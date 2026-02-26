/**
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/


#include "rashomon/rashomon_terminal_solver.h"
#include "solver/solver.h"

namespace SORTD {

    template <class OT>
    RashomonTerminalSolver<OT>::RashomonTerminalSolver(Solver<OT>* solver) :
            task(solver->GetTask()), num_features(solver->NumFeatures()),
            cost_calculator(solver->GetTask(), solver->NumFeatures(), solver->NumLabels(), solver->GetFeatureOrder()),
            children_info(solver->NumFeatures()), solver_parameters(&(solver->GetSolverParameters())),
            temp_branch_node(INT32_MAX, OT::worst, 0, 0),
            temp_leaf_node(OT::worst_label, OT::worst),
            sols(solver->NumLabels()),
            labels(solver->NumLabels()),
            small_solution_cache(&(solver->GetCache()->GetSmallSolutionCache()))
    {
        num_labels = solver->NumLabels();
        for (int left_label = 0; left_label < num_labels; left_label++) {
            for (int right_label = 0; right_label < num_labels; right_label++) {
//                if (num_labels > 1 && left_label == right_label) continue;
                if(left_label == right_label) continue;
                label_assignments.push_back({ left_label, right_label });
            }
        }
        for (int left_label = 0; left_label < num_labels; left_label++) {
            label_assignments.push_back({ left_label, left_label });
        }
        results.solutions_map->resize(num_features+1);
    }


    template <class OT>
    RashomonTerminalResults<OT>& RashomonTerminalSolver<OT>::Solve(const ADataView& data, const Context& context, SolContainer& UB, int num_nodes) {
        bool changes_made = cost_calculator.Initialize(data, context, num_nodes);
        if (!changes_made && (UB.solution <= this->UB.solution)) return results;
        this->UB = UB;
        results.Clear();
        results.solutions_map->resize(num_features+1);
        if (cost_calculator.GetTotalCount() < solver_parameters->minimum_leaf_node_size) return results;
        InitialiseChildrenInfo(context, data);
        SolveOneNode(data, context, true);
        if (num_nodes == 1) {
            return results;
        }

        typename RashomonTerminalSolver<OT>::SolType left_sol;
        typename RashomonTerminalSolver<OT>::SolType right_sol;
        int  total0, total1;
        IndexInfo index;
        Counts counts;
        for (int f1 = 0; f1 < num_features; f1++) {
            if (!task->MayBranchOnFeature(f1)) continue;
            total0 = cost_calculator.GetCount00(f1, f1);
            total1 = cost_calculator.GetCount11(f1, f1);
            if (total0 < solver_parameters->minimum_leaf_node_size || total1 < solver_parameters->minimum_leaf_node_size) continue;

            for (int f2 = f1 + 1; f2 < num_features; f2++) {
                if (!task->MayBranchOnFeature(f2)) continue;
                if (f1 == f2) continue;

                cost_calculator.GetIndexInfo(f1, f2, index);
                cost_calculator.GetCounts(counts, index);
                runtime_assert(total0 == counts.count00 + counts.count01);
                runtime_assert(total1 == counts.count10 + counts.count11);

                if ((counts.count00 < solver_parameters->minimum_leaf_node_size || counts.count01 < solver_parameters->minimum_leaf_node_size)
                    && (counts.count10 < solver_parameters->minimum_leaf_node_size || counts.count11 < solver_parameters->minimum_leaf_node_size)
                    && (counts.count00 < solver_parameters->minimum_leaf_node_size || counts.count10 < solver_parameters->minimum_leaf_node_size)
                    && (counts.count01 < solver_parameters->minimum_leaf_node_size || counts.count11 < solver_parameters->minimum_leaf_node_size)) {
                    continue;
                }

                const auto branch_left_costs = cost_calculator.GetBranchingCosts0(counts.count00  + counts.count01, f1, f2);
                const auto branch_right_costs = cost_calculator.GetBranchingCosts1(counts.count10 + counts.count11 , f1, f2);

                const auto branch_left_costs_rev = cost_calculator.GetBranchingCosts0(counts.count00  + counts.count10 , f2, f1);
                const auto branch_right_costs_rev = cost_calculator.GetBranchingCosts1(counts.count01 + counts.count11, f2, f1);



                for (int label = 0; label < num_labels; label++) {
                    cost_calculator.CalcSols(counts, sols[label], labels[label], label, index);
                }

                // Find left children (first=0)
                if (counts.count00 >= solver_parameters->minimum_leaf_node_size && counts.count01 >= solver_parameters->minimum_leaf_node_size) {
                    typename RashomonTerminalSolver<OT>::SolType best_left_sol = OT::worst;
                    auto best_label = RashomonLabelAssignment();
                    for (auto& la : label_assignments) {
                        OT::Add(sols[la.left_label].sol00, sols[la.right_label].sol01, left_sol);
                        OT::Add(left_sol, branch_left_costs, left_sol);
                        if(best_left_sol > left_sol) {
                            best_left_sol = left_sol;
                            best_label = la;
                        }
                    }
                    auto best_pair = std::make_pair(labels[best_label.left_label].label00, labels[best_label.right_label].label01);
                    UpdateBestLeftChild(f1, f2, best_left_sol,best_pair);

                }

                // Find right children (first=1)
                if (counts.count10 >= solver_parameters->minimum_leaf_node_size && counts.count11 >= solver_parameters->minimum_leaf_node_size) {
                    typename RashomonTerminalSolver<OT>::SolType best_right_sol = OT::worst;
                    auto best_label = RashomonLabelAssignment();
                    for (auto& la : label_assignments) {
                        OT::Add(sols[la.left_label].sol10, sols[la.right_label].sol11, right_sol);
                        OT::Add(right_sol, branch_right_costs, right_sol);
                        if(best_right_sol > right_sol) {
                            best_right_sol = right_sol;
                            best_label = la;
                        }
                    }
                    auto best_pair = std::make_pair(labels[best_label.left_label].label10, labels[best_label.right_label].label11);
                    UpdateBestRightChild(f1, f2, best_right_sol,best_pair);
                }

                // Find left children (rev, first=0)
                if (counts.count00 >= solver_parameters->minimum_leaf_node_size && counts.count10 >= solver_parameters->minimum_leaf_node_size) {
                    typename RashomonTerminalSolver<OT>::SolType best_left_sol = OT::worst;
                    auto best_label = RashomonLabelAssignment();
                    for (auto& la : label_assignments) {
                        OT::Add(sols[la.left_label].sol00, sols[la.right_label].sol10, left_sol);
                        OT::Add(left_sol, branch_left_costs_rev, left_sol);
                        if(best_left_sol > left_sol) {
                            best_left_sol = left_sol;
                            best_label = la;
                        }
                    }
                    auto best_pair = std::make_pair(labels[best_label.left_label].label00, labels[best_label.right_label].label10);
                    UpdateBestLeftChild(f2, f1, best_left_sol,best_pair);
                }

                // Find right children (rev, first=1)
                if (counts.count01 >= solver_parameters->minimum_leaf_node_size && counts.count11 >= solver_parameters->minimum_leaf_node_size) {
                    typename RashomonTerminalSolver<OT>::SolType best_right_sol = OT::worst;
                    auto best_label = RashomonLabelAssignment();
                    for (auto& la : label_assignments) {
                        OT::Add(sols[la.left_label].sol01, sols[la.right_label].sol11, right_sol);
                        OT::Add(right_sol, branch_right_costs_rev, right_sol);
                        if(best_right_sol > right_sol) {
                            best_right_sol = right_sol;
                            best_label = la;
                        }
                    }
                    auto best_pair = std::make_pair(labels[best_label.left_label].label01, labels[best_label.right_label].label11);
                    UpdateBestRightChild(f2, f1, best_right_sol,best_pair);
                }

            }

            UpdateBestTwoNodeAssignment(context, f1);
            UpdateBestThreeNodeAssignment(context, f1);

        }

//        AddSubOptimalSols<OT>(results.two_nodes_solutions, results.one_node_solutions, UB);
//        AddSubOptimalSols<OT>(results.three_nodes_solutions, results.two_nodes_solutions, UB);
        return results;
    }

    template <class OT>
    void RashomonTerminalSolver<OT>::SolveOneNode(const ADataView& data, const typename RashomonTerminalSolver<OT>::Context& context, bool initialized) {
        runtime_assert(initialized); // for now
//        auto& result = results.one_node_solutions;
//        auto& trees = results.one_node_trees;

        Node<OT> node;
        typename RashomonTerminalSolver<OT>::SolType merged_sol;
        // Add leaf nodes
        //if constexpr (OT::custom_leaf) {
        //
        //	AddSubOptimalSol<OT>(result, task->SolveLeafNode(data, context));
        //} else
        {
            typename OT::SolLabelType out_label;
            Node<OT> best_label_assignment = Node<OT>();
            SolLabelType alt_label = OT::worst_label;
            for (int label = 0; label < data.NumLabels(); label++) {
                //node = Node<OT>(label, task->GetLeafCosts(data, context, label));
                cost_calculator.CalcLeafSol(merged_sol, label, out_label, context);
                node.Set(INT32_MAX, out_label, merged_sol, 0, 0);

//                if (OT::has_constraint && !SatisfiesConstraint(node, context)) continue;
//                if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, node)) continue;
                if (node.solution + SOLUTION_PRECISION < best_label_assignment.solution) {
                    best_label_assignment = node;
                    alt_label = OT::worst_label;
                } else if (std::abs(best_label_assignment.solution - node.solution) < SOLUTION_PRECISION) {
                    alt_label = node.label;
                }
            }

            if (best_label_assignment.solution <= UB.solution) {
                auto new_sol = CreateLeafSolution(best_label_assignment.solution, best_label_assignment.label, alt_label);
                runtime_assert(new_sol->GetLabel() == best_label_assignment.label);
                runtime_assert(new_sol->GetAltLabel() == alt_label);
                runtime_assert(best_label_assignment.label != alt_label);
                results.solutions->push_back(new_sol);
                (*results.solutions_map)[num_features][STOI(typename OT::SolType, node.solution)] = new_sol;
                ++results.num_solutions;
            }
        }

        bool computed_leaves = false;

        if (initialized) {
            Counts counts;
            IndexInfo index;
            for (int feature = 0; feature < num_features; feature++) {
                if (!task->MayBranchOnFeature(feature)) continue;
                cost_calculator.GetIndexInfo(feature, feature, index);
                cost_calculator.GetCounts(counts, index);
                runtime_assert(counts.count00 + counts.count11 >= solver_parameters->minimum_leaf_node_size); // If even a leaf node is too small, D2-solver should not be called
                if (counts.count00 < solver_parameters->minimum_leaf_node_size || counts.count11 < solver_parameters->minimum_leaf_node_size) continue;
                for (int label = 0; label < num_labels; label++) {
                    cost_calculator.CalcSols(counts, sols[label], labels[label], label, index);
                }
                auto branching_costs = cost_calculator.GetBranchingCosts(feature);

                //for every possible combination of different left and right labels
                Node<OT> best_assignment = InitializeSol<OT>();
                auto best_label = RashomonLabelAssignment();
                for (auto& la : label_assignments) {
                    OT::Add(sols[la.left_label].sol00, sols[la.right_label].sol11, merged_sol);
                    OT::Add(merged_sol, branching_costs, merged_sol);
                    node.Set(feature, OT::worst_label, merged_sol, 0, 0);
                    if (OT::has_constraint && !SatisfiesConstraint(node, context)) continue;
                    if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, node)) continue;
                    if (!LeftStrictDominatesRightSol<OT>(UB, node)) {
                        if (node.solution + SOLUTION_PRECISION < best_assignment.solution) {
                            best_assignment = node;
                            best_label = la;
                        }
                    }
                }
                if (IsTrivialExtension(labels[best_label.left_label].label00, labels[best_label.right_label].label11)) {
                    continue;
                }
                if (best_assignment.solution <= UB.solution) {
                    auto new_sol = CreateD1Solution(best_assignment.solution, labels[best_label.left_label].label00, labels[best_label.right_label].label11, feature);
                    results.solutions->push_back(new_sol);
                    (*results.solutions_map)[num_features][STOI(typename OT::SolType, best_assignment.solution)] = new_sol; // TODO check if this map already has this solution?
                    ++results.num_solutions;
                    //ADD children
                }
            }
        }
    }

    template <class OT>
    void RashomonTerminalSolver<OT>::InitialiseChildrenInfo(const Context& context, const ADataView& data) {
        for (int f = 0; f < num_features; f++) {
            auto& child_info = children_info[f];
            child_info.Clear();
            if constexpr (OT::terminal_compute_context) {
                task->GetLeftContext(data, context, f, child_info.left_context);
                task->GetRightContext(data, context, f, child_info.right_context);
            }
        }
    }

    template <class OT>
    void RashomonTerminalSolver<OT>::UpdateBestLeftChild(int root_feature, int feature, const SolType& solution, std::pair<typename OT::SolLabelType, typename OT::SolLabelType>& label) {
        auto& child_info = children_info[root_feature];
        const auto & context = child_info.left_context;
        temp_branch_node.feature = feature;
        temp_branch_node.solution = solution;
        if (IsTrivialExtension(label.first, label.second)) {
            return;
        }
        if (OT::has_constraint && !SatisfiesConstraint(temp_branch_node, context)) return;
        if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, temp_branch_node)) return;
        bool updated = AddSubOptimalSols<OT>(child_info.left_child_assignments, temp_branch_node, UB);
        if (updated) {
            child_info.left_child_d1feature_assignments.push_back(feature);
            child_info.left_child_d1_labels.push_back(label);
        }

    }

    template <class OT>
    void RashomonTerminalSolver<OT>::UpdateBestRightChild(int root_feature, int feature, const SolType& solution, std::pair<typename OT::SolLabelType, typename OT::SolLabelType>& label) {
        auto& child_info = children_info[root_feature];
        const auto& context = child_info.right_context;
        temp_branch_node.feature = feature;
        temp_branch_node.solution = solution;
        if (IsTrivialExtension(label.first, label.second)) {
            return;
        }
        if (OT::has_constraint && !SatisfiesConstraint(temp_branch_node, context)) return;
        if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, temp_branch_node)) return;
        bool updated = AddSubOptimalSols<OT>(child_info.right_child_assignments, temp_branch_node, UB);
        if (updated) {
            child_info.right_child_d1feature_assignments.push_back(feature);
            child_info.right_child_d1_labels.push_back(label);
        }
    }

    template <class OT>
    void RashomonTerminalSolver<OT>::UpdateBestTwoNodeAssignment(const typename RashomonTerminalSolver<OT>::Context& context, int root_feature) {
        auto& child_info = children_info[root_feature];
        const auto& left_context = child_info.left_context;
        const auto& right_context = child_info.right_context;
        Counts counts;
        IndexInfo index;

        auto left_leaves = std::make_shared <Container<OT>>();
        auto right_leaves = std::make_shared <Container<OT>>();

        cost_calculator.GetIndexInfo(root_feature, root_feature, index);
        cost_calculator.GetCounts(counts, index);
        int left_size = counts.count00;
        int right_size = counts.count11;

        typename OT::SolD2Type costs;
        typename RashomonTerminalSolver<OT>::SolType leaf_sol;
        typename OT::SolLabelType assign_label;

        //for (int label = 0; label < num_labels; label++) {
        //	cost_calculator.CalcSols(counts, sols[label], label, index);
        //}
        std::vector<std::pair<typename OT::SolLabelType, typename OT::SolLabelType>> left_leaves_label;
        std::vector<std::pair<typename OT::SolLabelType, typename OT::SolLabelType>> right_leaves_label;

        Node<OT> node;
        if (left_size >= solver_parameters->minimum_leaf_node_size) {
            Node<OT> best_left_assignment = Node<OT>();
            for (int label = 0; label < num_labels; label++) {
                costs = cost_calculator.GetCosts00(label, root_feature, root_feature);
                task->ComputeD2Costs(costs, left_size, leaf_sol);
                assign_label = cost_calculator.GetLabel(label, costs, left_size);
                node.Set(INT32_MAX, assign_label, leaf_sol, 0, 0);
                //node.Set(INT32_MAX, OT::worst_label, sols[label].sol00, 0, 0);
                if (OT::has_constraint && !SatisfiesConstraint(node, left_context)) continue;
                if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, node)) continue;
                if (node.solution + SOLUTION_PRECISION < best_left_assignment.solution) {
                    best_left_assignment = node;
                }
            }
            AddSubOptimalSols<OT>(left_leaves, best_left_assignment, UB);
            left_leaves_label.push_back(std::make_pair(best_left_assignment.label,best_left_assignment.label));
        }
        if (right_size >= solver_parameters->minimum_leaf_node_size) {
            Node<OT> best_right_assignment = Node<OT>();
            for (int label = 0; label < num_labels; label++) {
                costs = cost_calculator.GetCosts11(label, root_feature, root_feature);
                task->ComputeD2Costs(costs, right_size, leaf_sol);
                assign_label = cost_calculator.GetLabel(label, costs, right_size);
                node.Set(INT32_MAX, assign_label, leaf_sol, 0, 0);
                //node.Set(INT32_MAX, OT::worst_label, sols[label].sol11, 0, 0);
                if (OT::has_constraint && !SatisfiesConstraint(node, right_context)) continue;
                if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, node)) continue;
                if (node.solution + SOLUTION_PRECISION < best_right_assignment.solution) {
                    best_right_assignment = node;
                }
            }
            AddSubOptimalSols<OT>(right_leaves, best_right_assignment, UB);
            right_leaves_label.push_back(std::make_pair(best_right_assignment.label,best_right_assignment.label));
        }

        auto& left_children = children_info[root_feature].left_child_assignments;
        auto& right_children = children_info[root_feature].right_child_assignments;
        auto& left_children_d1_features = children_info[root_feature].left_child_d1feature_assignments;
        auto& right_children_d1_features = children_info[root_feature].right_child_d1feature_assignments;
        auto& left_children_d1_labels = children_info[root_feature].left_child_d1_labels;
        auto& right_children_d1_labels = children_info[root_feature].right_child_d1_labels;
        std::vector<int> temp_vector;

        Merge(root_feature, context, left_children, right_leaves,left_children_d1_features, temp_vector,left_children_d1_labels,right_leaves_label);
        Merge(root_feature, context, left_leaves, right_children,temp_vector,right_children_d1_features,left_leaves_label,right_children_d1_labels);
    }

    template <class OT>
    void RashomonTerminalSolver<OT>::UpdateBestThreeNodeAssignment(const typename RashomonTerminalSolver<OT>::Context& context, int root_feature) {
        auto& left_children = children_info[root_feature].left_child_assignments;
        auto& right_children = children_info[root_feature].right_child_assignments;
        auto& left_children_d1_features = children_info[root_feature].left_child_d1feature_assignments;
        auto& right_children_d1_features = children_info[root_feature].right_child_d1feature_assignments;
        auto& left_children_d1_labels = children_info[root_feature].left_child_d1_labels;
        auto& right_children_d1_labels = children_info[root_feature].right_child_d1_labels;
        Merge(root_feature, context, left_children, right_children,left_children_d1_features,right_children_d1_features,left_children_d1_labels,right_children_d1_labels );

    }

    template<class OT>
    void RashomonTerminalSolver<OT>::Merge(int feature, const typename RashomonTerminalSolver<OT>::Context& context, 
        std::shared_ptr<Container<OT>> left_solutions, std::shared_ptr<Container<OT>> right_solutions, 
        std::vector<int>& left_d1_features, std::vector<int>& right_d1_features, 
        std::vector<std::pair<typename OT::SolLabelType, typename OT::SolLabelType>>& left_d1_labels,
        std::vector<std::pair<typename OT::SolLabelType, typename OT::SolLabelType>>& right_d1_labels) {
        if (left_solutions->Size() == 0 || right_solutions->Size() == 0) return;
        auto branching_costs = cost_calculator.GetBranchingCosts(feature);
        {
            std::vector<size_t> left_indices(left_solutions->GetSolutions().size());
            for (size_t i = 0; i < left_indices.size(); ++i) left_indices[i] = i;

            std::sort(left_indices.begin(), left_indices.end(), [&](size_t i, size_t j) {
                return left_solutions->GetSolutions()[i].solution < left_solutions->GetSolutions()[j].solution;
            });

            std::vector<size_t> right_indices(right_solutions->GetSolutions().size());
            for (size_t i = 0; i < right_indices.size(); ++i) right_indices[i] = i;

            std::sort(right_indices.begin(), right_indices.end(), [&](size_t i, size_t j) {
                return right_solutions->GetSolutions()[i].solution < right_solutions->GetSolutions()[j].solution;
            });

            Node<OT> node;
            for (auto& i : left_indices) {
                auto& left_sol = left_solutions->GetSolutions()[i];
                for (auto& j : right_indices) {
                    auto& right_sol = right_solutions->GetSolutions()[j];
//                    int nodes = left_sol.NumNodes() + right_sol.NumNodes() + 1;

                    CombineSols(feature, left_sol, right_sol, branching_costs, node);
                    if (!SatisfiesConstraint(node, context)) continue;
                    if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, node)) break;
                    ++results.num_solutions;
                    int64_t i_obj = STOI(typename OT::SolType, node.solution);
                    auto _sol = (*results.solutions_map)[feature][i_obj];
                    if (_sol == nullptr) {
                        _sol = std::make_shared<RecursiveSolutionTracker<OT>>(node.solution, node.feature, num_features);
                        results.solutions->push_back(_sol);
                        (*results.solutions_map)[feature][i_obj] = _sol;
                    }
                    auto sol = dynamic_cast<RecursiveSolutionTracker<OT>*>(_sol.get());
                    runtime_assert(sol);
                    ++sol->num_solutions;
                    auto left_child_sol = left_sol.feature == INT32_MAX ?
                        CreateLeafSolution(left_sol.solution, left_d1_labels[0].first, OT::worst_label) :
                        CreateD1Solution(left_sol.solution, left_d1_labels[i].first, left_d1_labels[i].second, left_d1_features[i]);
                    auto right_child_sol = right_sol.feature == INT32_MAX ?
                        CreateLeafSolution(right_sol.solution, right_d1_labels[0].first, OT::worst_label) :
                        CreateD1Solution(right_sol.solution, right_d1_labels[j].first, right_d1_labels[j].second, right_d1_features[j]);
                    sol->solutions.push_back(std::make_pair(left_child_sol,right_child_sol));

                    if (left_child_sol->GetFeature() != -1)  ++sol->feature_count[left_child_sol->GetFeature()];
                    if (right_child_sol->GetFeature() != -1 && right_child_sol->GetFeature() != left_child_sol->GetFeature())  ++sol->feature_count[right_child_sol->GetFeature()];
                    ++sol->feature_count[feature];

                    if (left_sol.feature != INT32_MAX && right_sol.feature != INT32_MAX) sol->UpdateNodeCount(3, 1);
                    else sol->UpdateNodeCount(2, 1);
                }
            }
        }
    }

    template<class OT>
    bool RashomonTerminalSolver<OT>::SatisfiesConstraint(const Node<OT>& sol, const RashomonTerminalSolver<OT>::Context& context) const {
        if constexpr (!OT::has_constraint || !OT::terminal_filter) {
            return true;
        } else {
            return task->SatisfiesConstraint(sol, context);
        }
    }

    template <class OT>
    SolutionTrackerP<OT> RashomonTerminalSolver<OT>::CreateLeafSolution(typename OT::SolType solution_value, typename OT::SolLabelType& label, typename OT::SolLabelType alt_label) {
        auto sol = std::make_shared<LeafSolutionTracker<OT>>(solution_value, label);
        sol->SetAltLabel(alt_label);
        if (alt_label != OT::worst_label) {
            return small_solution_cache->RetrieveOrAdd(sol);
        }
        return sol;
    }
    
    template <class OT>
    SolutionTrackerP<OT> RashomonTerminalSolver<OT>::CreateD1Solution(typename OT::SolType solution_value, typename OT::SolLabelType& label_left, typename OT::SolLabelType& label_right, int feature) {
        runtime_assert(!solver_parameters->ignore_trivial_extentions || label_left != label_right);
        auto sol = std::make_shared<D1SolutionTracker<OT>>(solution_value, feature, label_left, label_right);
        return small_solution_cache->RetrieveOrAdd(sol);
    }


    template <class OT>
    bool RashomonTerminalSolver<OT>::IsTrivialExtension(SolLabelType label_left, SolLabelType label_right){
    if (!solver_parameters->ignore_trivial_extentions) return false;
    return std::abs(label_left - label_right) < SOLUTION_PRECISION;
    }

    template class RashomonTerminalSolver<CostComplexAccuracy>;
    template class RashomonTerminalSolver<CostComplexRegression>;
    template class RashomonTerminalSolver<AverageDepthAccuracy>;

}
