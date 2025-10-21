/**
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <variant>
#include "utils/parameter_handler.h"
#include "solver/result.h"
#include "solver/solver.h"
#include "model/data.h"

namespace py = pybind11;
using namespace SORTD;

// Custom hasher using Python's built-in hash()
struct PyObjectHash {
    std::size_t operator()(const py::object& obj) const {
        return py::hash(obj);
    }
};

// Custom comparator using Python's == operator
struct PyObjectEqual {
    bool operator()(const py::object& a, const py::object& b) const {
        return a.equal(b);  // calls Python's __eq__
    }
};

enum task_type {
    cost_complex_accuracy,
    cost_complex_regression,
};

task_type get_task_type_code(std::string& task) {
    if (task == "cost-complex-accuracy") return cost_complex_accuracy;
    else if (task == "cost-complex-regression") return cost_complex_regression;
    else {
        std::cout << "Encountered unknown optimization task: " << task << std::endl;
        exit(1);
    }
}

template<class LT, class ET>
void NumpyToSORTDData(const py::array_t<int, py::array::c_style>& _X,
                        const py::array_t<LT, py::array::c_style>& _y,
                        const std::vector<ET>& extra_data,
                        AData& data, ADataView& data_view) {
    const bool regression = std::is_same<LT, double>::value;
    std::vector<const AInstance*> instances;
    auto X = _X.template unchecked<2>();
    auto y = _y.template unchecked<1>();
    const int num_instances = int(X.shape(0));
    const int num_features = int(X.shape(1));
    
    std::vector<std::vector<const AInstance*>> instances_per_label;
    if (regression) instances_per_label.resize(1);
    ET ed;
    std::vector<bool> v(num_features);
    for (py::size_t i = 0; i < num_instances; i++) {
        LT label = y.size() == 0 ? 0 : y(i);
        if(extra_data.size() > 0) {
            ed = extra_data[i];
        }
        for (py::size_t j = 0; j < num_features; j++) {
            v[j] = X(i,j);
        }
        auto instance = new Instance<LT, ET>(int(i), v, label, ed);
        data.AddInstance(instance);
        if (regression) {
            instances_per_label[0].push_back(instance);
        } else {
            int i_label = static_cast<int>(label);
            if (instances_per_label.size() <= i_label) { instances_per_label.resize(i_label + 1); }
            instances_per_label[i_label].push_back(instance);
        } 
    }
    data.SetNumFeatures(num_features);
	data_view = ADataView(&data, instances_per_label, {});
}

std::vector<bool> NumpyRowToBoolVector(const py::array_t<int, py::array::c_style>& _X) {
    auto X = _X.template unchecked<1>();
    std::vector<bool> v(X.shape(0));
    for (py::size_t j = 0; j < X.shape(0); j++) {
        v[j] = X(j);
    }
    return v;
}

template <class OT>
void RecursiveEvaluateObjective(Solver<OT>& _solver,
    std::vector<std::shared_ptr<AbstractSolutionTracker<OT>>>& rashomon_solutions_in,
    const ADataView& data, const Branch& branch,
    std::vector<std::vector<std::shared_ptr<AbstractSolutionTracker<OT, py::object>>>>& rashomon_solutions_out,
    py::function leaf, py::function add) {

    // For each incoming solution tracker, we return a list of outgoing solution trackers
    rashomon_solutions_out.resize(rashomon_solutions_in.size());

    //std::cout << "rash size: " << rashomon_solutions.size() << "|data| = " << data.Size() << " num-features: " << data.NumFeatures() << std::endl;

    for (int f = -1; f < data.NumFeatures(); f++) {
        
        //std::cout << "Evaluate Obj D = " << branch.Depth() << ", F = " << f << std::endl;

        std::vector<std::shared_ptr<AbstractSolutionTracker<OT>>> left_solution_trackers_in, right_solution_trackers_in;
        std::vector<size_t> solution_indices;
        bool contains_f = false;
        for (size_t i = 0; i < rashomon_solutions_in.size(); ++i) {
            if (rashomon_solutions_in[i]->GetFeature() == f) {
                contains_f = true;
                break;
            }
        }
        if (!contains_f) continue;

        ADataView left_data, right_data;
        Branch left_branch, right_branch;
        if (f >= 0) {
            _solver.GetSplitter().Split(data, branch, f, left_data, right_data);
            Branch::LeftChildBranch(branch, f, left_branch);
            Branch::RightChildBranch(branch, f, right_branch);
        }

        for (size_t i = 0; i < rashomon_solutions_in.size(); ++i) {
            auto sol = rashomon_solutions_in[i];
            // Only consider solutions with this feature (or -1 for leaf node) at its root
            if (sol->GetFeature() != f)  continue;

            // Handle the three solution-tracker cases
            if (auto leaf_tracker = dynamic_cast<LeafSolutionTracker<OT>*>(sol.get())) {
                //std::cout << "Evaluate Obj D = " << branch.Depth() << ", F = " << f << ", Leaf Sol" << std::endl;
                // Evaluate the leaf function on the current data
                
                
                py::object leaf_sol = leaf(py::cast(data, py::return_value_policy::reference), leaf_tracker->label);
                /*py::object leaf_sol = py::make_tuple(0, 0, 0);
                if constexpr (std::is_same<OT::SolLabelType, int>::value) {
                    leaf_sol = leaf_tracker->label == 1
                        ? py::make_tuple(0, data.NumInstancesForLabel(0), 1)
                        : py::make_tuple(data.NumInstancesForLabel(1), 0, 1);
                }*/
                
                rashomon_solutions_out[i].emplace_back(std::make_shared<LeafSolutionTracker<OT, py::object>>(leaf_sol, leaf_tracker->label));
            } else if (auto d1_tracker = dynamic_cast<D1SolutionTracker<OT>*>(sol.get())) {
                //std::cout << "Evaluate Obj D = " << branch.Depth() << ", F = " << f << ", D1 Sol" << std::endl;
                // Evaluate the leaf function on the left and right data
                py::object left_sol = leaf(py::cast(left_data, py::return_value_policy::reference), d1_tracker->left_label);
                py::object right_sol = leaf(py::cast(right_data, py::return_value_policy::reference), d1_tracker->right_label);
                
                /*py::object left_sol = py::make_tuple(0, 0, 0);
                py::object right_sol = py::make_tuple(0, 0, 0);
                if constexpr (std::is_same<OT::SolLabelType, int>::value) {
                    left_sol = d1_tracker->left_label == 1
                        ? py::make_tuple(0, left_data.NumInstancesForLabel(0), 1)
                        : py::make_tuple(left_data.NumInstancesForLabel(1), 0, 1);
                    right_sol = d1_tracker->right_label == 1
                        ? py::make_tuple(0, right_data.NumInstancesForLabel(0), 1)
                        : py::make_tuple(right_data.NumInstancesForLabel(1), 0, 1);
                }*/
                
                py::object comb_sol = add(left_sol, right_sol);
                rashomon_solutions_out[i].emplace_back(std::make_shared<D1SolutionTracker<OT, py::object>>(comb_sol, f, d1_tracker->left_label, d1_tracker->right_label));
            } else if (auto recursive_tracker = dynamic_cast<RecursiveSolutionTracker<OT>*>(sol.get())) {
                //std::cout << "Evaluate Obj D = " << branch.Depth() << ", F = " << f << ", Recursive Sol" << std::endl;
                // For each solution pair, add the solution trackers for both to the list of 
                // left and right trackers, and store the index of the tracker index, and the pair index
                for (auto& sol_pair : recursive_tracker->solutions) {
                    left_solution_trackers_in.push_back(sol_pair.first);
                    right_solution_trackers_in.push_back(sol_pair.second);
                    solution_indices.emplace_back(i);
                }
            } else {
                std::cout << "Solution tracker is neither leaf, nor d1, nor recursive. Abort." << std::endl;
                std::exit(1);
            }
        }

        // if no recursive solutions for this feature, continue
        if (solution_indices.empty()) {
            //std::cout << "Evaluate Obj D = " << branch.Depth() << ", F = " << f << ", No recursion" << std::endl;
            continue;
        }

        //std::cout << "Evaluate Obj D = " << branch.Depth() << ", F = " << f << ", Recursive Call" << std::endl;
        // Perform the recursive call for left and right
        std::vector <std::vector<std::shared_ptr<AbstractSolutionTracker<OT, py::object>>>> left_solutions_out, right_solutions_out;
        RecursiveEvaluateObjective(_solver, left_solution_trackers_in, left_data, left_branch, left_solutions_out, leaf, add);
        RecursiveEvaluateObjective(_solver, right_solution_trackers_in, right_data, right_branch, right_solutions_out, leaf, add);

        //std::cout << "Evaluate Obj D = " << branch.Depth() << ", F = " << f << ", Combine" << std::endl;
        //std::cout << "SI: " << solution_indices.size() << ", LS " << left_solutions.size() << ", RS " << right_solutions.size() << std::endl;

        size_t sol_index = 0;
        for (size_t i: solution_indices) {
            //std::cout << "Combine " << i << std::endl;
            auto& left_sols = left_solutions_out[sol_index];
            auto& right_sols = right_solutions_out[sol_index];
            
            // Group solutions for the same pair of solution trackers by their objective (py::object)
            std::unordered_map<py::object, std::shared_ptr<RecursiveSolutionTracker<OT, py::object>>, PyObjectHash, PyObjectEqual> solution_map;

            for (size_t j = 0; j < left_sols.size(); j++) {
                auto left_sol = left_sols[j];
                for (size_t k = 0; k < right_sols.size(); k++) {
                    auto right_sol = right_sols[k];
                    py::object comb_sol = add(left_sol->obj, right_sol->obj);
                    //solutions[i].emplace_back(comb_sol);
                    auto sol = solution_map[comb_sol];
                    if (sol == nullptr) {
                        sol = std::make_shared<RecursiveSolutionTracker<OT, py::object>>(comb_sol, f, data.NumFeatures());
                        rashomon_solutions_out[i].emplace_back(sol);
                        solution_map[comb_sol] = sol;
                    }
                    sol->UpdateNumSolutions(left_sol->GetNumSolutions() * right_sol->GetNumSolutions());
                    sol->solutions.push_back(std::make_pair(left_sol, right_sol));
                }
            }
            sol_index++;
        }

    }

    //std::cout << "Evaluate Obj D = " << branch.Depth() << " return." << std::endl;

}

template <class LT, class ET>
py::class_<Instance<LT, ET>, AInstance> DefineInstance(py::module& m, const std::string& label_name, const std::string& suffix) {
    py::class_<Instance<LT, ET>, AInstance> instance(m, ("Instance" + label_name + suffix).c_str());
    //TODO: instance.def_property_readonly("features", &Instance<OT::LabelType, OT::ET>::GetFeatures); 
    instance.def_property_readonly("label", &Instance<LT, ET>::GetLabel);
    return instance;
}



template <class OT>
py::class_<Solver<OT>> DefineSolver(py::module& m, const std::string& name) {
        
    py::class_<Solver<OT>> solver(m, (name + "Solver").c_str());
    
    solver.def("_update_parameters", [](Solver<OT>& _solver, const ParameterHandler& parameters) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        // parameters.CheckParameters();
        _solver.UpdateParameters(parameters);
    });

    solver.def("_get_parameters", &Solver<OT>::GetParameters);

    solver.def("_solve", [](Solver<OT>& _solver,
            const py::array_t<int, py::array::c_style>& X,
            const py::array_t<typename OT::LabelType, py::array::c_style>& y,
            const std::vector<typename OT::ET> extra_data) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        
        if (_solver.GetParameters().GetBooleanParameter("verbose")) { _solver.GetParameters().PrintParametersDifferentFromDefault(); }
        
        auto train_data = std::make_shared<AData>();
        ADataView train_data_view;
        NumpyToSORTDData<typename OT::LabelType, typename OT::ET>(X, y, extra_data, *train_data, train_data_view);
        _solver.PreprocessData(*train_data);
        py::object result;
        result = py::cast(_solver.Solve(train_data_view));
        return py::make_tuple(result, train_data); // return train data long, so that we can hold its reference
    });

    solver.def("_extend", [](Solver<OT>& _solver, std::shared_ptr<SolverResult>& solver_result) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        auto result = std::static_pointer_cast<SolverTaskResult<OT>>(solver_result);
        try {
            _solver.ConstructRashomonSet(result);
        } catch (const std::exception& e) {
            std::cerr << "Exception in C++: " << e.what() << std::endl;
            throw;
        }        
    });

    solver.def("_predict", [](Solver<OT>& _solver,
            std::shared_ptr<Tree<OT>>& tree,
            const py::array_t<int, py::array::c_style>& X,
            const std::vector<typename OT::ET> extra_data) -> py::object {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        AData test_data;
        ADataView test_data_view;
        py::array_t<typename OT::LabelType, py::array::c_style> y;
        NumpyToSORTDData<typename OT::LabelType, typename OT::ET>(X, y, extra_data, test_data, test_data_view);
        _solver.PreprocessData(test_data, false);
        std::vector<typename OT::LabelType> predictions = _solver.Predict(tree, test_data_view);
        return py::array_t<typename OT::LabelType, py::array::c_style>(predictions.size(), predictions.data()); 
    });
    
    solver.def("_test_performance", [](Solver<OT>& _solver,
            std::shared_ptr<SolverResult>& solver_result,
            const py::array_t<int, py::array::c_style>& X,
            const py::array_t<typename OT::LabelType, py::array::c_style>& y_true,
            const std::vector<typename OT::ET> extra_data) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        AData test_data;
        ADataView test_data_view;
        NumpyToSORTDData<typename OT::LabelType, typename OT::ET>(X, y_true, extra_data, test_data, test_data_view);
        _solver.PreprocessData(test_data, false);
        return _solver.TestPerformance(solver_result, test_data_view);
    });

    solver.def("_rashomon_test_performance", [](Solver<OT>& _solver,
                                        const Tree<OT>* tree,
                                       const py::array_t<int, py::array::c_style>& X,
                                       const py::array_t<typename OT::LabelType, py::array::c_style>& y_true,
                                       const std::vector<typename OT::ET> extra_data) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        AData test_data;
        ADataView test_data_view;
        NumpyToSORTDData<typename OT::LabelType, typename OT::ET>(X, y_true, extra_data, test_data, test_data_view);
        _solver.PreprocessData(test_data, false);
        return _solver.RashomonTestPerformance(tree, test_data_view);
    });

    solver.def("_get_tree", [](Solver<OT>& _solver,
            std::shared_ptr<SolverResult>& solver_result) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return result->trees[result->best_index];
    });

    solver.def("_get_rashomon_set_size", [](Solver<OT>& _solver,
            std::shared_ptr<SolverResult>& solver_result) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return result->rashomon_set_size;
    });

    solver.def("_get_tree_n", [](Solver<OT>& _solver,
                                        std::shared_ptr<SolverResult>& solver_result,
                                        size_t& n) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return _solver.CreateRashomonTreeN(result->rashomon_solutions,result->root_solution_counts, n);
    });

    solver.def("_get_root_feature_stats", [](Solver<OT>& _solver,
                                 std::shared_ptr<SolverResult>& solver_result) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return result->root_feature_stats;
    });

    solver.def("_get_feature_stats", [](Solver<OT>& _solver,
                                             std::shared_ptr<SolverResult>& solver_result) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return result->feature_stats;
    });

    solver.def("_get_node_num_stats", [](Solver<OT>& _solver,
                                        std::shared_ptr<SolverResult>& solver_result) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return result->num_nodes_stats;
    });

    solver.def("_get_trees_with_root_feature", [](Solver<OT>& _solver,
                                                  std::shared_ptr<SolverResult>& solver_result,
                                                  int& feature) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return _solver.CalculateTreesWithRootFeature(result,feature);
    });

    solver.def("_get_trees_with_feature", [](Solver<OT>& _solver,
                                             std::shared_ptr<SolverResult>& solver_result,
                                             int& feature) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return _solver.CalculateTreesWithFeature(result,feature);
    });

    solver.def("_get_trees_without_feature", [](Solver<OT>& _solver,
                                             std::shared_ptr<SolverResult>& solver_result,
                                             int& feature) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return _solver.CalculateTreesWithoutFeature(result,feature);
    });

    solver.def("_get_trees_with_node_budget", [](Solver<OT>& _solver,
                                                 std::shared_ptr<SolverResult>& solver_result,
                                                 int& node_budget) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return _solver.CalculateTreesWithNodeBudget(result,node_budget);
    });

    solver.def("_get_solution_list", [](Solver<OT>& _solver, std::shared_ptr<SolverResult>& solver_result) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return *(result->rashomon_solutions);
    });

    solver.def("_get_num_used_splits", [](Solver<OT>& _solver,
                                             std::shared_ptr<SolverResult>& solver_result) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return result->num_active_split_tracker_per_depth;
    });

    solver.def("_get_average_solution_size", [](Solver<OT>& _solver,
        std::shared_ptr<SolverResult>& solver_result) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return result->num_average_solution_per_split_tracker;
    });

    solver.def("_evaluate_objective", [](Solver<OT>& _solver, std::shared_ptr<SolverResult>& solver_result,
        const py::array_t<int, py::array::c_style>& X,
        const py::array_t<typename OT::LabelType, py::array::c_style>& y,
        const std::vector<typename OT::ET> extra_data, 
        size_t rashomon_start_index, py::function leaf, py::function add) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        
        AData train_data;
        ADataView train_data_view;
        NumpyToSORTDData<typename OT::LabelType, typename OT::ET>(X, y, extra_data, train_data, train_data_view);
        _solver.PreprocessData(train_data);

        std::vector<std::shared_ptr<AbstractSolutionTracker<OT, py::object>>> solutions_out;
        std::vector<std::vector<std::shared_ptr<AbstractSolutionTracker<OT, py::object>>>> solutions;
        try {
            auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
            const Branch branch;
            std::vector<std::shared_ptr<AbstractSolutionTracker<OT>>> solutions_in(result->rashomon_solutions->begin() + rashomon_start_index, result->rashomon_solutions->end());
            
            RecursiveEvaluateObjective<OT>(_solver, solutions_in, train_data_view, branch, solutions, leaf, add);

            // Flatten the vectors of vectors of solution trackers //

            // Create a solution map for each feature
            std::vector<std::unordered_map<py::object, std::shared_ptr<RecursiveSolutionTracker<OT, py::object>>, PyObjectHash, PyObjectEqual>> solution_map(train_data_view.NumFeatures());
            
            for (size_t i = 0; i < solutions.size(); i++) {
                //std::cout << "Parsing SolTracker " << i << " with " << solutions[i].size() << " solutions." << std::endl;
                for (auto& sol_tracker: solutions[i]) {
                    
                    if (auto rec_tracker = std::dynamic_pointer_cast<RecursiveSolutionTracker<OT, py::object>>(sol_tracker)) {
                        std::shared_ptr<RecursiveSolutionTracker<OT, py::object>> sol = solution_map[rec_tracker->GetFeature()][rec_tracker->obj];
                        if (sol == nullptr) {
                            sol = rec_tracker;
                            solutions_out.emplace_back(sol);
                            solution_map[rec_tracker->GetFeature()][sol->obj] = sol;
                        } else {
                            sol->UpdateNumSolutions(rec_tracker->GetNumSolutions());
                            sol->solutions.insert(sol->solutions.end(), rec_tracker->solutions.begin(), rec_tracker->solutions.end());
                        }
                    } else {
                        solutions_out.emplace_back(sol_tracker);
                    }
                }
            }
            //std::sort(solution_list.begin(), solution_list.end(), [](const std::pair<std::shared_ptr<Tree<OT>>, double>& a, const std::pair<std::shared_ptr<Tree<OT>>, double>& b) {
            //    return a.second < b.second;
            //});
        } catch (const std::exception& e) {
            std::cerr << "C++ exception: " << e.what() << std::endl;
        }
        return solutions_out;
    });

    py::class_<Tree<OT>, std::shared_ptr<Tree<OT>>> tree(m, (name + "Tree").c_str());

    tree.def("is_leaf_node", &Tree<OT>::IsLabelNode, "Return true if this node is a leaf node.");
    tree.def("is_branching_node", &Tree<OT>::IsFeatureNode, "Return true if this node is a branching node.");
    tree.def("get_depth", &Tree<OT>::Depth, "Return the depth of the tree.");
    tree.def("get_num_branching_nodes", &Tree<OT>::NumBranchingNodes, "Return the number of branching nodes in the tree.");
    tree.def("get_num_leaf_nodes", &Tree<OT>::NumLeafNodes, "Return the number of leaf nodes in the tree.");
    tree.def("__str__", &Tree<OT>::ToString);
    tree.def_readonly("left_child", &Tree<OT>::left_child, "Return a reference to the left child node.");
    tree.def_readonly("right_child", &Tree<OT>::right_child, "Return a reference to the right child node.");
    tree.def_readonly("feature", &Tree<OT>::feature, "Get the index of the feature on this branching node.");
    tree.def_readonly("label", &Tree<OT>::label, "Get the label of this leaf node.");

    py::class_<AbstractSolutionTracker<OT>, std::shared_ptr<AbstractSolutionTracker<OT>>> sol_tracker(m, (name + "SolutionTracker").c_str());
    sol_tracker.def_property_readonly("num_solutions", &AbstractSolutionTracker<OT>::GetNumSolutions);
    sol_tracker.def_readonly("objective", &AbstractSolutionTracker<OT>::obj);

    py::class_<AbstractSolutionTracker<OT, py::object>, std::shared_ptr<AbstractSolutionTracker<OT, py::object>>> pysol_tracker(m, (name + "PySolutionTracker").c_str());
    pysol_tracker.def_property_readonly("num_solutions", &AbstractSolutionTracker<OT, py::object>::GetNumSolutions);
    pysol_tracker.def_readonly("objective", &AbstractSolutionTracker<OT, py::object>::obj);

    py::class_<LeafSolutionTracker<OT>, std::shared_ptr<LeafSolutionTracker<OT>>, AbstractSolutionTracker<OT>> leaf_sol_tracker(m, (name + "LeafSolutionTracker").c_str());
    py::class_<D1SolutionTracker<OT>,   std::shared_ptr<D1SolutionTracker<OT>>, AbstractSolutionTracker<OT>>   d1_sol_tracker(m,   (name + "D1SolutionTracker").c_str());
    py::class_<RecursiveSolutionTracker<OT>, std::shared_ptr<RecursiveSolutionTracker<OT>>, AbstractSolutionTracker<OT>> rec_sol_tracker(m, (name + "RecursiveSolutionTracker").c_str());
    rec_sol_tracker.def_property_readonly("feature", &RecursiveSolutionTracker<OT>::GetFeature);

    py::class_<LeafSolutionTracker<OT, py::object>, std::shared_ptr<LeafSolutionTracker<OT, py::object>>, AbstractSolutionTracker<OT, py::object>> pyleaf_sol_tracker(m, (name + "PyLeafSolutionTracker").c_str());
    py::class_<D1SolutionTracker<OT, py::object>, std::shared_ptr<D1SolutionTracker<OT, py::object>>, AbstractSolutionTracker<OT, py::object>>   pyd1_sol_tracker(m, (name + "PyD1SolutionTracker").c_str());
    py::class_<RecursiveSolutionTracker<OT, py::object>, std::shared_ptr<RecursiveSolutionTracker<OT, py::object>>, AbstractSolutionTracker<OT, py::object>> pyrec_sol_tracker(m, (name + "PyRecursiveSolutionTracker").c_str());
    pyrec_sol_tracker.def_property_readonly("feature", &RecursiveSolutionTracker<OT, py::object>::GetFeature);

    return solver;
}

void ExposeStringProperty(py::class_<ParameterHandler>& parameter_handler, const std::string& cpp_property_name, const std::string& py_property_name) {
    parameter_handler.def_property(py_property_name.c_str(), 
        [=](const ParameterHandler& p) { return p.GetStringParameter(cpp_property_name); },
        [=](ParameterHandler& p, const std::string& new_value) { return p.SetStringParameter(cpp_property_name, new_value); }); 
}

void ExposeIntegerProperty(py::class_<ParameterHandler>& parameter_handler, const std::string& cpp_property_name, const std::string& py_property_name) {
    parameter_handler.def_property(py_property_name.c_str(), 
        [=](const ParameterHandler& p) { return p.GetIntegerParameter(cpp_property_name); },
        [=](ParameterHandler& p, const int64_t new_value) { return p.SetIntegerParameter(cpp_property_name, new_value); });
}

void ExposeBooleanProperty(py::class_<ParameterHandler>& parameter_handler, const std::string& cpp_property_name, const std::string& py_property_name) {
    parameter_handler.def_property(py_property_name.c_str(), 
        [=](const ParameterHandler& p) { return p.GetBooleanParameter(cpp_property_name); },
        [=](ParameterHandler& p, const bool new_value) { return p.SetBooleanParameter(cpp_property_name, new_value); }); 
}

void ExposeFloatProperty(py::class_<ParameterHandler>& parameter_handler, const std::string& cpp_property_name, const std::string& py_property_name) {
    parameter_handler.def_property(py_property_name.c_str(), 
        [=](const ParameterHandler& p) { return p.GetFloatParameter(cpp_property_name); },
        [=](ParameterHandler& p, const double new_value) { return p.SetFloatParameter(cpp_property_name, new_value); }); 
}

PYBIND11_MODULE(csortd, m) {
    m.doc() = "This is documentation";
    
    /*************************************
          DataView
    ************************************/
    py::class_<AData, std::shared_ptr<AData>>(m, "AData");

    py::class_<ADataView> dataview(m, "DataView");
    dataview.def("get_instances_for_label", &ADataView::GetInstancesForLabel);
    dataview.def("num_instances_for_label", &ADataView::NumInstancesForLabel);
    dataview.def("num_instances_for_feature", &ADataView::NumInstancesForFeature);
    dataview.def("num_instances_for_label_and_feature", &ADataView::NumInstancesForLabelAndFeature);
    dataview.def_property_readonly("size", &ADataView::Size);

    /*************************************
          Instance
    ************************************/
    py::class_<AInstance> a_instance(m, "AInstance");
    a_instance.def_property_readonly("weight", &AInstance::GetWeight);
    a_instance.def("is_feature_present", &AInstance::IsFeaturePresent);
    DefineInstance<int, ExtraData>(m, "IntLabel", "");
    DefineInstance<int, CCAccExtraData>(m, "IntLabel", "CC");
    DefineInstance<double, ExtraData>(m, "FloatLabel", "");
    DefineInstance<double, RegExtraData>(m, "FloatLabel", "RegCC");


    /*************************************
          SolverResult
    ************************************/
    py::class_<SolverResult, std::shared_ptr<SolverResult>> solver_result(m, "SolverResult");

    solver_result.def("is_feasible", &SolverResult::IsFeasible);

    solver_result.def("is_optimal", [](const SolverResult &solver_result) {
        return solver_result.IsProvenOptimal();
    });

    solver_result.def("is_complete_enumaration", [](const SolverResult &solver_result) {
        return solver_result.IsCompleteEnumaration();
    });

    solver_result.def("is_exhausted", [](const SolverResult& solver_result) {
        return solver_result.IsExhausted();
    });

    solver_result.def("score", [](const SolverResult &solver_result) {
        return solver_result.optimal_scores[solver_result.best_index]->score;
    });

    solver_result.def("tree_depth", [](const SolverResult &solver_result) {
        return solver_result.GetBestDepth();
    });

    solver_result.def("tree_nodes", [](const SolverResult &solver_result) {
        return solver_result.GetBestNodeCount();
    });


    /*************************************
           ParameterHandler
     ************************************/
    py::class_<ParameterHandler> parameter_handler(m, "ParameterHandler");

    parameter_handler.def(py::init(&ParameterHandler::DefineParameters));
    parameter_handler.def("check_parameters", &ParameterHandler::CheckParameters);
    ExposeStringProperty(parameter_handler, "task", "optimization_task");
    ExposeIntegerProperty(parameter_handler, "max-depth", "max_depth");
    ExposeIntegerProperty(parameter_handler, "max-num-nodes", "max_num_nodes");
    ExposeBooleanProperty(parameter_handler, "use-rashomon-multiplier", "use_rashomon_multiplier");
    ExposeFloatProperty(parameter_handler, "rashomon-multiplier", "rashomon_multiplier");
    ExposeIntegerProperty(parameter_handler, "random-seed", "random_seed");
    ExposeFloatProperty(parameter_handler, "time", "time_limit");
    ExposeFloatProperty(parameter_handler, "cost-complexity", "cost_complexity");
    ExposeStringProperty(parameter_handler, "feature-ordering", "feature_ordering");
    ExposeBooleanProperty(parameter_handler, "verbose", "verbose");
    ExposeFloatProperty(parameter_handler, "train-test-split", "validation_set_fraction");
    ExposeIntegerProperty(parameter_handler, "min-leaf-node-size", "min_leaf_node_size");
    ExposeBooleanProperty(parameter_handler, "use-branch-caching", "use_branch_caching");
    ExposeBooleanProperty(parameter_handler, "use-dataset-caching", "use_dataset_caching");
    ExposeStringProperty(parameter_handler, "regression-bound", "regression_lower_bound");
    ExposeBooleanProperty(parameter_handler, "track-tree-statistics", "track_tree_statistics");
    ExposeBooleanProperty(parameter_handler, "ignore-trivial-extensions", "ignore_trivial_extensions");

    
    ExposeIntegerProperty(parameter_handler, "max-num-trees", "max_num_trees");
    
    /*************************************
           RandomEngine
     ************************************/
    py::class_<std::default_random_engine> random_engine(m, "RandomEngine");
    random_engine.def(py::init<uint_fast32_t>());
    /*************************************
           Solver
     ************************************/

    DefineSolver<CostComplexAccuracy>(m, "CostComplexAccuracy");
    DefineSolver<CostComplexRegression>(m, "CostComplexRegression");

    m.def("initialize_sortd_solver", [](ParameterHandler& parameters, std::default_random_engine& rng) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));

        // parameters.CheckParameters();
	    bool verbose = parameters.GetBooleanParameter("verbose");

        SORTD::AbstractSolver* solver;
        std::string task = parameters.GetStringParameter("task");
        switch(get_task_type_code(task)) {
            case cost_complex_accuracy: solver = new Solver<CostComplexAccuracy>(parameters, &rng); break;
            case cost_complex_regression: solver = new Solver<CostComplexRegression>(parameters, &rng); break;
        }
        return solver;
    }, py::keep_alive<0, 1>());


}