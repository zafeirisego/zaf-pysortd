/**
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
 */


#include "solver/solver.h"
#include "utils/debug.h"
#include "rashomon/branch_tracker.h"
#include "utils/utils.h"

namespace SORTD {

	/************************
	*    SolverParameters   *
	************************/

	SolverParameters::SolverParameters(const ParameterHandler& parameters) :
		verbose(parameters.GetBooleanParameter("verbose")),
        rashomon_multiplier(parameters.GetFloatParameter("rashomon-multiplier")),
		minimum_leaf_node_size(int(parameters.GetIntegerParameter("min-leaf-node-size"))),
        ignore_trivial_extentions(parameters.GetBooleanParameter("ignore-trivial-extensions")){ }

	/************************
	*    AbstractSolver     *
	************************/

	AbstractSolver::AbstractSolver(ParameterHandler& parameters, std::default_random_engine* rng) :
		parameters(parameters), solver_parameters(parameters), rng(rng),
		data_splitter(MAX_DEPTH), progress_tracker(-1), reconstructing(false) { }

	void AbstractSolver::UpdateParameters(const ParameterHandler& parameters) {
		this->parameters = parameters;
		solver_parameters = SolverParameters(parameters);
	}

	/************************
	*    ProgressTracker    *
	************************/
	
	ProgressTracker::ProgressTracker(int num_features) : num_features(num_features), count(0) {
		double max_dots = 40;
		print_progress_every = int(std::ceil(double(num_features) / max_dots));
		progress_length = std::max(1, int(std::floor(max_dots / double(num_features))));
		Reset();
	}

	void ProgressTracker::UpdateProgressCount(int new_count) {
		for (; count <= new_count; count++) {
			if (count % print_progress_every == 0) {
				for (int _p = 0; _p < progress_length; _p++) std::cout << ".";
			}
		}
	}
	
	void ProgressTracker::Done() {
		UpdateProgressCount(num_features);
	}
		

	/************************
	*        Solver         *
	************************/

	template<class OT>
	Solver<OT>::Solver(ParameterHandler& parameters, std::default_random_engine* rng) : AbstractSolver(parameters, rng) {
		task = new OT(parameters);
	}

	template<class OT>
	Solver<OT>::~Solver() {
		if (cache != nullptr) delete cache;
		if (terminal_solver1 != nullptr) delete terminal_solver1;
		if (terminal_solver2 != nullptr) delete terminal_solver2;
        if (rashomon_terminal_solver1 != nullptr) delete rashomon_terminal_solver1;
        if (rashomon_terminal_solver2 != nullptr) delete rashomon_terminal_solver2;
		if (similarity_lower_bound_computer != nullptr) delete similarity_lower_bound_computer;
		if (task != nullptr) delete task;
	}

    template<class OT>
    void Solver<OT>::PrintSolverParameters() {
        auto p = solver_parameters;
        std:: cout << "use_terminal_solver: " << p.use_terminal_solver << std::endl <<
        " use_lower_bounding: " << p.use_lower_bounding << std::endl <<
        " use_task_lower_bounding: "  << p.use_task_lower_bounding << std::endl <<
        " subtract_ub: " << p.subtract_ub << std::endl <<
        " use_upper_bounding: " << p.use_upper_bounding << std::endl <<
        " similarity_lb: " << p.similarity_lb << std::endl <<
        " cache_data_splits: " << p.cache_data_splits << std::endl <<
        " use_lower_bound_early_stop: " << p.use_lower_bound_early_stop << std::endl <<
        " minimum_leaf_node_size: " << p.minimum_leaf_node_size << std::endl <<
        " UB_LB_max_size: " << p.UB_LB_max_size << std::endl <<
        " rashomon_multiplier: " << p.rashomon_multiplier << std::endl <<
        " ignore_trivial_extentions: " << p.ignore_trivial_extentions << std::endl;

        std:: cout << "task: " << parameters.GetStringParameter("task") << std::endl <<
        "max-depth: " << parameters.GetIntegerParameter("max-depth") << std::endl <<
        "max-num-nodes: " << parameters.GetIntegerParameter("max-num-nodes") << std::endl <<
        "max-num-features: " << parameters.GetIntegerParameter("max-num-features") << std::endl <<
        "num-instances: " << parameters.GetIntegerParameter("num-instances") << std::endl <<
        "train-test-split: " << parameters.GetFloatParameter("train-test-split") << std::endl <<
        "stratify: " << parameters.GetBooleanParameter("stratify") << std::endl <<
        "min-leaf-node-size: " << parameters.GetIntegerParameter("min-leaf-node-size") << std::endl <<
        "feature-ordering: " << parameters.GetStringParameter("feature-ordering") << std::endl <<
         "random-seed: " << parameters.GetIntegerParameter("random-seed") << std::endl <<
         "use-branch-caching: " << parameters.GetBooleanParameter("use-branch-caching") << std::endl <<
         "use-dataset-caching: " << parameters.GetBooleanParameter("use-dataset-caching") << std::endl <<
         "duplicate-factor: " << parameters.GetIntegerParameter("duplicate-factor") << std::endl <<
         "cost-complexity: " << parameters.GetFloatParameter("cost-complexity") << std::endl <<
         "regression-bound: " << parameters.GetStringParameter("regression-bound") << std::endl <<
         "use-rashomon-multiplier: " << parameters.GetBooleanParameter("use-rashomon-multiplier") << std::endl <<
         "rashomon-multiplier: " << parameters.GetFloatParameter("rashomon-multiplier") << std::endl <<
         "track-tree-statistics: " << parameters.GetBooleanParameter("track-tree-statistics") << std::endl <<
         "max-num-trees: " << parameters.GetIntegerParameter("max-num-trees") << std::endl <<
         "ignore-trivial-extensions: " << parameters.GetBooleanParameter("ignore-trivial-extensions") << std::endl;



    }

	template<class OT>
	std::shared_ptr<SolverResult> Solver<OT>::Solve(const ADataView& _train_data) {

        stopwatch.Initialise(parameters.GetFloatParameter("time"));
		InitializeSolver(_train_data);

        // PrintSolverParameters();

		Solver<OT>::Context root_context;

		// if no UB is given yet, compute the leaf solutions in the root node and use these as UB
		auto result = InitializeSol<OT>();
		rashomon_bound_delta = 0;
		if (CheckEmptySol<OT>(global_UB)) {
			global_UB = InitializeSol<OT>();
			// If an upper bound is provided, and the objective is numeric, add it to the UB
//			if constexpr (std::is_same<Solver<OT>::SolType, double>::value || std::is_same<Solver<OT>::SolType, int>::value) {
//				double ub = parameters.GetFloatParameter("upper-bound");
//				// if INT32-MAX upper bound (default), but the solution value is a double, change the UB to DBL_MAX
//				if (std::abs(ub - INT32_MAX) < 1 && std::is_same<Solver<OT>::SolType, double>::value) {
//					ub = DBL_MAX;
//				}
//				AddSol<OT>(global_UB, Node<OT>(ub));
//			}
			result = SolveLeafNode(train_data, root_context, global_UB);
		}
		// Initialize the Rashomon bound delta
		//rashomon_bound_delta = result.solution* solver_parameters.rashomon_multiplier;

		// If all number of nodes options should be considered, set min_num_nodes to 1, else to max
		int min_num_nodes = int(parameters.GetIntegerParameter("max-num-nodes"));
//		if (parameters.GetBooleanParameter("all-trees")) { min_num_nodes = 1; }

		int max_num_nodes = int(parameters.GetIntegerParameter("max-num-nodes"));
		int max_depth = std::min(int(parameters.GetIntegerParameter("max-depth")), max_num_nodes);
		int max_depth_searched = int(parameters.GetIntegerParameter("max-depth"));
		max_depth_finished = 0;
		if (min_num_nodes == 1) {
			// For each number of nodes that should be considered, find the optimal solution
			for (int num_nodes = min_num_nodes; num_nodes <= max_num_nodes; num_nodes++) {
				if (!stopwatch.IsWithinTimeLimit()) { break; }
				if (solver_parameters.verbose) {
					progress_tracker.Reset();
					std::cout << "Search n = " << std::setw(3) << num_nodes << " | ";
				}
				max_depth = std::min(int(parameters.GetIntegerParameter("max-depth")), num_nodes);
				IncrementalResetCache();
				auto sol = SolveSubTree(train_data, root_context, global_UB, max_depth, num_nodes);
				max_depth_searched = max_depth;
				max_depth_finished = int_log2(num_nodes + 1);
				AddSols<OT>(task, 0, result, sol);
				AddSols<OT>(global_UB, sol);
				//rashomon_bound_delta = result.solution* solver_parameters.rashomon_multiplier;
				if (result.IsFeasible()) {
					global_UB.solution = global_UB.solution = result.solution - OT::minimum_difference;
					}
				if (solver_parameters.verbose) {
					progress_tracker.Done();
					std::cout << " | " << std::setw(5) << stopwatch.TimeElapsedInSeconds() << " seconds ";
					std::cout << " Solution = " << std::setw(5) << OT::SolToString(result.solution) << std::endl;
				}
				if (!stopwatch.IsWithinTimeLimit()) break;
			}
		} else {
			// Search for trees of increasing depth
			int min_max_depth = std::min(2, max_depth);
			for (int _max_depth = min_max_depth; _max_depth <= max_depth; _max_depth++) {
				int _num_nodes = std::min(max_num_nodes, (1 << _max_depth) - 1);
				if (solver_parameters.verbose ) {
					progress_tracker.Reset();
					std::cout << "Search d = " << std::setw(2) << _max_depth << " | ";
				}
				solver_parameters.use_lower_bound_early_stop = !sparse_objective || _max_depth == max_depth;
				IncrementalResetCache();
				auto sol = SolveSubTree(train_data, root_context, global_UB, _max_depth, _num_nodes);
				AddSols<OT>(task, 0, result, sol);
				AddSols<OT>(global_UB, sol);
				//rashomon_bound_delta = result.solution* solver_parameters.rashomon_multiplier;
				if (result.IsFeasible()) {
					global_UB.solution = std::min(global_UB.solution, result.solution - OT::minimum_difference);
				}
				max_depth_searched = _max_depth;
				max_depth_finished = _max_depth;
				if (solver_parameters.verbose) {
					progress_tracker.Done();
					std::cout << " | " << std::setw(5) << stopwatch.TimeElapsedInSeconds() << " seconds ";
					std::cout << " Solution = " << std::setw(5) << OT::SolToString(result.solution) << std::endl;
				}
				if (!stopwatch.IsWithinTimeLimit()) break;
			}
		}
		//max_depth_finished = 0;
		
		// Evaluate the results
		auto solver_result = std::make_shared<SolverTaskResult<OT>>();
		solver_result->is_proven_optimal = stopwatch.IsWithinTimeLimit();
		reconstructing = true;
		
		if (result.IsFeasible()) {
			clock_t clock_start = clock();
			int depth = int_log2(result.NumNodes() + 1);
			for (; depth <= max_depth_searched && depth <= result.NumNodes(); depth++) {
				if (cache->IsOptimalAssignmentCached(train_data, root_context.GetBranch(), depth, result.NumNodes())
					&& cache->RetrieveOptimalAssignment(train_data, root_context.GetBranch(), depth, result.NumNodes()) == result) {
					break;
				}
			}
			depth = std::min(depth, max_depth_searched);
			auto tree = ConstructOptimalTree(result, train_data, root_context, depth, result.NumNodes());
			stats.time_reconstructing += double(clock() - clock_start) / CLOCKS_PER_SEC;
			auto score = InternalTrainScore<OT>::ComputeTrainPerformance(&data_splitter, task, tree.get(), train_data);
			PostProcessTree(tree);
            solver_result->AddSolution(tree, score);
            if (parameters.GetBooleanParameter("use-rashomon-multiplier")) {
                rashomon_bound = (1 + solver_parameters.rashomon_multiplier) * result.solution;
            } else {
                auto temp_UB = InitializeSol<OT>();
                auto leaf_solution = SolveLeafNode(train_data,root_context,temp_UB);
                rashomon_bound = leaf_solution.solution;
            }

			if (solver_parameters.verbose) {
				std::cout << "Optimal Solution: " << result.solution 
					<< " \tRashomon bound : " << std::setprecision(int(std::log10(rashomon_bound)) + 5) << rashomon_bound
					<< " \tTime: " << stopwatch.TimeElapsedInSeconds() << " seconds " << std::endl;
				std::cout << "------------------------" << std::endl;
			}
            ConstructRashomonSet(solver_result);
            solver_result->is_complete_enumaration = stopwatch.IsWithinTimeLimit();

		}

		stats.total_time += stopwatch.TimeElapsedInSeconds();
		if (solver_parameters.verbose) {
			stats.Print();
		}

		return solver_result;
	}

	template<class OT>
	void Solver<OT>::InitializeSolver(const ADataView& _train_data, bool reset) {
		// Initialize reconstructing state to false
		reconstructing = false;
		
		// Initialize the progress tracker
		progress_tracker = ProgressTracker(_train_data.NumFeatures());
		
		// Inform the task about the solver parameters
		task->UpdateParameters(parameters);

		// If the training data is the same, the cache does not need to be repopulated
		// Except if the hyper tune configuration sets reset to true
		if (!reset && org_train_data == _train_data) return;
		
		// Update the data objects, and inform the task about the new data
		org_train_data = _train_data;
		PreprocessTrainData(org_train_data, train_data);
		train_summary = DataSummary(train_data);
		task->InformTrainData(train_data, train_summary);

		// Reset the cache, data split cache, terminal solvers, sim-bound computer
		ResetCache();
		if (terminal_solver1 != nullptr) delete terminal_solver1;
		if (terminal_solver2 != nullptr) delete terminal_solver2;
		if (rashomon_terminal_solver1 != nullptr) delete rashomon_terminal_solver1;
        if (rashomon_terminal_solver2 != nullptr) delete rashomon_terminal_solver2;
		if (similarity_lower_bound_computer != nullptr) delete similarity_lower_bound_computer;
		cache = new Cache<OT>(parameters, MAX_DEPTH, train_data.Size());
		if (!solver_parameters.use_lower_bounding) cache->DisableLowerBoundCaching();

		if constexpr (OT::use_terminal) {
			terminal_solver1 = new TerminalSolver<OT>(this);
			terminal_solver2 = new TerminalSolver<OT>(this);
            rashomon_terminal_solver1 = new RashomonTerminalSolver<OT>(this);
            rashomon_terminal_solver2 = new RashomonTerminalSolver<OT>(this);
        }
		// Only use the similarity lower bound computer if the optimization task is element-wise additive
		if constexpr (OT::element_additive) {
			similarity_lower_bound_computer = new SimilarityLowerBoundComputer<OT>(task,
				NumLabels(), MAX_DEPTH, int(parameters.GetIntegerParameter("max-num-nodes")), train_data.Size());
			if (!solver_parameters.similarity_lb) similarity_lower_bound_computer->Disable();
		} else {
			solver_parameters.similarity_lb = false;
		}

		// Disable data split cache, if so configured
		if (!solver_parameters.cache_data_splits) data_splitter.Disable();
		data_splitter.Clear();

		// Initialize the global UB with an empty solution
		global_UB = InitializeSol<OT>();

       if (!parameters.GetBooleanParameter("use-rashomon-multiplier")) {
           solver_parameters.rashomon_multiplier = 3.0;
       } else {
           solver_parameters.rashomon_multiplier = parameters.GetFloatParameter("rashomon-multiplier");
       }
	}
	
	template <class OT>
	void Solver<OT>::InitializeTest(const ADataView& _test_data, bool reset) {
		// If the training data is the same, the cache does not need to be repopulated
		// Except if the hyper tune configuration sets reset to true
		if (!reset && org_test_data == _test_data) return;

		// Update the data objects, and inform the task about the new data
		org_test_data = _test_data;
		PreprocessTestData(org_test_data, test_data);
		test_summary = DataSummary(test_data);
		task->InformTestData(test_data, test_summary);

		data_splitter.Clear(true);
	}

	template <class OT>
	void Solver<OT>::ResetCache() {
		if (cache != nullptr) delete cache;
		if (similarity_lower_bound_computer != nullptr) delete similarity_lower_bound_computer;
		cache = new Cache<OT>(parameters, MAX_DEPTH, train_data.Size());
		if (!solver_parameters.use_lower_bounding) cache->DisableLowerBoundCaching();
		// Only use the similarity lower bound computer if the optimization task is element-wise additive
		if constexpr (OT::element_additive) {
			similarity_lower_bound_computer = new SimilarityLowerBoundComputer<OT>(task,
				NumLabels(), MAX_DEPTH, int(parameters.GetIntegerParameter("max-num-nodes")), train_data.Size());
			if (!solver_parameters.similarity_lb) similarity_lower_bound_computer->Disable();
		} else {
			solver_parameters.similarity_lb = false;
		}
	}

	template <class OT>
	void Solver<OT>::IncrementalResetCache() {
		if constexpr (OT::element_additive) {
			if (similarity_lower_bound_computer) similarity_lower_bound_computer->Reset();
		}
	}

	template <class OT>
	typename Solver<OT>::SolContainer Solver<OT>::SolveSubTree(ADataView & data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer UB_, int max_depth, int num_nodes) {
		runtime_assert(0 <= max_depth && max_depth <= num_nodes);
		if (!stopwatch.IsWithinTimeLimit()) { return InitializeSol<OT>(); }

		const Branch& branch = context.GetBranch();
		auto UB = CopySol<OT>(UB_); // Make a copy  of the UB, because we are going to update it, but these updates should not be passed to higher nodes

		ReduceNodeBudget(data, context, UB, max_depth, num_nodes);

		if (max_depth == 0 || num_nodes == 0) {
			return SolveLeafNode(data, context, UB);
		}

		// Check Cache
		{
			auto results = cache->RetrieveOptimalAssignment(data, branch, max_depth, num_nodes);
			if (!CheckEmptySol<OT>(results)) {
				return results;
			}
		}

		auto solutions = InitializeSol<OT>();
		auto leaf_solutions = InitializeSol<OT>();
		if (solver_parameters.use_lower_bounding) {
			if constexpr (OT::element_additive) {
				// Update the cache using the similarity-based lower bound
				// If an optimal solution was found in the process, return it
				bool updated_optimal_solution = UpdateCacheUsingSimilarity(data, branch, max_depth, num_nodes);
				if (updated_optimal_solution) {
					auto results = cache->RetrieveOptimalAssignment(data, branch, max_depth, num_nodes);
					if (!CheckEmptySol<OT>(results)) {
						//stats.num_cache_hit_optimality++;
						return results;
					}
				}
			}

			//Check LB > UB and return infeasible if true
			auto lower_bound = InitializeLB<OT>();
			ComputeLowerBound(data, context, lower_bound, max_depth, num_nodes, false);

			if (solver_parameters.use_upper_bounding && UBStrictDominatesRight<OT>(UB, lower_bound, rashomon_bound_delta)) {
				return InitializeSol<OT>();
			}

			//Check lower-bound vs leaf node solution and return if same
			auto empty_UB = InitializeSol<OT>();
			leaf_solutions = SolveLeafNode(data, context, empty_UB);
			if (UBStrictDominatesRight<OT>(leaf_solutions, leaf_solutions, rashomon_bound_delta)) {
				if (!CheckEmptySol<OT>(leaf_solutions)) {
					cache->StoreOptimalBranchAssignment(data, branch, leaf_solutions, max_depth, num_nodes);
				} else {
					cache->UpdateLowerBound(data, branch, UB, max_depth, num_nodes);
				}
				return leaf_solutions;
			}

			solutions = SolveLeafNode(data, context, UB);
			if (!SatisfiesMinimumLeafNodeSize(data, 2))
				return solutions;
			
			// search for solutions in the cache for lower depth limits
			int current_depth = branch.Depth();
			for (int lower_max_depth = max_depth - 1; lower_max_depth > 0; lower_max_depth--) {
				int lower_num_nodes = std::min(num_nodes, 1 << (lower_max_depth)-1);
				if (cache->IsOptimalAssignmentCached(data, branch, lower_max_depth, lower_num_nodes)) {
					const auto cached_sols = cache->RetrieveOptimalAssignment(data, branch, lower_max_depth, lower_num_nodes);
					if (solver_parameters.use_upper_bounding && LeftStrictDominatesRight<OT>(UB, cached_sols)) break;
						AddSols<OT>(task, current_depth, solutions, cached_sols);
						AddSols<OT>(UB, cached_sols);
					break;
				}
			}
			auto branch_lb = InitializeLB<OT>();
			ComputeLowerBound(data, context, branch_lb, max_depth, num_nodes, true);
			if (UBStrictDominatesRight<OT>(solutions, branch_lb, rashomon_bound_delta)) return solutions;
		} else {
			solutions = SolveLeafNode(data, context, UB);
		}

		// Use the specialised algorithm for small trees
		if constexpr (OT::use_terminal) {
			if (IsTerminalNode(max_depth, num_nodes)) {
				return SolveTerminalNode(data, context, UB, max_depth, num_nodes);
			}
		}

		// In all other cases, run the recursive general case
		return SolveSubTreeGeneralCase(data, context, UB, solutions, max_depth, num_nodes);
	}

	template <class OT>
	typename Solver<OT>::SolContainer Solver<OT>::SolveSubTreeIncrementalUB(ADataView& data, const Solver<OT>::Context& context, typename Solver<OT>::SolType UB, int max_depth, int num_nodes) {
		auto _UB = InitializeSol<OT>();
		
		SolContainer tempUB = InitializeSol<OT>();
		auto leaf_sol = SolveLeafNode(data, context, tempUB);
		SolType branching_costs = GetBranchingCosts(data, context, 0);
		SolType current_UB = UseIncrementalRashomonBound()
			? std::min(UB + SOLUTION_PRECISION, double(leaf_sol.solution + branching_costs))
			: UB + SOLUTION_PRECISION;
		SolType UB_min_step = current_UB;

		auto solution = InitializeSol<OT>();
		while (current_UB < UB || CheckEmptySol<OT>(solution)) {
			if (current_UB >= UB) _UB = InitializeSol<OT>();
			else _UB.solution = current_UB;
			solution = SolveSubTree(data, context, _UB, max_depth, num_nodes);
			if (solution.solution <= _UB.solution && solution.solution <= UB) break;
			
			double delta = std::min(double(UB_min_step), double((UB + SOLUTION_PRECISION) - current_UB));
			if (delta < 0.2 * UB) {
				current_UB = (UB + SOLUTION_PRECISION);
			} else {
				current_UB += 0.5 * delta;
			}
		}
		return solution;
	}

	template <class OT>
	typename Solver<OT>::SolContainer Solver<OT>::SolveSubTreeGeneralCase(ADataView& data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer& UB, typename Solver<OT>::SolContainer& solutions, int max_depth, int num_nodes) {
		runtime_assert(max_depth <= num_nodes);

		const Branch& branch = context.GetBranch();
		auto orgUB = CopySol<OT>(UB); // Copy the original UB and keep it, in case this branch is infeasible. The original UB can then be stored as LB
		auto infeasible_lb = InitializeSol<OT>();
		int current_depth = branch.Depth();
		const int max_size_subtree = std::min((1 << (max_depth - 1)) - 1, num_nodes - 1); //take the minimum between a full tree of max_depth or the number of nodes - 1
		const int min_size_subtree = num_nodes - 1 - max_size_subtree;
		typename Solver<OT>::SolContainer lb, left_lower_bound, right_lower_bound;
		
		auto branch_lb = InitializeLB<OT>();
		ComputeLowerBound(data, context, branch_lb, max_depth, num_nodes, true);

		// Initialize the feature selector
		std::unique_ptr<FeatureSelectorAbstract> feature_selector;
		if (parameters.GetStringParameter("feature-ordering") == "in-order") {
			feature_selector = std::make_unique<FeatureSelectorInOrder>(data.NumFeatures());
		} else if (parameters.GetStringParameter("feature-ordering") == "gini") {
			runtime_assert(!(std::is_same<typename OT::LabelType, double>::value)); // Regression does not work with Gini
			feature_selector = std::make_unique<FeatureSelectorGini>(data.NumFeatures());
		}  else { std::cout << "Unknown feature ordering strategy!" << std::endl; exit(1); }
		feature_selector->Initialize(data);

		// Loop over each feature
		int f_count = 0;
		while (feature_selector->AreThereAnyFeaturesLeft()) {

			if (solver_parameters.verbose && current_depth == 0) {
				progress_tracker.UpdateProgressCount(f_count++);
			}

			if (!stopwatch.IsWithinTimeLimit()) break;
			// if the current set of solutions equals the LB for this branch: break
			if (solver_parameters.use_lower_bound_early_stop && solver_parameters.use_lower_bounding) {
				if (UBStrictDominatesRight<OT>(solutions, branch_lb, rashomon_bound_delta)) break;
				if (solver_parameters.use_upper_bounding
					&& UBStrictDominatesRight<OT>(UB, branch_lb, rashomon_bound_delta)) break;
			}
			int feature = feature_selector->PopNextFeature();
			if (branch.HasBranchedOnFeature(feature) 
				|| !task->MayBranchOnFeature(feature)
				|| redundant_features[feature]) continue;
			auto branching_costs = GetBranchingCosts(data, context, feature);
			// Break if the current UB is lower than the constant branching costs (if applicable)
			if constexpr (Solver<OT>::sparse_objective) {
				if (solver_parameters.use_upper_bounding
					&& UBStrictDominatesRight<OT>(UB.solution + DBL_DIFF, branching_costs, rashomon_bound_delta)) break;
			}

			// Split the data and skip if the split does not meet the minimum leaf node size requirements
			ADataView left_data;
			ADataView right_data;
			data_splitter.Split(data, branch, feature, left_data, right_data);
			if (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;

			// Generate the context descriptors for the left and richt sub-branch
			Solver<OT>::Context left_context, right_context;
			task->GetLeftContext(data, context, feature, left_context);
			task->GetRightContext(data, context, feature, right_context);

			// switch the left and right branch if the left has more data
			ADataView* left_data_ptr = &left_data;
			ADataView* right_data_ptr = &right_data;
			Solver<OT>::Context* left_context_ptr = &left_context;
			Solver<OT>::Context* right_context_ptr = &right_context;
			bool swap_left_right = false;
			if (left_data.Size() < right_data.Size()) {
				swap_left_right = true;
				std::swap(left_data_ptr, right_data_ptr);
				std::swap(left_context_ptr, right_context_ptr);
			}
			const Branch& left_branch = left_context_ptr->GetBranch();
			const Branch& right_branch = right_context_ptr->GetBranch();

			// Loop over every possible way of distributing the node budget over the left and right subtrees
			for (int left_subtree_size = min_size_subtree; left_subtree_size <= max_size_subtree; left_subtree_size++) {
				int right_subtree_size = num_nodes - left_subtree_size - 1; //the '-1' is necessary since using the parent node counts as a node
				int left_depth = std::min(max_depth - 1, left_subtree_size);
				int right_depth = std::min(max_depth - 1, right_subtree_size);

				// Compute the left and right and combined LBs
				ComputeLeftRightLowerBound(feature, context, branching_costs, lb, left_lower_bound, right_lower_bound,
					*left_data_ptr, *left_context_ptr, left_depth, left_subtree_size, *right_data_ptr, *right_context_ptr, right_depth, right_subtree_size);
				if (solver_parameters.use_upper_bounding && UBStrictDominatesRight<OT>(UB, lb, rashomon_bound_delta)) {
					AddSols<OT>(infeasible_lb, lb);
					continue;
				}

				//if (solver_parameters.use_lower_bound_early_stop && SolutionsEqual<OT>(lb, solutions)) continue;
				
				// substract the right LB from the UB to get a UB for the left branch
				auto leftUB = InitializeSol<OT>();
				SubtractUBs(context, UB, right_lower_bound, solutions, branching_costs, leftUB);
				
				// Solve the left branch
				auto left_solutions = SolveSubTree(*left_data_ptr, *left_context_ptr, leftUB, left_depth, left_subtree_size);

				if (!stopwatch.IsWithinTimeLimit()) break;
				if (CheckEmptySol<OT>(left_solutions)) {
					ComputeLeftRightLowerBound(feature, context, branching_costs, lb, left_lower_bound, right_lower_bound,
						*left_data_ptr, *left_context_ptr, left_depth, left_subtree_size, *right_data_ptr, *right_context_ptr, right_depth, right_subtree_size);
					AddSols<OT>(infeasible_lb, lb);
					continue;
				}

				// substract the left solutions from the UB to get a UB for the right branch
				auto rightUB = InitializeSol<OT>();
				SubtractUBs(context, UB, left_solutions, solutions, branching_costs, rightUB);

				
				// Solve the right branch
				auto right_solutions = SolveSubTree(*right_data_ptr, *right_context_ptr, rightUB, right_depth, right_subtree_size);

				if (!stopwatch.IsWithinTimeLimit()) break;
				if (CheckEmptySol<OT>(right_solutions)) {
					ComputeLeftRightLowerBound(feature, context, branching_costs, lb, left_lower_bound, right_lower_bound,
						*left_data_ptr, *left_context_ptr, left_depth, left_subtree_size, *right_data_ptr, *right_context_ptr, right_depth, right_subtree_size);
					AddSols<OT>(infeasible_lb, lb);
					continue;
				}

				// Combine left and right solutions and store the solution
				Node<OT> new_node;
				if (swap_left_right) {
					CombineSols(feature, right_solutions, left_solutions, branching_costs, new_node);
				} else {
					CombineSols(feature, left_solutions, right_solutions, branching_costs, new_node);
				}
				if (solver_parameters.use_upper_bounding && UBStrictDominatesRight<OT>(UB, new_node, rashomon_bound_delta)) {
					AddSols<OT>(infeasible_lb, new_node);
					continue;
				}
				if (LeftStrictDominatesRightSol<OT>(new_node, solutions)) {
					solutions = new_node;
					//if (current_depth == 0 && new_node.solution * solver_parameters.rashomon_multiplier < rashomon_bound_delta) {
					//	rashomon_bound_delta = new_node.solution* solver_parameters.rashomon_multiplier;
					//}
				}
				UpdateUB(context, UB, new_node);

			}

		}

		// If a feasible solution is found (better than the UB), store it in the cache
		// Or update the LB for this branch
		if (!CheckEmptySol<OT>(solutions)) {
			cache->StoreOptimalBranchAssignment(data, branch, solutions, max_depth, num_nodes);
		} else {
			if (SolutionsEqual<OT>(infeasible_lb, InitializeSol<OT>())) {
				infeasible_lb = orgUB;
			} else {
				AddSolsInv<OT>(infeasible_lb, orgUB);
			}
			cache->UpdateLowerBound(data, branch, infeasible_lb, max_depth, num_nodes);
		}
		if constexpr (OT::element_additive) {
			similarity_lower_bound_computer->UpdateArchive(data, branch, max_depth);
		}

		return solutions;
	}

	template <class OT>
	template <typename U, typename std::enable_if<U::use_terminal, int>::type>
	typename Solver<OT>::SolContainer Solver<OT>::SolveTerminalNode(ADataView& data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer& UB, int max_depth, int num_nodes) {
		const Branch& branch = context.GetBranch();
		runtime_assert(max_depth <= 2 && 1 <= num_nodes && num_nodes <= 3 && max_depth <= num_nodes);
		runtime_assert(num_nodes != 3 || !cache->IsOptimalAssignmentCached(data, branch, 2, 3));
		runtime_assert(num_nodes != 2 || !cache->IsOptimalAssignmentCached(data, branch, 2, 2));
		runtime_assert(num_nodes != 1 || !cache->IsOptimalAssignmentCached(data, branch, 1, 1));

		stats.num_terminal_nodes_with_node_budget_one += (num_nodes == 1);
		stats.num_terminal_nodes_with_node_budget_two += (num_nodes == 2);
		stats.num_terminal_nodes_with_node_budget_three += (num_nodes == 3);

		DebugBranch(branch);

		// To maximize efficiency, use the terminal solver that already has computed frequency counts
		// for a dataset that is most similar to the new dataset
		clock_t clock_start = clock();
		int diff1 = terminal_solver1->ProbeDifference(data);
		int diff2 = terminal_solver2->ProbeDifference(data);
		TerminalSolver<OT>* tsolver = diff1 < diff2 ? terminal_solver1 : terminal_solver2;
		TerminalResults<OT>& results = tsolver->Solve(data, context, UB, rashomon_bound_delta, num_nodes);
		stats.time_in_terminal_node += double(clock() - clock_start) / CLOCKS_PER_SEC;

		// Store solutions in the cache for different node budgets
		if (!cache->IsOptimalAssignmentCached(data, branch, 1, 1)) {
			auto& one_node_solutions = results.one_node_solutions;
			if (!CheckEmptySol<OT>(one_node_solutions)) {
				cache->StoreOptimalBranchAssignment(data, branch, one_node_solutions, 1, 1);
			} else {
				cache->UpdateLowerBound(data, branch, UB, 1, 1);
			}
		}
		if (!cache->IsOptimalAssignmentCached(data, branch, 2, 2)) {
			auto& two_nodes_solutions = results.two_nodes_solutions;
			if (!CheckEmptySol<OT>(two_nodes_solutions)) {
				cache->StoreOptimalBranchAssignment(data, branch, two_nodes_solutions, 2, 2);
			} else {
				cache->UpdateLowerBound(data, branch, UB, 2, 2);
			}
		}
		if (!cache->IsOptimalAssignmentCached(data, branch, 2, 3)) {
			auto& three_nodes_solutions = results.three_nodes_solutions;
			if (!CheckEmptySol<OT>(three_nodes_solutions)) {
				cache->StoreOptimalBranchAssignment(data, branch, three_nodes_solutions, 2, 3);
			} else {
				cache->UpdateLowerBound(data, branch, UB, 2, 3);
			}
		}
		if constexpr (OT::element_additive) {
			similarity_lower_bound_computer->UpdateArchive(data, branch, max_depth);
		}

		// Return solutions based on node budget
		if (num_nodes == 1) {
			if (UBStrictDominatesRight<OT>(UB, results.one_node_solutions, rashomon_bound_delta)) return InitializeSol<OT>();
			return CopySol<OT>(results.one_node_solutions);
		} else if (num_nodes == 2) {
			if (UBStrictDominatesRight<OT>(UB, results.two_nodes_solutions, rashomon_bound_delta)) return InitializeSol<OT>();
			return CopySol<OT>(results.two_nodes_solutions);
		}
		if (UBStrictDominatesRight<OT>(UB, results.three_nodes_solutions, rashomon_bound_delta)) return InitializeSol<OT>();
		return CopySol<OT>(results.three_nodes_solutions);
	}

    template <class OT>
    std::pair<std::shared_ptr<std::vector<std::shared_ptr<typename Solver<OT>::SolutionTracker>>>,int> Solver<OT>::SolveRashomonTerminalNode(ADataView& data, const typename Solver<OT>::Context& context, typename Solver<OT>::SolContainer& UB, int max_depth, int num_nodes) {
		if constexpr (OT::use_terminal) {
			const Branch& branch = context.GetBranch();
			runtime_assert(max_depth <= 2 && 1 <= num_nodes && num_nodes <= 3 && max_depth <= num_nodes);

			stats.num_terminal_nodes_with_node_budget_one += (num_nodes == 1);
			stats.num_terminal_nodes_with_node_budget_two += (num_nodes == 2);
			stats.num_terminal_nodes_with_node_budget_three += (num_nodes == 3);

			DebugBranch(branch);

			// To maximize efficiency, use the terminal solver that already has computed frequency counts
			// for a dataset that is most similar to the new dataset
			clock_t clock_start = clock();
			int diff1 = rashomon_terminal_solver1->ProbeDifference(data);
			int diff2 = rashomon_terminal_solver2->ProbeDifference(data);
			RashomonTerminalSolver<OT>* tsolver = diff1 < diff2 ? rashomon_terminal_solver1 : rashomon_terminal_solver2;
			RashomonTerminalResults<OT>& results = tsolver->Solve(data, context, UB, num_nodes);
			stats.time_rashomon_terminal += double(clock() - clock_start) / CLOCKS_PER_SEC;

			if constexpr (OT::element_additive) {
				similarity_lower_bound_computer->UpdateArchive(data, branch, max_depth);
			}

			// Return solutions based on node budget
	//        if (num_nodes == 1) {
	////            if (LeftStrictDominatesRight<OT>(UB, results.one_node_solutions)) return InitializeSol<OT>();
	//            return CopySol<OT>(results.one_node_solutions)->GetSolutions();
	//        } else if (num_nodes == 2) {
	////            if (LeftStrictDominatesRight<OT>(UB, results.two_nodes_solutions)) return InitializeSol<OT>();
	//            return CopySol<OT>(results.two_nodes_solutions)->GetSolutions();
	//        }
	//        if (LeftStrictDominatesRight<OT>(UB, results.three_nodes_solutions)) return InitializeSol<OT>();
	//        return CopySol<OT>(results.solutions)->GetSolutions();
			return std::make_pair(results.solutions, results.num_solutions);
		}
		throw std::runtime_error("Terminal solver not implemented for this optimization task.");
    }

	template<class OT>
	typename Solver<OT>::SolContainer Solver<OT>::SolveLeafNode(const ADataView& data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer& UB) const {
		if (!SatisfiesMinimumLeafNodeSize(data)) return InitializeSol<OT>();
		const Branch& branch = context.GetBranch();

		// If the optimization task has a custom leaf node function defined, use it
		if constexpr (OT::custom_leaf) {
			// Return the optimal feasible solution, requires custom implementation
			auto sol = task->SolveLeafNode(data, context);
			if (solver_parameters.use_upper_bounding && UBStrictDominatesRight<OT>(UB, sol, rashomon_bound_delta)) return InitializeSol<OT>();
			UpdateUB(context, UB, sol);
			return sol;
		} else { // Otherwise, for each possible label, calculate the costs and store if better
			auto result = InitializeSol<OT>();

			for (int label = 0; label < data.NumLabels(); label++) {
				auto sol = Node<OT>(label, task->GetLeafCosts(data, context, label));
				if (!SatisfiesConstraint(sol, context)) continue;
				if (solver_parameters.use_upper_bounding && UBStrictDominatesRight<OT>(UB, sol, rashomon_bound_delta)) continue;
				AddSol<OT>(task, branch.Depth(), result, sol);
				UpdateUB(context, UB, sol);
			}
			return result;
		}
	}

	template <class OT>
	void Solver<OT>::ComputeLowerBound(ADataView& data, const typename Solver<OT>::Context& context, typename Solver<OT>::SolContainer& lb, int depth, int num_nodes, bool root_is_branching_node) {
		lb = InitializeLB<OT>();
		auto& branch = context.GetBranch();
		if (solver_parameters.use_lower_bounding) {
			auto cached_lb = cache->RetrieveLowerBound(data, branch, depth, num_nodes);
			AddSolsInv<OT>(lb, cached_lb);

			auto custom_lb = InitializeLB<OT>();
			if constexpr (OT::custom_lower_bound) {
				if (solver_parameters.use_task_lower_bounding) {
					custom_lb = task->ComputeLowerBound(data, branch, depth, num_nodes);
					AddSolsInv<OT>(lb, custom_lb);
			    }
			}

			if constexpr (Solver<OT>::sparse_objective) {
				// If the search space up to depth d is fully exhausted, then any improving tree must have at least d+1 nodes
				// Therefore, apply a lower bound of (d+1) * branching_costs
				// This is similar to the hierarchical lower bound prestend by Lin et al., "Generalized and Scalable 
				// Optimal Sparse Decision Trees," ICML-20.
				auto branching_costs = GetBranchingCosts(data, context, 0); // Constant branching costs, so the feature does not matter
			    auto solutions = InitializeSol<OT>(false);
				if (!OT::expensive_leaf) {
					auto ub = InitializeSol<OT>(false);
					solutions = SolveLeafNode(data, context, ub);
				}
				int current_depth = branch.Depth();
				int max_depth_searched = 0;
				for (int lower_max_depth = depth - 1; lower_max_depth > 0; lower_max_depth--) {
					int lower_num_nodes = std::min(num_nodes, (1 << (lower_max_depth))-1);
					if (cache->IsOptimalAssignmentCached(data, branch, lower_max_depth, lower_num_nodes)) {
						AddSols<OT>(task, current_depth, solutions, cache->RetrieveOptimalAssignment(data, branch, lower_max_depth, lower_num_nodes));
						max_depth_searched = lower_max_depth;
						break;
					}
				}
				if (!reconstructing) {
					max_depth_searched = std::max(max_depth_searched, max_depth_finished - current_depth);
				}
				int delta = reconstructing || !root_is_branching_node ? 0 : 1;
				int min_nodes_to_use = max_depth_searched + delta;
				int min_nodes_to_use_extra = std::min(0, min_nodes_to_use - custom_lb.NumNodes());
				
				if (solutions.solution <= min_nodes_to_use_extra * branching_costs + custom_lb.solution) {
					// The previous solution is cheaper than splitting on at least min_nodes_to_use, so its a LB
					AddSolsInv<OT>(lb, solutions);
				} else {
					// The previous solution is worse than splitting on at least min_nodes_to_use, so use the minimum
					// number of nodes as the lower bound.
					Node<OT> split_lb(0, min_nodes_to_use_extra * branching_costs + custom_lb.solution, solutions.num_nodes_left, solutions.num_nodes_right);
					AddSolsInv<OT>(lb, split_lb);
				}
			}

		}
	}

	template <class OT>
	void Solver<OT>::ComputeLeftRightLowerBound(int feature, const typename Solver<OT>::Context& context, const typename Solver<OT>::SolType& branching_costs, 
		typename Solver<OT>::SolContainer& lb, typename Solver<OT>::SolContainer& left_lower_bound, typename Solver<OT>::SolContainer& right_lower_bound,
		ADataView& left_data, const Solver<OT>::Context& left_context, int left_depth, int left_nodes,
		ADataView& right_data, const Solver<OT>::Context& right_context, int right_depth, int right_nodes) {
		lb = InitializeLB<OT>();
		left_lower_bound = InitializeLB<OT>();
		right_lower_bound = InitializeLB<OT>();

		if (solver_parameters.use_lower_bounding) {
			auto& left_branch = left_context.GetBranch();
			auto& right_branch = right_context.GetBranch();
			ComputeLowerBound(left_data, left_context, left_lower_bound, left_depth, left_nodes, false);
			ComputeLowerBound(right_data, right_context, right_lower_bound, right_depth, right_nodes, false);
			
			CombineSols(feature, left_lower_bound, right_lower_bound, branching_costs, lb);
		}
	}
	

	template <class OT>
	void Solver<OT>::SubtractUBs(const Solver<OT>::Context& context, const typename Solver<OT>::SolContainer& UB, const typename Solver<OT>::SolContainer& sols,
			const typename Solver<OT>::SolContainer& current_solutions, const typename Solver<OT>::SolType& branching_costs, typename Solver<OT>::SolContainer& updatedUB) {
		clock_t clock_start = clock();
		const Branch& branch = context.GetBranch();
		if (solver_parameters.use_upper_bounding && solver_parameters.subtract_ub) {
			// Subtract the solution and the branching costs
			if (LeftDominatesRight<OT>(current_solutions.solution, UB.solution)) {
				OT::Subtract(current_solutions.solution - OT::minimum_difference, sols.solution, updatedUB.solution);
			} else {
				OT::Subtract(UB.solution, sols.solution, updatedUB.solution);
			}
			OT::Subtract(updatedUB.solution, branching_costs, updatedUB.solution);
		} else {
			updatedUB.solution = UB.solution;
		}
		// In the root nod of the search, feasible solutions can be relaxed by removing information that is related
		// to constraint satisfaction from the solution
		stats.time_ub_subtracting += double(clock() - clock_start) / CLOCKS_PER_SEC;
	}

	template<class OT>
	bool Solver<OT>::SatisfiesConstraint(const Node<OT>& sol, const Solver<OT>::Context& context) const {
		if constexpr (!OT::has_constraint) {
			return true;
		} else {
			return task->SatisfiesConstraint(sol, context);
		}
	}

	template <class OT>
	void Solver<OT>::UpdateUB(const Solver<OT>::Context& context, typename Solver<OT>::SolContainer& UB, Node<OT> sol) const {
		if (solver_parameters.use_upper_bounding) {
			AddSol<OT>(UB, sol);
		}
	}

	template <class OT>
	void Solver<OT>::ReduceNodeBudget(const ADataView& data, const Solver<OT>::Context& context, const typename Solver<OT>::SolContainer& UB, int& max_depth, int& num_nodes) const {
		if (!solver_parameters.use_upper_bounding) return;
		int nodes = num_nodes;
		if constexpr (Solver<OT>::sparse_objective) {
			if (UB.solution >= 0.9 * DBL_MAX) return;
			double ub = double(UB.solution) + rashomon_bound_delta;
			auto branching_costs = GetBranchingCosts(data, context, 0);

			if constexpr(std::is_same<OT, CostComplexAccuracy>::value || std::is_same<OT, CostComplexRegression>::value) {
				// Handle the special case for CostComplexAccuracy && CostComplexRegression which computes twice the
				// branching costs for the root branch to account for costs per leaf (for comparison with TreeFarms)
				if (OT::leaf_penalty && context.GetBranch().Depth() == 0) {
					branching_costs /= 2;
					ub -= branching_costs;
				}
			}

			if (branching_costs <= 0) return;
			nodes = int(std::max(0.0, std::min(double(nodes), (ub + 1e-6) / double(branching_costs))));
		}
		if (data.Size() < solver_parameters.minimum_leaf_node_size * nodes) {
			nodes = std::min(nodes, std::max((GetDataWeight(data) / solver_parameters.minimum_leaf_node_size) - 1, 0));
		}
		if (nodes < num_nodes) {
			int new_max_depth = std::min(max_depth, nodes);
			if (new_max_depth < max_depth) {
				max_depth = new_max_depth;
				num_nodes = std::min(num_nodes, (1 << max_depth) - 1);
				runtime_assert(max_depth <= num_nodes);
			}
		}
	}

	template<class OT>
	std::shared_ptr<Tree<OT>> Solver<OT>::ConstructOptimalTree(const Node<OT>& sol, ADataView& data, const Solver<OT>::Context& context, int max_depth, int num_nodes) {
		runtime_assert(num_nodes >= 0);
//		stopwatch.Disable();
		max_depth = std::min(max_depth, num_nodes);
		num_nodes = std::min(num_nodes, (1 << max_depth) - 1);

		if (max_depth == 0 || num_nodes == 0 || sol.NumNodes() == 0) {
			return Tree<OT>::CreateLabelNode(sol.label);
		}

		// Special D1 reconstruct
		bool d1 = max_depth == 1 || num_nodes == 1 || sol.NumNodes() == 1;

		// Special D2 reconstruct
		if constexpr (OT::use_terminal) {
			if (!d1 && IsTerminalNode(max_depth, num_nodes)) {
				try {
					return terminal_solver1->ConstructOptimalTree(sol, data, context, max_depth, num_nodes);
				} catch (...){} // Continue to the default way of reconstructing
			}
		}

		// Initialize empty UBs
		auto UB = InitializeSol<OT>();
		AddSol<OT>(UB, OT::worst);
		auto UBleft = InitializeSol<OT>();
		AddSol<OT>(UBleft, OT::worst);
		auto UBright = InitializeSol<OT>();
		AddSol<OT>(UBright, OT::worst);

		auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(sol.feature);

		const Branch& branch = context.GetBranch();
		ADataView left_data, right_data;
		data_splitter.Split(data, branch, sol.feature, left_data, right_data);
		runtime_assert(SatisfiesMinimumLeafNodeSize(left_data) && SatisfiesMinimumLeafNodeSize(right_data));
		Solver<OT>::Context left_context, right_context;
		task->GetLeftContext(data, context, sol.feature, left_context);
		task->GetRightContext(data, context, sol.feature, right_context);

		const int left_subtree_size = sol.num_nodes_left;
		const int right_subtree_size = sol.num_nodes_right;
		int left_depth = std::min(max_depth - 1, left_subtree_size);
		int right_depth = std::min(max_depth - 1, right_subtree_size);

		int left_size = left_subtree_size;
		int right_size = right_subtree_size;
		Solver<OT>::SolContainer left_sols, right_sols;
		
		bool use_cache = cache->UseCache();
		if (use_cache) {
			const int max_size_subtree = std::min((1 << (max_depth - 1)) - 1, num_nodes - 1); //take the minimum between a full tree of max_depth or the number of nodes - 1
			const int min_size_subtree = num_nodes - 1 - max_size_subtree;
			int min_left_subtree_size = std::max(sol.num_nodes_left, min_size_subtree);
			int min_right_subtree_size = std::max(sol.num_nodes_right, min_size_subtree);

			left_size = min_left_subtree_size;
			for (; left_size <= max_size_subtree; left_size++) {
				left_depth = std::min(max_depth - 1, left_size);
				if (left_size == 0)
					left_sols = SolveLeafNode(left_data, left_context, UBleft);
				else
					left_sols = cache->RetrieveOptimalAssignment(left_data, left_context.GetBranch(), left_depth, left_size);
				if (!CheckEmptySol<OT>(left_sols)) break;
			}

			right_size = min_right_subtree_size;
			for (; right_size <= max_size_subtree; right_size++) {
				right_depth = std::min(max_depth - 1, right_size);
				if (right_size == 0)
					right_sols = SolveLeafNode(right_data, right_context, UBright);
				else
					right_sols = cache->RetrieveOptimalAssignment(right_data, right_context.GetBranch(), right_depth, right_size);
				if (!CheckEmptySol<OT>(right_sols)) break;
			}

		}
		
		if (!use_cache || CheckEmptySol<OT>(left_sols)) {
			left_depth = std::min(max_depth - 1, left_subtree_size);
			left_sols = SolveSubTree(left_data, left_context, UBleft, left_depth, left_subtree_size);
			if (CheckEmptySol<OT>(left_sols)) {
				left_sols = SolveSubTree(left_data, left_context, UBleft, left_depth, left_subtree_size);
			}
			runtime_assert(!CheckEmptySol<OT>(left_sols));
		}
		if (!use_cache || CheckEmptySol<OT>(right_sols)) {
			right_depth = std::min(max_depth - 1, right_subtree_size);
			right_sols = SolveSubTree(right_data, right_context, UBright, right_depth, right_subtree_size);
			if (CheckEmptySol<OT>(right_sols)) {
				right_sols = SolveSubTree(right_data, right_context, UBright, right_depth, right_subtree_size);
			}
			runtime_assert(!CheckEmptySol<OT>(right_sols));
		}

		// Reconstruct Merge
		tree->left_child = ConstructOptimalTree(left_sols, left_data, left_context, left_depth, left_size);
		tree->right_child = ConstructOptimalTree(right_sols, right_data, right_context, right_depth, right_size);	
		
		return tree;
	}


    template<class OT>
    std::shared_ptr<Tree<OT>> Solver<OT>::ConstructD1Tree(ADataView data, int feature, Solver<OT>::Context context, ADataView& left_data, ADataView& right_data, Solver<OT>::Context& left_context, Solver<OT>::Context& right_context) {

        auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature);
        const Branch &branch = context.GetBranch();
        data_splitter.Split(data, branch, feature, left_data, right_data);
        if (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) {
            return nullptr;
        }
        task->GetLeftContext(data, context, feature, left_context);
        task->GetRightContext(data, context, feature, right_context);
        Solver<OT>::SolContainer right_child_left_sols, right_child_right_sols;
        return tree;
    }
    template<class OT>
    void Solver<OT>::Generate2FeatureRashomonSet(std::shared_ptr<SolverTaskResult<OT>>& result, SolType threshold, ADataView& data, int max_depth, int max_num_nodes) {
		if (max_depth == 0 || max_num_nodes < 3) return;
        // Initialize empty UBs
        auto rashomonUB = InitializeSol<OT>();
		AddSol<OT>(rashomonUB, threshold);
		auto UB = rashomonUB;
		
        Solver<OT>::Context base_context;
		double branching_cost = GetBranchingCosts(data, base_context, 0) / 2; // Divide by two, because the root note gets twice the costs
        Solver<OT>::SolContainer root_sol = SolveLeafNode(data, base_context, UB);
        if (root_sol.solution <= threshold) {
                auto tree = Tree<OT>::CreateLabelNode(root_sol.label);
                auto score = InternalTrainScore<OT>::ComputeTrainPerformance(&data_splitter, task, tree.get(), data);
                result->AddSolution(tree,score);
        }


        for (int feature1 = 0; feature1 < data.NumFeatures(); ++feature1){

            Solver<OT>::Context context;
            ADataView left_data, right_data;
            Solver<OT>::Context left_context, right_context;
            std::shared_ptr<Tree<OT>> root_node = ConstructD1Tree(data, feature1, context,left_data, right_data, left_context, right_context);
            if (root_node == nullptr) continue;

            Solver<OT>::SolContainer left_sols, right_sols;
			UB = rashomonUB;
            left_sols = SolveLeafNode(left_data, left_context, UB);
			UB = rashomonUB;
            right_sols = SolveLeafNode(right_data, right_context, UB);

            // Construct tree with one branching node
            if (!CheckEmptySol<OT>(left_sols) && !CheckEmptySol<OT>(right_sols)) {
                auto error = (left_sols.solution + right_sols.solution);
                if (error + 2 * branching_cost <= threshold) {
                    auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature1);
                    tree->left_child = Tree<OT>::CreateLabelNode(left_sols.label);
                    tree->right_child = Tree<OT>::CreateLabelNode(right_sols.label);
                    auto score = InternalTrainScore<OT>::ComputeTrainPerformance(&data_splitter, task, tree.get(), data);
                    result->AddSolution(tree,score);
                }
            }
            if (max_depth == 1) continue;
            for (int feature2 = 0; feature2 < data.NumFeatures(); ++feature2) {

                if (feature1 == feature2) continue;

                ADataView left_child_left_data, left_child_right_data;
                Solver<OT>::Context left_child_left_context, left_child_right_context;
                Solver<OT>::SolContainer left_child_left_sols, left_child_right_sols;
                auto tree_left = ConstructD1Tree(left_data, feature2, left_context,left_child_left_data, left_child_right_data, left_child_left_context, left_child_right_context);
                if( tree_left != nullptr) {
					UB = rashomonUB;
                    left_child_left_sols = SolveLeafNode(left_child_left_data, left_child_left_context, UB);
					UB = rashomonUB;
                    left_child_right_sols = SolveLeafNode(left_child_right_data, left_child_right_context, UB);
                }

                ADataView right_child_left_data, right_child_right_data;
                Solver<OT>::Context right_child_left_context, right_child_right_context;
                Solver<OT>::SolContainer right_child_left_sols, right_child_right_sols;
                auto tree_right = ConstructD1Tree(right_data, feature2, right_context,right_child_left_data, right_child_right_data, right_child_left_context, right_child_right_context);
                if(tree_right != nullptr) {
					UB = rashomonUB;
                    right_child_left_sols = SolveLeafNode(right_child_left_data, right_child_left_context, UB);
					UB = rashomonUB;
                    right_child_right_sols = SolveLeafNode(right_child_right_data, right_child_right_context, UB);
                }

                // Construct tree with one parent branching node and one right branching node
                if (!CheckEmptySol<OT>(left_sols) && !CheckEmptySol<OT>(right_child_left_sols) && !CheckEmptySol<OT>(right_child_right_sols)) {
                    auto error = (left_sols.solution + right_child_left_sols.solution + right_child_right_sols.solution);
                    if (error + 3 * branching_cost <= threshold) {
                        tree_right->left_child = Tree<OT>::CreateLabelNode(right_child_left_sols.label);
                        tree_right->right_child = Tree<OT>::CreateLabelNode(right_child_right_sols.label);
                        auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature1);
                        tree->left_child = Tree<OT>::CreateLabelNode(left_sols.label);
                        tree->right_child = tree_right;
                        auto score = InternalTrainScore<OT>::ComputeTrainPerformance(&data_splitter, task, tree.get(), data);
                        result->AddSolution(tree,score);
                    }
                }
                // Construct tree with one parent branching node and one left branching node
                if (!CheckEmptySol<OT>(right_sols) && !CheckEmptySol<OT>(left_child_left_sols) && !CheckEmptySol<OT>(left_child_right_sols)) {
                    auto error = (right_sols.solution + left_child_left_sols.solution + left_child_right_sols.solution);
                    if (error + 3 * branching_cost <= threshold) {
                        tree_left->left_child = Tree<OT>::CreateLabelNode(left_child_left_sols.label);
                        tree_left->right_child = Tree<OT>::CreateLabelNode(left_child_right_sols.label);
                        auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature1);
                        tree->left_child = tree_left;
                        tree->right_child = Tree<OT>::CreateLabelNode(right_sols.label);
                        auto score = InternalTrainScore<OT>::ComputeTrainPerformance(&data_splitter, task, tree.get(), data);
                        result->AddSolution(tree,score);
                    }
                }

                if(tree_left == nullptr && tree_right == nullptr) continue;
                if (max_num_nodes < 3) continue;
                for (int feature3 = feature2; feature3 < data.NumFeatures(); ++feature3) {
                    if (feature1 == feature3) continue;

                    ADataView left_child_left_data2, left_child_right_data2;
                    Solver<OT>::Context left_child_left_context2, left_child_right_context2;
                    Solver<OT>::SolContainer left_child_left_sols2, left_child_right_sols2;
                    std::shared_ptr<Tree<OT>> tree_left2;
                    if (tree_right != nullptr){
                        tree_left2 = ConstructD1Tree(left_data, feature3, left_context,left_child_left_data2, left_child_right_data2, left_child_left_context2, left_child_right_context2);
                        if( tree_left2 != nullptr) {
							UB = rashomonUB;
                            left_child_left_sols2 = SolveLeafNode(left_child_left_data2, left_child_left_context2, UB);
							UB = rashomonUB;
                            left_child_right_sols2 = SolveLeafNode(left_child_right_data2, left_child_right_context2, UB);
                        }
                    }

                    if (!CheckEmptySol<OT>(right_child_left_sols) && !CheckEmptySol<OT>(right_child_right_sols) && !CheckEmptySol<OT>(left_child_left_sols2) && !CheckEmptySol<OT>(left_child_right_sols2)) {
                        auto error = (right_child_left_sols.solution + right_child_right_sols.solution + left_child_left_sols2.solution + left_child_right_sols2.solution);
                        if (error + 4 * branching_cost <= threshold) {
                            tree_left2->left_child = Tree<OT>::CreateLabelNode(left_child_left_sols2.label);
                            tree_left2->right_child = Tree<OT>::CreateLabelNode(left_child_right_sols2.label);
                            auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature1);
                            tree->left_child = tree_left2;
                            if (tree_right->left_child == nullptr) {
                                tree_right->left_child = Tree<OT>::CreateLabelNode(right_child_left_sols.label);
                                tree_right->right_child = Tree<OT>::CreateLabelNode(right_child_right_sols.label);
                            }
                            tree->right_child = tree_right;
                            auto score = InternalTrainScore<OT>::ComputeTrainPerformance(&data_splitter, task, tree.get(), data);
                            result->AddSolution(tree,score);
                        }
                    }

                    if (feature2 == feature3) continue;
                    ADataView right_child_left_data2, right_child_right_data2;
                    Solver<OT>::Context right_child_left_context2, right_child_right_context2;
                    Solver<OT>::SolContainer right_child_left_sols2, right_child_right_sols2;
                    std::shared_ptr<Tree<OT>> tree_right2;
                    if (tree_left != nullptr){
                        tree_right2 = ConstructD1Tree(right_data, feature3, right_context,right_child_left_data2, right_child_right_data2, right_child_left_context2, right_child_right_context2);
                        if(tree_right2 != nullptr) {
							UB = rashomonUB;
                            right_child_left_sols2 = SolveLeafNode(right_child_left_data2, right_child_left_context2, UB);
							UB = rashomonUB;
                            right_child_right_sols2 = SolveLeafNode(right_child_right_data2, right_child_right_context2, UB);
                        }
                    }


                    if (!CheckEmptySol<OT>(right_child_left_sols2) && !CheckEmptySol<OT>(right_child_right_sols2) && !CheckEmptySol<OT>(left_child_left_sols) && !CheckEmptySol<OT>(left_child_right_sols)) {
                        auto error = (right_child_left_sols2.solution + right_child_right_sols2.solution + left_child_left_sols.solution + left_child_right_sols.solution);
                        if (error + 4 * branching_cost <= threshold) {
                            tree_right2->left_child = Tree<OT>::CreateLabelNode(right_child_left_sols2.label);
                            tree_right2->right_child = Tree<OT>::CreateLabelNode(right_child_right_sols2.label);
                            auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature1);
                            if (tree_left->left_child == nullptr) {
                                tree_left->left_child = Tree<OT>::CreateLabelNode(left_child_left_sols.label);
                                tree_left->right_child = Tree<OT>::CreateLabelNode(left_child_right_sols.label);
                            }
                            tree->left_child = tree_left;
                            tree->right_child = tree_right2;
                            auto score = InternalTrainScore<OT>::ComputeTrainPerformance(&data_splitter, task, tree.get(), data);
                            result->AddSolution(tree,score);
                        }
                    }
                }
            }
        }
    }

    template<class OT>
	void Solver<OT>::ConstructBruteForceRashomonSet(std::shared_ptr<SolverTaskResult<OT>>& result, const Node<OT>& sol, ADataView& data, int max_depth, int num_nodes) {
		runtime_assert(num_nodes >= 0);
		max_depth = std::min(max_depth, num_nodes);
        auto threshold = sol.solution * (1 + solver_parameters.rashomon_multiplier);

        Generate2FeatureRashomonSet(result,threshold, train_data, max_depth, num_nodes);

        std::vector<std::string> tree_features;
        for (auto tree : result->trees) {
            tree_features.push_back(tree->SortedFeatures());
        }
        std::sort(tree_features.begin(), tree_features.end());

        std::vector<int> tree_scores;
        for (auto score : result->optimal_scores) {
            tree_scores.push_back(score->score * data.Size());
        }
        std::sort(tree_scores.begin(), tree_scores.end());
    }

    template<class OT>
	void Solver<OT>::ConstructRashomonSet(std::shared_ptr<SolverTaskResult<OT>>& result) {
		clock_t clock_start = clock();
		size_t start_solution_index = 0;
		bool track_tree_statistics = parameters.GetBooleanParameter("track-tree-statistics");
		int max_depth = int(parameters.GetIntegerParameter("max-depth"));

		if (!root_tracker) {
			
			rashomon_bound_delta = 0;

			int num_nodes = (1 << max_depth) - 1;
			Solver<OT>::Context base_context;
			root_tracker = AbstractTracker<OT>::CreateTracker(this, cache, train_data, base_context, max_depth, num_nodes, rashomon_bound);

			if (track_tree_statistics) {
				result->root_feature_stats.resize(train_data.NumFeatures(), 0);
				result->feature_stats.resize(train_data.NumFeatures(), 0);
				result->num_nodes_stats.resize(num_nodes + 1, 0);
			}
			result->rashomon_set_size = 0;
			result->rashomon_solutions = std::make_shared<std::vector<std::shared_ptr<SolutionTracker>>>();
			result->root_solution_counts.clear();

			stats.time_rashomon_init = double(clock() - clock_start) / CLOCKS_PER_SEC;
		} else {
			if (!result) {
				std::cout << "Result object is not initialized!" << std::endl;
				std::exit(1);
			} else if (!result->rashomon_solutions) {
				std::cout << "Result Rashomon solution list is not initialized!" << std::endl;
				std::exit(1);
			}

			// Continue where the last result left
			start_solution_index = result->rashomon_solutions->size();

			// Reinitialize the stopwatch
			stopwatch.Initialise(parameters.GetFloatParameter("time"));
		}

		size_t max_num_trees = parameters.GetIntegerParameter("max-num-trees");
		size_t solution_index = start_solution_index;
		result->is_exhausted = true; // assume we exit because of exhausting the Rashomon set
		while (root_tracker->HasNSolution(solution_index)) {
			auto& sol = root_tracker->GetSolutionN(solution_index++);
			if (sol->obj > rashomon_bound + SOLUTION_PRECISION) break;

			result->rashomon_set_size += sol->GetNumSolutions();
			if (result->rashomon_set_size >= max_num_trees || !stopwatch.IsWithinTimeLimit()) {
				result->is_exhausted = false;
				break;
			}
		}

		if (solver_parameters.verbose) {
			std::cout << "Number of solutions in the Rashomon set: " << result->rashomon_set_size << std::endl;
			std::cout << "solver count: " << sol_count << std::endl; // TODO change to more meaningful print statement
		}

		// Copy the solutions from the root-tracker to the result object
		auto& root_solutions = root_tracker->GetSolutions();
		if (start_solution_index == 0) {
			result->rashomon_solutions->assign(root_solutions.begin(), root_solutions.begin() + solution_index);
		} else {
			result->rashomon_solutions->insert(result->rashomon_solutions->end(), root_solutions.begin() + start_solution_index, root_solutions.begin() + solution_index);
		}
		// Compute the cumulative counts of solutions up to each solution-set
		result->root_solution_counts.reserve(result->rashomon_solutions->size());
		size_t cumulative_rashomon_size = 0;
		for (auto& sol : *(result->rashomon_solutions)) {
			cumulative_rashomon_size += sol->GetNumSolutions();
			result->root_solution_counts.push_back(cumulative_rashomon_size - 1);
		}
		
        stats.time_rashomon_total = (double(clock() - clock_start) / CLOCKS_PER_SEC);

        result->num_active_split_tracker_per_depth.resize(max_depth);
        result->num_average_solution_per_split_tracker.resize(max_depth);
        for (size_t i = 0; i < rashomon_sol_stats.size(); ++i){
            auto depth_stats = rashomon_sol_stats[i];
            size_t sols_size = 0;
            size_t total_count = 0;
            for (auto& pair: depth_stats) {
                if (pair.second > 0) result->num_active_split_tracker_per_depth[i] += pair.second;
                sols_size += pair.first;
                total_count += pair.second;
            }
            result->num_average_solution_per_split_tracker[i] = total_count != 0 ? sols_size / total_count : 0;
        }

        if (track_tree_statistics) PrepareRashomonStatistics(result);
	}

    template <class OT>
	template <typename U, typename std::enable_if<U::element_additive, int>::type>
	bool Solver<OT>::UpdateCacheUsingSimilarity(ADataView& data, const Branch& branch, int max_depth, int num_nodes) {
		PairLowerBoundOptimal<OT> result = similarity_lower_bound_computer->ComputeLowerBound(data, branch, max_depth, num_nodes, cache);
		if (CheckEmptySol<OT>(result.lower_bound)) return false;
		if (result.optimal) { return true; }
		static SolContainer empty_sol = InitializeLB<OT>();
		if (!SolutionsEqual<OT>(empty_sol, result.lower_bound)) {
			cache->UpdateLowerBound(data, branch, result.lower_bound, max_depth, num_nodes);
		}
		return false;

	}

	template <class OT>
	typename Solver<OT>::SolType Solver<OT>::GetBranchingCosts(const ADataView& data, const Solver<OT>::Context& context, int feature) const {
		if constexpr (!OT::has_branching_costs) {
			return OT::best;
		} else {
			return task->GetBranchingCosts(data, context, feature);
		}
	}

	template <class OT>
	int Solver<OT>::GetDataWeight(const ADataView& data) const {
		if constexpr (!OT::use_weights) return data.Size();
		int weight = 0;
		for (int k = 0; k < data.NumLabels(); k++) {
			for (auto& i : data.GetInstancesForLabel(k)) {
				weight += int(i->GetWeight());
			}
		}
		return weight;
	}

	template <class OT>
	bool Solver<OT>::SatisfiesMinimumLeafNodeSize(const ADataView& data, int multiplier) const {
		int mlsz = solver_parameters.minimum_leaf_node_size * multiplier;
		if constexpr (OT::use_weights) {
			int weight = 0;
			for (int k = 0; k < data.NumLabels(); k++) {
				for (auto& i : data.GetInstancesForLabel(k)) {
					weight += int(i->GetWeight());
					if (weight >= mlsz) return true;
				}
			}
			return false;
		} else {
			return data.Size() >= mlsz;
		}
	}

	template <class OT>
	void Solver<OT>::PreprocessData(AData& data, bool train) {
		if (train) {
			redundant_features.clear();
			redundant_features.resize(data.NumFeatures(), 0);
			// Flip features that are satisified more than 50% of the time
			flipped_features.clear();
			flipped_features.resize(data.NumFeatures(), 0);
			for (int f = 0; f < data.NumFeatures(); f++) {
				if (!task->MayBranchOnFeature(f)) continue;
				int positive_count = 0;
				for (int i = 0; i < data.Size(); i++) {
					auto instance = data.GetInstance(i);
					if (instance->IsFeaturePresent(f))
						positive_count++;
				}
				if (positive_count > data.Size() / 2) {
					// Flip this feature, to improve the performance of the D2-solver
					flipped_features[f] = 1;
					for (int i = 0; i < data.Size(); i++) {
						auto instance = data.GetMutableInstance(i);
						instance->FlipFeature(f);
					}
				}
				if (positive_count < solver_parameters.minimum_leaf_node_size 
					|| positive_count > data.Size() - solver_parameters.minimum_leaf_node_size) {
					redundant_features[f] = true;
				}
			}
			// Find duplicate features and turn them off
			for (int f1 = 0; f1 < data.NumFeatures() - 1; f1++) {
				if (redundant_features[f1]) continue;
				for (int f2 = f1 + 1; f2 < data.NumFeatures(); f2++) {
					if (redundant_features[f2]) continue;
					bool same = true;
					for (int i = 0; i < data.Size(); i++) {
						auto instance = data.GetInstance(i);
						if (instance->IsFeaturePresent(f1) != instance->IsFeaturePresent(f2)) {
							same = false;
							break;
						}
					}
					if (same) {
						redundant_features[f2] = true;
					}
				}
			}
			for (int f = 0; f < data.NumFeatures(); f++) {
				if (redundant_features[f]) {
					for (int i = 0; i < data.Size(); i++) {
						auto instance = data.GetMutableInstance(i);
						instance->DisableFeature(f);
						runtime_assert(!instance->IsFeaturePresent(f));
					}
				}
			}

			for (int i = 0; i < data.Size(); i++) {
				data.GetMutableInstance(i)->ComputeFeaturePairIndices();
			}
		} else {
			for (int f = 0; f < data.NumFeatures(); f++) {
				if (flipped_features[f] == 1) {
					for (int i = 0; i < data.Size(); i++) {
						auto instance = data.GetMutableInstance(i);
						instance->FlipFeature(f);
					}
				}
			}
		}
		if constexpr (OT::preprocess_data) {
			task->PreprocessData(data, train);
		}
	}

	template <class OT>
	void Solver<OT>::PreprocessTrainData(const ADataView& org_train_data, ADataView& train_data) {
		train_data = org_train_data;
		if constexpr (OT::preprocess_train_test_data) {
			task->PreprocessTrainData(train_data);
		}
	}

	template <class OT>
	void Solver<OT>::PreprocessTestData(const ADataView& org_test_data, ADataView& test_data) {
		test_data = org_test_data;
		if constexpr (OT::preprocess_train_test_data) {
			task->PreprocessTestData(test_data);
		}
	}

	template <class OT>
	void Solver<OT>::PostProcessTree(std::shared_ptr<Tree<OT>> tree) {
		tree->FlipFlippedFeatures(flipped_features);
		if constexpr (OT::postprocess_tree) {
			task->PostProcessTree(tree);
		}
	}

	template <class OT>
	std::shared_ptr<SolverResult> Solver<OT>::TestPerformance(const std::shared_ptr<SolverResult>& _result, const ADataView& _test_data) {
		InitializeTest(_test_data, false);
		const SolverTaskResult<OT>* result = static_cast<const SolverTaskResult<OT>*>(_result.get());
		auto solver_result = std::make_shared<SolverTaskResult<OT>>(*result);
		for (size_t i = 0; i < result->NumSolutions(); i++) {
			auto score = InternalTestScore<OT>::ComputeTestPerformance(&data_splitter, task, result->trees[i].get(), flipped_features, test_data);
			solver_result->SetScore(i, score);
		}
		return solver_result;
	}

    template <class OT>
    double Solver<OT>::RashomonTestPerformance(const Tree<OT>* tree, const ADataView& _test_data) {
        InitializeTest(_test_data, false);
        auto test_score = InternalTestScore<OT>::ComputeTestPerformance(&data_splitter, task, tree, flipped_features, test_data);
        return test_score->score;
    }

	template <class OT>
	std::vector<typename OT::LabelType> Solver<OT>::Predict(const std::shared_ptr<Tree<OT>>& tree, const ADataView& _test_data) {
		InitializeTest(_test_data, false);
		std::vector<typename OT::LabelType> labels(test_data.Size());
		typename OT::ContextType context;
		tree->Classify(&data_splitter, task, context, flipped_features, test_data, labels);
		return labels;
	}

    template<class OT>
    std::shared_ptr<Tree<OT>> Solver<OT>::CreateRashomonTreeN(std::shared_ptr<std::vector<std::shared_ptr<typename Solver<OT>::SolutionTracker>>>& solutions, std::vector<size_t>& cumulative_counts, size_t n) {

        auto it = std::lower_bound(cumulative_counts.begin(), cumulative_counts.end(), n);
        size_t solution_index = std::distance(cumulative_counts.begin(), it);
        auto solution = (*solutions)[solution_index];
        if (solution_index != 0)  n -= cumulative_counts[solution_index-1] + 1;
        return solution->CreateTreeWithIndexN(this, n);
    }

    template<class OT>
    void Solver<OT>::PrepareRashomonStatistics(std::shared_ptr<SolverTaskResult<OT>>& result) {
		auto& solutions = result->rashomon_solutions;
		bool root_sol_exists = false;
        for (auto &sol: *solutions) {
            if (sol->GetFeature() != -1) {
                result->root_feature_stats[sol->GetFeature()] += sol->GetNumSolutions();
				if (!sol->IsRecursive()) {
					++result->feature_stats[sol->GetFeature()];
					++result->num_nodes_stats[1];
				} else if (auto _sol = dynamic_cast<RecursiveSolutionTracker<OT>*>(sol.get())) {
					std::transform(result->feature_stats.begin(), result->feature_stats.end(),
						_sol->feature_count.begin(), result->feature_stats.begin(), std::plus<size_t>());
					size_t size = 0;
					for (int i = 0; i < _sol->num_nodes_count.size(); i++) {
						result->num_nodes_stats[i] += _sol->num_nodes_count[i];
						size += _sol->num_nodes_count[i];
					}
				}

                
            } else {
                result->num_nodes_stats[0] += 1;
            }
        }
    }

    template<class OT>
    std::vector<std::shared_ptr<Tree<OT>>> Solver<OT>::CalculateTreesWithRootFeature(SolverTaskResult<OT>* result, int feature) {
        std::vector<std::shared_ptr<Tree<OT>>> trees;
		size_t num_solutions = result->root_feature_stats[feature];
        if (num_solutions == 0) return trees;

        size_t count = 0;
        for (size_t i = 0; i < result->rashomon_solutions->size(); ++i) {
            auto sol = (*result->rashomon_solutions)[i];
            if (sol->GetFeature() == feature) {
                count += sol->GetNumSolutions();
                for (size_t j = 0; j < sol->GetNumSolutions(); ++j) {
                    auto tree = sol->CreateTreeWithIndexN(this, j);
                    trees.push_back(tree);
                }
            }
            if (num_solutions == count) break;
        }
        return trees;
    }

    template<class OT>
    std::vector<std::shared_ptr<Tree<OT>>> Solver<OT>::CalculateTreesWithFeature(SolverTaskResult<OT>* result, int feature) {
        std::vector<std::shared_ptr<Tree<OT>>> trees;
        if (feature >= result->feature_stats.size()) return trees;
        for (size_t i = 0; i < result->rashomon_solutions->size(); ++i) {
            auto sol = (*result->rashomon_solutions)[i];
            if (sol->GetFeature() == -1) continue;
            if (sol->GetFeatureCount(feature) == 0) continue;
            auto sol_trees = sol->CreateQueryTreesWithFeature(this, feature);
            for (auto& tree : sol_trees) {
                trees.push_back(tree);
            }
        }
        return trees;
    }

    template<class OT>
    std::vector<std::shared_ptr<Tree<OT>>> Solver<OT>::CalculateTreesWithoutFeature(SolverTaskResult<OT>* result, int feature) {
        std::vector<std::shared_ptr<Tree<OT>>> trees;
        if (feature >= result->feature_stats.size()) return trees;
        for (size_t i = 0; i < result->rashomon_solutions->size(); ++i) {
            auto sol = (*result->rashomon_solutions)[i];
            if (sol->GetFeature() == -1) {
                auto tree = sol->CreateTreeWithIndexN(this,0);
                trees.push_back(tree);
                continue;
            }
            if (sol->GetFeatureCount(feature) == sol->GetNumSolutions()) continue;
            auto sol_trees = sol->CreateQueryTreesWithoutFeature(this, feature);
            for (auto& tree : sol_trees) {
                trees.push_back(tree);
            }
        }
        return trees;
    }

    template<class OT>
    std::vector<std::shared_ptr<Tree<OT>>> Solver<OT>::CalculateTreesWithNodeBudget(SolverTaskResult<OT>* result, int node_budget) {
        std::vector<std::shared_ptr<Tree<OT>>> trees;
        if (node_budget > result->num_nodes_stats.size()-1 || result->num_nodes_stats[node_budget] == 0) return trees;
        for (size_t i = 0; i < result->rashomon_solutions->size(); ++i) {
            auto sol = (*result->rashomon_solutions)[i];
            if(node_budget == 0 && sol->GetFeature() == -1) {
                trees.push_back(sol->CreateTreeWithIndexN(this, 0));
                return trees;
            }
            if (sol->GetNodeCount(node_budget) == 0) continue;
            auto sol_trees = sol->CreateNodeBudgetQueryTrees(this, node_budget);
            for (auto& tree : sol_trees) {
                trees.push_back(tree);
            }
        }
        return trees;
    }

	template class Solver<CostComplexAccuracy>;
	template class Solver<CostComplexRegression>;
	template class Solver<AverageDepthAccuracy>;
}

