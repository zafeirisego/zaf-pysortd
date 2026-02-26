/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/
#include "utils/parameter_handler.h"
#include "utils/stopwatch.h"
#include "solver/solver.h"
#include "tasks/tasks.h"

using namespace std;

int main(int argc, char* argv[]) {
	SORTD::ParameterHandler parameters = SORTD::ParameterHandler::DefineParameters();

	if (argc > 1) {
		parameters.ParseCommandLineArguments(argc, argv);
	} else {
		cout << "No parameters specified." << endl << endl;
		parameters.PrintHelpSummary();
		exit(1);
	}

	if (parameters.GetBooleanParameter("verbose")) { parameters.PrintParametersDifferentFromDefault(); }
	std::default_random_engine rng;
	if (parameters.GetIntegerParameter("random-seed") == -1) { 
		rng = std::default_random_engine(int(time(0)));
	} else { 
		rng = std::default_random_engine(int(parameters.GetIntegerParameter("random-seed")));
	}


	// parameters.CheckParameters();
	bool verbose = parameters.GetBooleanParameter("verbose");
	
	SORTD::AData data(int(parameters.GetIntegerParameter("max-num-features")));
	SORTD::ADataView train_data, test_data;
	
	// Initialize the solver and the data based on the optimization task at hand
	SORTD::Stopwatch stopwatch;
	stopwatch.Initialise(0);
	
	SORTD::AbstractSolver* solver;
	std::string task = parameters.GetStringParameter("task");
	if (verbose) { std::cout << "Reading data...\n"; }
	if (task == "cost-complex-accuracy") {
		solver =  new SORTD::Solver<SORTD::CostComplexAccuracy>(parameters, &rng);
		SORTD::FileReader::ReadData<SORTD::CostComplexAccuracy>(parameters, data, train_data, test_data, &rng);
	} else if (task == "cost-complex-regression") {
		solver =  new SORTD::Solver<SORTD::CostComplexRegression>(parameters, &rng);
		SORTD::FileReader::ReadData<SORTD::CostComplexRegression>(parameters, data, train_data, test_data, &rng);
	} else if (task == "average-depth-accuracy") {
	        solver =  new SORTD::Solver<SORTD::AverageDepthAccuracy>(parameters, &rng);
		SORTD::FileReader::ReadData<SORTD::AverageDepthAccuracy>(parameters, data, train_data, test_data, &rng);
	} else {
		std::cout << "Encountered unknown optimization task: " << task << std::endl;
		exit(1);
	}
	clock_t clock_before_solve = clock();
	std::shared_ptr<SORTD::SolverResult> result;
	// Preprocess the data
	solver->PreprocessData(data, true);
	// Solve with hyper-tuning or directly
	if (verbose) { std::cout << "Optimal tree computation started!\n"; }
    result = solver->Solve(train_data);
	solver->InitializeTest(test_data);
	auto test_result = solver->TestPerformance(result, test_data);
	// report results
	std::cout << "TIME: " << stopwatch.TimeElapsedInSeconds() << " seconds\n";
	std::cout << "CLOCKS FOR SOLVE: " << ((double)clock() - (double)clock_before_solve) / CLOCKS_PER_SEC << "\n";


	if (verbose) {
		if (result->NumSolutions() > 0) {
			if (!result->IsProvenOptimal()) {
				std::cout << std::endl << "Warning: No proof of optimality. Results are best solution found before the time-out." << std::endl << std::endl;
			}

			std::cout << "Solutions: " << result->NumSolutions() << " \tD\tN\t\tTrain \t\tTest\t\tAvg. Path length" << std::endl;
			for (int i = 0; i < result->NumSolutions(); i++) {
				auto train_score = result->optimal_scores[i];
				auto test_score = test_result->optimal_scores[i];
				std::cout << "Solution " << i << ": \t" << std::setw(2) << result->depths[i] << " \t" << result->num_nodes[i] << " \t\t";
				std::cout << std::setprecision (std::numeric_limits<double>::digits10 + 1) << train_score->score << " \t";
				std::cout << std::setprecision (std::numeric_limits<double>::digits10 + 1) << test_score->score << " \t"; 
				std::cout << test_score->average_path_length << std::endl;

				std::cout << "Tree " << i << ": " << result->tree_strings[i] << std::endl;
				
			}
		} else {
			std::cout << std::endl << "No tree found" << std::endl;
		}
	}



	delete solver;

	cout << endl << "SORTD closed successfully!" << endl;
}

