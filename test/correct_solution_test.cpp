/**
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/

#include <stdio.h>
#include <stdlib.h>
#include "base.h"
#include "solver/solver.h"

#include <filesystem>

#ifdef DEBUG
#define debug_assert(x) { assert(x); }
#else
#define debug_assert(x) {}
#endif

#define test_assert(x) {if (!(x)) { printf(#x); debug_assert(x); exit(1); }}
#define test_assert_m(x, m) {if (!(x)) { printf(m); debug_assert(x); exit(1); }}
#define test_failed(m) { printf(m); debug_assert(1==0); exit(1); }

using namespace SORTD;

struct SolverSetup {
	bool use_branch_caching{ false };
	bool use_dataset_caching{ true };
    std::string feature_ordering{ "gini" };
    bool use_rashomon_multiplier = false;
    double rashomon_multiplier = DBL_MAX;
    double cost_complexity  = 0.01;
    bool track_tree_statistics = true;
    size_t max_num_trees = INT64_MAX;
    float time_limit = 3600;
    bool ignore_trivial_extensions = false;

	std::string ToString() const {
		std::ostringstream os;
		os << "Solver setup. branch-cache: " << use_branch_caching << ", dataset-cache: " << use_dataset_caching
            << ", feature-ordering: " << feature_ordering << ", use-rashomon-multiplier: " << use_rashomon_multiplier
            << ", rashomon-multiplier: " << rashomon_multiplier  << ", cost_complexity: " << cost_complexity
            << ", track-tree-statistics: " << track_tree_statistics  << ", max-num-trees: " << max_num_trees
            << ", time: " << time_limit << ", ignore_trivial_extensions: " << ignore_trivial_extensions;
		return os.str();
	}

	void Apply(ParameterHandler& params) const {
		params.SetBooleanParameter("use-branch-caching", use_branch_caching);
		params.SetBooleanParameter("use-dataset-caching", use_dataset_caching);
		params.SetStringParameter("feature-ordering", feature_ordering);
        params.SetBooleanParameter("use-rashomon-multiplier", use_rashomon_multiplier);
        params.SetFloatParameter("rashomon-multiplier", rashomon_multiplier);
        params.SetFloatParameter("cost-complexity", cost_complexity);
        params.SetBooleanParameter("track-tree-statistics", track_tree_statistics);
        params.SetIntegerParameter("max-num-trees", max_num_trees);
        params.SetFloatParameter("time", time_limit);
        params.SetBooleanParameter("ignore-trivial-extensions", ignore_trivial_extensions);
	}
};

struct TestSetup {
	std::string file;
	int max_depth;

	std::string ToString() const {
		std::ostringstream os;
		os << "File: " << file << ", D=" << max_depth;
		return os.str();
	}
};


void RunAccuracyTest(const TestSetup& test, const SolverSetup& solver_setup) {
	ParameterHandler parameters = SORTD::ParameterHandler::DefineParameters();
	parameters.SetIntegerParameter("max-depth", test.max_depth);
	parameters.SetStringParameter("file", test.file);
	solver_setup.Apply(parameters);
	auto rng = std::default_random_engine(0);

	std::shared_ptr<SolverResult> result, result_UB;
	AData data;
	ADataView train_data, test_data;
	FileReader::ReadData<CostComplexAccuracy>(parameters, data, train_data, test_data, &rng);
	auto solver = new SORTD::Solver<CostComplexAccuracy>(parameters, &rng);
	solver->PreprocessData(data, true);
	result = solver->Solve(train_data);		
	delete solver;
	auto score = std::static_pointer_cast<InternalTrainScore<CostComplexAccuracy>>(result->optimal_scores[0]);
}

void EnumerateSolverRashomonSetupOptions(std::vector<SolverSetup>& solver_setups) {
    std::vector<double> cost_complexity{ 0.01};
    std::vector<double> rashomon_multiplier{0.1};

    for (int i = 0; i < cost_complexity.size(); i++) {
        for (int j = 0; j < rashomon_multiplier.size(); j++) {
            SolverSetup s;
            s.cost_complexity  = cost_complexity[i];
            s.rashomon_multiplier = rashomon_multiplier[j];
            solver_setups.push_back(s);
        }
    }

}

int main(int argc, char **argv) {
	std::vector<SolverSetup> solver_setups;
    EnumerateSolverRashomonSetupOptions(solver_setups);

	std::vector<TestSetup> accuracy_tests{
        { "data/accuracy/monk1.csv", 4}
	};


    for (auto& ats : accuracy_tests) {
	    for (auto& ss : solver_setups) {
            ss.use_rashomon_multiplier = true;
            ss.ignore_trivial_extensions = false;
            RunAccuracyTest(ats, ss);
            break;
		}
	}

	
}