/**
From Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/

#pragma once
#include "tasks/optimization_task.h"

namespace SORTD {

	struct AVDAccExtraData {
	        int unique_feature_vector_id{ 0 };
	        static AVDAccExtraData ReadData(std::istringstream& iss, int num_labels) {return {};}	
	};

        class AverageDepthAccuracy : public Classification {
        public:
		using ET = AVDAccExtraData;
                using SolType = double;                    // The data type of the solution
                using SolD2Type = double;                  // The data type of the solution in the terminal solver
		using BranchSolD2Type = double; 
                using TestSolType = int;                // The data type of the solution that is used for evaluation

                static const bool custom_leaf = false;          // Set to true if you want to implement a custom leaf function (for optimization)
                static constexpr int worst = INT32_MAX;         // An UB for the worst solution value possible
                static constexpr double best = 0;                          // A LB for the best solution value possible
                static constexpr int minimum_difference = 1;// The minimum difference between two solutions
		static constexpr bool leaf_penalty = true;
		static const bool has_branching_costs = false; 
		static const bool element_branching_costs = false;
	        static const bool preprocess_data = true;
	        static const bool preprocess_train_data = true; 
	        static const bool use_terminal = false;
	        static const bool terminal_filter = false;
	        static const bool custom_lower_bound = true;	

                AverageDepthAccuracy(const ParameterHandler& parameters) : Classification(parameters) {}

		inline void UpdateParameters(const ParameterHandler& parameters) {
			cost_complexity_parameter = std::max(0.0, parameters.GetFloatParameter("cost-complexity"));
	                lower_bound_cache.resize(parameters.GetIntegerParameter("max-depth") + 1);		
		}

                // Compute the leaf costs for the data in the context when assigning label
                double GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const;

                // Compute the test leaf costs for the data in the context when assigning label
                inline int GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const; 
                // Compute the leaf costs for an instance given a assigned label
                inline void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, double& costs, int multiplier) const { costs = multiplier * ((org_label == label) ? 0 : 1); }

		double GetBranchingCosts(const BranchContext& context, int feature) const {
			if constexpr (leaf_penalty) {
				int multiplier = 1;
				if (context.GetBranch().Depth() == 0) multiplier = 2;
				return cost_complexity_parameter * train_summary.size * multiplier;
			} else  {
				return cost_complexity_parameter * train_summary.size;
			}
		}			

                // Compute the solution value from a terminal solution value
                void ComputeD2Costs(const int& d2costs, int count, double& costs, std::optional<BranchContext> context = std::nullopt) const { 
			if constexpr (leaf_penalty) {
				if (context && context->GetBranch().Depth() == 0) {
					costs = d2costs + cost_complexity_parameter * count;
				} else {
					costs = d2costs;
				}
			} else {
				costs = d2costs;
			}
		}

                // Return true if the terminal solution value is zero
                inline bool IsD2ZeroCost(const double d2costs) const { return std::abs(d2costs) <= 1e-6; }

		double ComputeD2BranchingCosts(const double& d2costs, int count) const { return d2costs; }

                // Get a bound on the worst contribution to the objective of a single instance with label
                inline int GetWorstPerLabel(int label) const { return 1; }

		Node<CostComplexAccuracy> CostComplexAccuracy::ComputeLowerBound(const ADataView& data, const Branch& branch, int max_depth, int num_nodes) ;

		void PreprocessData(AData& data, bool train);
		void PreprocessTrainData(ADataView& train_data);

                // Compute the train score from the training solution value
                inline double ComputeTrainScore(int test_value) const { return ((double)(train_summary.size - test_value)) / ((double)train_summary.size); }

                // Compute the test score on the training data from the test solution value
                inline double ComputeTrainTestScore(int test_value) const { return ((double)(train_summary.size - test_value)) / ((double)train_summary.size); }

                // Compute the test score on the test data from the test solution value
                inline double ComputeTestTestScore(int test_value) const { return ((double)(test_summary.size - test_value)) / ((double)test_summary.size); }

		static TuneRunConfiguration GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& train_data, int phase);
        };

}
