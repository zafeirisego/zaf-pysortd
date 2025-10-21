/**
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/

#pragma once
#include "tasks/optimization_task.h"

namespace SORTD {

	struct CCAccExtraData {
		int unique_feature_vector_id{ 0 };

		static CCAccExtraData ReadData(std::istringstream& iss, int num_labels) { return {}; }
	};

	class CostComplexAccuracy : public Classification {
	public:
		using ET = CCAccExtraData;			// The extra data stores the unique feature vector id
		using SolType = double;
		using SolD2Type = double;
		using BranchSolD2Type = double;		// The data type of the branching costs in the terminal solver
		using TestSolType = int;

		static const bool custom_leaf = false;
		static const bool element_additive = true;
		static const bool has_branching_costs = true;
		static const bool element_branching_costs = false;
		static const bool constant_branching_costs = true;
		static const bool terminal_filter = true;
		static const bool custom_lower_bound = true;		// A custom lower bound is provided (equiv-points)
		static const bool combine_custom_lb_purification = true; // the equiv. points bound can be combined with the purification bound
		static const bool preprocess_data = true;			// The data is preprocessed (set ids based on features sorting)
		static const bool preprocess_train_test_data = true;// The training data is preprocessed (sort on features)
		static constexpr int worst = INT32_MAX;
		static constexpr double best = 0;
		static constexpr bool leaf_penalty = true;          // whether to penalize leaves instead of branching nodes (adds a constant to the objective)

		CostComplexAccuracy(const ParameterHandler& parameters) 
			: Classification(parameters), 
			cost_complexity_parameter(parameters.GetFloatParameter("cost-complexity")),
			lower_bound_cache(parameters.GetIntegerParameter("max-depth") + 1) {}

		inline void UpdateParameters(const ParameterHandler& parameters) {
			cost_complexity_parameter = std::max(0.0, parameters.GetFloatParameter("cost-complexity"));
			lower_bound_cache.resize(parameters.GetIntegerParameter("max-depth") + 1);
		}

		double GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const;
		int GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const;

		double GetBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const { 
			if constexpr (leaf_penalty) {
				int multiplier = 1;
				if (context.GetBranch().Depth() == 0) multiplier = 2;
				return cost_complexity_parameter * train_summary.size * multiplier;
			} else {
				return cost_complexity_parameter * train_summary.size;
			}
		}
		int GetTestBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const {
			return 0;
		}
		double GetBranchingCosts(const BranchContext& context, int feature) const { 
			if constexpr (leaf_penalty) {
				int multiplier = 1;
				if (context.GetBranch().Depth() == 0) multiplier = 2;
				return cost_complexity_parameter * train_summary.size * multiplier;
			} else {
				return cost_complexity_parameter * train_summary.size;
			}
		}
		double ComputeD2BranchingCosts(const double& d2costs, int count) const { return d2costs; }

		inline void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, double& costs, int multiplier) const { costs = multiplier * ((org_label == label) ? 0 : 1); }
		
		void ComputeD2Costs(const double& d2costs, int count, double& costs, std::optional<BranchContext> context = std::nullopt) const { 
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
		inline bool IsD2ZeroCost(const double d2costs) const { return std::abs(d2costs) <= 1e-6; }
		
		inline double GetWorstPerLabel(int label) const { return 1; }

		/*
		* Node only used as a container for the SolType, the label has no significance.
		*/
		Node<CostComplexAccuracy> ComputeLowerBound(const ADataView& data, const Branch& branch, int max_depth, int num_nodes);
		
		void PreprocessData(AData& data, bool train);
		void PreprocessTrainData(ADataView& train_data);
		void PreprocessTestData(ADataView& test_data) {}

		inline double ComputeTrainTestScore(int test_value) const { return ((double)(train_summary.size - test_value)) / ((double)train_summary.size); }
		inline double ComputeTestTestScore(int test_value) const { return ((double)(test_summary.size - test_value)) / ((double)test_summary.size); }

		static TuneRunConfiguration GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& data, int phase);

	private:
		double cost_complexity_parameter{ 0.01 };

		// Cache of previously computed LBs
		std::vector<std::unordered_map<const Branch, Node<CostComplexAccuracy>, BranchHashFunction, BranchEquality>> lower_bound_cache;
	};

}