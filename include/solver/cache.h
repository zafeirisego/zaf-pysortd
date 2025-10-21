/**
 Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
 */

#pragma once
#include "base.h"
#include "utils/parameter_handler.h"
#include "solver/dataset_cache.h"
#include "solver/branch_cache.h"
#include "rashomon/small_solution_cache.h"
#include "solver/tree.h"
#include "solver/optimization_utils.h"
#include "tasks/tasks.h"


namespace SORTD {

	template <class OT>
	class Cache {
	public:
		using SolType = typename OT::SolType;
		using SolContainer = Node<OT>;

		Cache() = delete;
		Cache(const ParameterHandler& parameters, int max_depth, int datasize) : use_lower_bound_caching(true), use_optimal_caching(true),
			use_branch_caching(parameters.GetBooleanParameter("use-branch-caching")),
			use_dataset_caching(parameters.GetBooleanParameter("use-dataset-caching")),
			branch_cache(max_depth + 1), dataset_cache(datasize) {
			empty_sol = InitializeSol<OT>();
			empty_lb = InitializeSol<OT>(true);
		}

		bool IsOptimalAssignmentCached(ADataView& data, const Branch& branch, int depth, int num_nodes);
		void StoreOptimalBranchAssignment(ADataView& data, const Branch& branch, SolContainer& opt_sols, int depth, int num_nodes);
		SolContainer RetrieveOptimalAssignment(ADataView& data, const Branch& branch, int depth, int num_nodes);

		BranchTrackerCacheEntry<OT> RetrieveBranchTracker(ADataView& data, const Branch& branch, int depth, int num_nodes);
		void UpdateBranchTracker(ADataView& data, const Branch& branch, int depth, int num_nodes, BranchTrackerCacheEntry<OT>& branch_cache_entry);
		
		void TransferAssignmentsForEquivalentBranches(const ADataView&, const Branch& branch_source, const ADataView&, const Branch& branch_destination);//this updates branch_destination with all solutions from branch_source. Should only be done if the branches are equivalent.

		void UpdateLowerBound(ADataView&, const Branch& branch, SolContainer& lower_bound, int depth, int num_nodes);
		SolContainer RetrieveLowerBound(ADataView&, const Branch& branch, int depth, int num_nodes);

		void DisableLowerBoundCaching()	{ use_lower_bound_caching = false; }
		void DisableOptimalCaching()	{ use_optimal_caching = false; }
		void DisableBrachCaching()		{ use_branch_caching = false; }
		void DisableDatasetCaching()	{ use_dataset_caching = false; }
		bool UseCache() const { return use_branch_caching || use_dataset_caching; }

		inline SmallSolutionCache<OT>& GetSmallSolutionCache() { return small_solution_cache; }
	
	private:
		bool use_lower_bound_caching, use_optimal_caching,
			use_branch_caching, use_dataset_caching;
		BranchCache<OT> branch_cache;
		DatasetCache<OT> dataset_cache;
		SmallSolutionCache<OT> small_solution_cache;
		SolContainer empty_sol;
		SolContainer empty_lb;
	};


}