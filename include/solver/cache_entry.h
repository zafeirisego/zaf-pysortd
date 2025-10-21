/**
 Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
 */

#pragma once
#include "base.h"
#include "solver/optimization_utils.h"
#include "tasks/tasks.h"


namespace SORTD {

    template <class OT>
    struct AbstractTracker;

	template <class OT>
	struct SplitTracker;

	template <class OT>
	struct TrackerCompare;

    template <class OT, class SVT = typename OT::SolType>
    struct AbstractSolutionTracker;

    template <class OT>
    struct BranchTrackerCacheEntry {
		using SolType = typename OT::SolType;
		using SolutionList = std::shared_ptr<std::vector<std::shared_ptr<AbstractSolutionTracker<OT>>>>;
		using SplitTrackerList = std::shared_ptr<std::vector<SplitTracker<OT>>>;
		using TrackerPriorityQueue = std::shared_ptr<std::priority_queue<AbstractTracker<OT>*, std::vector<AbstractTracker<OT>*>, TrackerCompare<OT>>>;

		BranchTrackerCacheEntry() {};

		BranchTrackerCacheEntry(SolutionList solutions, SolType UB) :
			solutions(solutions), split_trackers(nullptr), best_next_split(nullptr), UB(UB) {
		}

		BranchTrackerCacheEntry(SolutionList solutions, SplitTrackerList split_trackers, TrackerPriorityQueue best_next_split, SolType UB) :
                         solutions(solutions), split_trackers(split_trackers), best_next_split(best_next_split), UB(UB) {}

		inline bool IsEmpty() const { return solutions == nullptr; }

        SolutionList solutions;
		TrackerPriorityQueue best_next_split;
		SplitTrackerList split_trackers;
        SolType UB;
    };

	template <class OT>
	struct CacheEntry {

		using SolType = typename OT::SolType;
		using SolContainer = Node<OT>;

		CacheEntry(int depth, int num_nodes) :
			depth(depth),
			num_nodes(num_nodes) {
			runtime_assert(depth <= num_nodes);
			lower_bound = InitializeSol<OT>(true);

		}

		CacheEntry(int depth, int num_nodes, const SolContainer& solutions) :
			optimal_solutions(solutions),
			lower_bound(solutions),
			depth(depth),
			num_nodes(num_nodes) {
			runtime_assert(depth <= num_nodes);
			runtime_assert(!CheckEmptySol<OT>(solutions));
		}

		SolContainer GetOptimalSolution() const {
			runtime_assert(IsOptimal());
			return CopySol<OT>(optimal_solutions);
		}

        BranchTrackerCacheEntry<OT> GetBranchTrackerCacheEntry() const {
            return branch_tracker_cache_entry;
        }

		inline const SolContainer& GetLowerBound() const { return lower_bound; }

		void SetOptimalSolutions(const SolContainer& optimal_solutions) {
			runtime_assert(!IsOptimal());
			runtime_assert(!CheckEmptySol<OT>(optimal_solutions));
			this->optimal_solutions = optimal_solutions;
			if (!CheckEmptySol<OT>(this->optimal_solutions)) {
				lower_bound = optimal_solutions;
			}
		}

        void SetBranchTrackerCacheEntry(BranchTrackerCacheEntry<OT>& btce) {
            this->branch_tracker_cache_entry = btce;
        }

		void UpdateLowerBound(const SolContainer& lower_bound) {
			runtime_assert(!IsOptimal());
			AddSolsInv<OT>(this->lower_bound, lower_bound);
		}

		inline bool IsOptimal() const { return !CheckEmptySol<OT>(optimal_solutions); }

		inline int GetNodeBudget() const { return num_nodes; }

		inline int GetDepthBudget() const { return depth; }

	private:
		SolContainer optimal_solutions;
		SolContainer lower_bound;
		BranchTrackerCacheEntry<OT> branch_tracker_cache_entry;
		int depth;
		int num_nodes;
	};

	template <class OT>
	struct CacheEntryVector {
		using SolType = typename OT::SolType;
		using SolContainer = Node<OT>;

		CacheEntryVector() = default;
		CacheEntryVector(int size, const CacheEntry<OT>& _default) : entries(size, _default) {}

		void push_back(const CacheEntry<OT>& entry) { entries.push_back(entry); }
		inline CacheEntry<OT>& operator[](size_t idx) { return entries[idx]; }
		inline CacheEntry<OT>& front() { return entries[0]; }

		bool exhausted{ false };
		std::vector<CacheEntry<OT>> entries;
	};
}