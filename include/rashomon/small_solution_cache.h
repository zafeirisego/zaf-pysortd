#ifndef SORTD_SMALL_SOLUTION_CACHE_H
#define SORTD_SMALL_SOLUTION_CACHE_H

#include "base.h"
#include "rashomon/solution_tracker.h"

namespace SORTD {

	template <class OT>
	struct SmallSolutionCache {

		SmallSolutionCache() = default;

		SolutionTrackerP<OT> RetrieveOrAdd(SolutionTrackerP<OT> solution) {
			auto it = cache.find(solution);
			if (it == cache.end()) {
				cache.insert(solution);
				return solution;
			}
			return *it;
		}

		void Reset() {
			cache.clear();
		}

		std::unordered_set<SolutionTrackerP<OT>, SolutionTrackerHash<OT>, SolutionTrackerEqual<OT>> cache;

	};

}

#endif