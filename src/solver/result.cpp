/**
From Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
 */


#include "solver/result.h"
#include "tasks/tasks.h"

namespace SORTD {

	int SolverResult::GetBestDepth() const {
		return depths[best_index];
	}

	int SolverResult::GetBestNodeCount() const {
		return num_nodes[best_index];
	}

}