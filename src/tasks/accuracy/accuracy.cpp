/**
From Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
 */

#include "tasks/accuracy/accuracy.h"

namespace SORTD {

	int Accuracy::GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const { // Replace by custom function later
		int error = 0;
		for (int k = 0; k < data.NumLabels(); k++) {
			if (k == label) continue;
			error += data.NumInstancesForLabel(k);
		}
		return error;
	}

}