/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/

#pragma once

#include "model/data.h"

namespace SORTD {

	struct DifferenceMetrics {
		DifferenceMetrics(int num_labels) :num_removals(num_labels, 0), total_difference(0) {}
		std::vector<int> num_removals;
		int total_difference;
		int GetNumRemovals() const;
	};

	class BinaryDataDifferenceComputer {
	public:
		// Compute the difference metrics for two datasets
		static DifferenceMetrics ComputeDifferenceMetrics(const ADataView& data_old, const ADataView& data_new);
		
		// Compute the difference between two datasets and store the difference in the data_to_add and data_to_remove dataviews
		static void ComputeDifference(const ADataView& data_old, const ADataView& data_new, ADataView& data_to_add, ADataView& data_to_remove);
	};
}