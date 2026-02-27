/**
From Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
 */

#include "tasks/average_depth_accuracy.h"

namespace SORTD {

        double AverageDepthAccuracy::GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const { // Replace by custom function later
                double error = 0;
                for (int k = 0; k < data.NumLabels(); k++) {
                        if (k == label) continue;
                        error += data.NumInstancesForLabel(k);
                }
                return error;
        }

}
