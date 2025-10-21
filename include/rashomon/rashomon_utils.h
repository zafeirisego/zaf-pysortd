#ifndef SORTD_RASHOMON_UTILS_H
#define SORTD_RASHOMON_UTILS_H

#include "rashomon/split_tracker.h"
#include "rashomon/leaf_tracker.h"

namespace SORTD {

    // Orders the split or leaf trackers of a branch tracker to efficiently determine its next best split
    template <class OT>
    struct TrackerCompare {
    public:
        bool operator() (AbstractTracker<OT>* sp1, AbstractTracker<OT>* sp2) {
            runtime_assert(sp1 && sp2);
            int64_t sol1 = STOI(typename OT::SolType, sp1->GetNextSolutionValue());
            int64_t sol2 = STOI(typename OT::SolType, sp2->GetNextSolutionValue());
            if (sol1 == sol2) {
                int f1 = dynamic_cast<LeafTracker<OT>*>(sp1) ? -1 : dynamic_cast<SplitTracker<OT>*>(sp1)->feature;
                int f2 = dynamic_cast<LeafTracker<OT>*>(sp2) ? -1 : dynamic_cast<SplitTracker<OT>*>(sp2)->feature;
                return f1 > f2;
            }
            return sol1 > sol2;
        }
    };

}

#endif