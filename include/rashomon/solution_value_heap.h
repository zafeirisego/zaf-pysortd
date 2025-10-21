
#ifndef SORTD_SOLUTION_VALUE_HEAP_H
#define SORTD_SOLUTION_VALUE_HEAP_H

#include "solver/optimization_utils.h"


namespace SORTD {

    // Used to order the solutions in the split tracker's heap (i.e., candidate solutions).
    struct HeapEntry {
        double solution;
        size_t lindex;
        size_t rindex;

        HeapEntry(const double& s, size_t i1, size_t i2) :
        solution(s), lindex(i1), rindex(i2)
        {}

        // Copy Constructor
        HeapEntry(const HeapEntry& other)
            : solution(other.solution), lindex(other.lindex), rindex(other.rindex) {}

        // Copy Assignment
        HeapEntry& operator=(const HeapEntry& other) {
            if (this != &other) {
                solution = other.solution;
                lindex = other.lindex;
                rindex = other.rindex;
            }
            return *this;
        }

        // Move Constructor
        HeapEntry(HeapEntry&& other) noexcept
                : solution(std::move(other.solution)),
                  lindex(other.lindex),
                  rindex(other.rindex) {}

        // Move Assignment
        HeapEntry& operator=(HeapEntry&& other) noexcept {
            if (this != &other) {
                solution = std::move(other.solution);
                lindex = other.lindex;
                rindex = other.rindex;
            }
            return *this;
        }

    };


    struct CompareHeapEntry {
        bool operator()(const HeapEntry a, const HeapEntry b) {
            if (SOL_EQUAL(double, a.solution, b.solution)) {
                if (a.lindex == b.lindex) {
                    return a.rindex > b.rindex;
                }
                return a.lindex > b.lindex;
            }
            return a.solution > b.solution;
        }
    };
}

#endif //SORTD_SOLUTION_VALUE_HEAP_H
