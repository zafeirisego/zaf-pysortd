
#ifndef SORTD_RASHOMON_SPLIT_TRACKER_H
#define SORTD_RASHOMON_SPLIT_TRACKER_H

#include "base.h"
#include <bitset>
#include "solver/solver.h"
#include "rashomon/tracker.h"
#include "rashomon/solution_value_heap.h"
#include "rashomon/solution_tracker.h"

namespace SORTD {

    template <class OT>
    struct BranchTracker;

    /*
	 * Visited keeps track of the left and right index combinations of the child branch trackers that are evaluated before.
     * Each SplitTracker has its own Visited object.
	*/
    struct Visited {

        void Insert(size_t left, size_t right) {
            if (largest_right.size() <= left) {
                largest_right.resize(left + 1, -1);
            }
            size_t delta1 = right - largest_right[left];
            runtime_assert(delta1 == 1);
            largest_right[left] = right;
            if (largest_left.size() <= right) {
                largest_left.resize(right + 1, -1);
            }
            size_t delta2 = left - largest_left[right];
            runtime_assert(delta2 == 1);
            largest_left[right] = left;

            total++;
        }

        bool IsVisited(size_t left, size_t right) const {
            if (largest_right.size() <= left || largest_left.size() <= right) {
                return false;
            }
            return right <= largest_right[left] && left <= largest_left[right];
        }

        // Return the first unvisited right index given a left index
        size_t GetFirstRightNotVisited(size_t left) const {
            size_t r = largest_right.size() <= left ? 0 : largest_right[left] + 1;
            runtime_assert((r <= 0 || IsVisited(left, r - 1)) && !IsVisited(left, r));
            return r;
        }

        // Return the first unvisited left index given a right index
        size_t GetFirstLeftNotVisited(size_t right) const {
            size_t l = largest_left.size() <= right ? 0 : largest_left[right] + 1;
            runtime_assert((l <= 0 || IsVisited(l - 1, right)) && !IsVisited(l, right));
            return l;
        }

        void Clear() {
            largest_right.clear();
            largest_right.shrink_to_fit();
            largest_left.clear();
            largest_left.shrink_to_fit();
        }

        std::size_t total{ 0 };// just for stats
        std::vector<size_t> largest_right;
        std::vector<size_t> largest_left;
    };

    struct AccumulateIndex {
        AccumulateIndex() = delete;
        AccumulateIndex(size_t li, size_t ri, bool el, bool er) : lindex(li), rindex(ri), expand_left(el), expand_right(er) {}
        size_t lindex, rindex;
        bool expand_left, expand_right;
    };

    /*
	* SplitTracker maintains a sorted set of solutions for a split in the subtree.
     * Therefore, SplitTracker is associated with a feature.
	*/
    template <class OT>
    struct SplitTracker : public AbstractTracker<OT> {
        using SolType = typename Solver<OT>::SolType;
        using SolContainer = typename Solver<OT>::SolContainer;
        using Context = typename Solver<OT>::Context;
        using SolutionTracker = typename Solver<OT>::SolutionTracker;
        using SolutionTrackerP = std::shared_ptr<SolutionTracker>;
        using TrackerP = std::shared_ptr<AbstractTracker<OT>>;

        SplitTracker() = default;
        SplitTracker(Solver<OT>* solver, Cache<OT>* cache, ADataView& data, const Context& context, int max_depth, int max_num_nodes, int num_features, SolType UB, int feature);

        // Retrieve the optimal solution of the split from the cache
        SolType FindInitialSolutionValue();

        // Check whether there exists multiple label assignments to the child nodes, which result in the same objective value.
        bool CanSwitchTrivialExtensionLabel(ADataView& left_data, ADataView& right_data);

        void Initialize();

        inline SolType GetNextSolutionValue() {
            return next_solution_value;
        }

        SolutionTrackerP GetLastUnusedSolution();

        inline bool HasNext() { return next_solution_value <= UB + SOLUTION_PRECISION; }

        const std::vector<SolutionTrackerP>& GetSolutions() const { return solutions; }

        // Generate new solutions by requesting those with indices (left+1,right) and (left,right+1).
        bool CreateCandidateSolution(std::shared_ptr<RecursiveSolutionTracker<OT>> solution, size_t left, size_t right);

        // Merge heap buffer to the heap.
        void MergeHeapBuffer();

        // Combine the left and right solutions computed by the depth-two solver.
        void CombineSolutions();

        // Populate the same-valued solutions by recursively checking
        // whether new solutions with indices (left+1,right) and (left,right+1) have the same solution value.
        void AccumulateSameSolutions(std::shared_ptr<RecursiveSolutionTracker<OT>> solution, std::vector<AccumulateIndex>& popped_indices);

        // Pop the solutions with the same solution value from the heap.
        void PopSameHeapSolutions(std::shared_ptr<RecursiveSolutionTracker<OT>> solution, std::vector<AccumulateIndex>& popped_indices);

        // Get the minimum valued solution from the heap (candidates queue) and add it to the sorted solution list.
        // While doing so, accumulate split tracker's solutions with the same value under the solution tracker.
        void Pop();

        void Erase();

        bool IsSolutionListComplete() const { return solution_list_complete; }

        // Create an abstract tracker (i.e., a branch, leaf or terminal tracker) for the split tracker.
        // The abstract tracker later becomes the left or the right tracker of the split tracker.
        std::shared_ptr<AbstractTracker<OT>> CreateChildTracker(Solver<OT>* solver, Cache<OT>* cache, const ADataView& _data, const Context& context, int _max_depth, int _max_num_nodes, SolType _UB);

        bool IsTrivialExtension(SolutionTrackerP left_sol, SolutionTrackerP right_sol) {
            if (!solver->GetSolverParameters().ignore_trivial_extentions) return false;
            return left_sol->IsLeaf() && right_sol->IsLeaf() && std::abs(left_sol->GetLabel() - right_sol->GetLabel()) < SOLUTION_PRECISION;
        }

        bool SwitchToAlternativeSolution(SolutionTrackerP left_sol, SolutionTrackerP right_sol);

        int feature;
        ADataView data;
        Context context;
        Solver<OT>* solver;
        Cache<OT>* cache;
        int num_features, max_depth, max_num_nodes;
        SolType UB;
        std::shared_ptr<AbstractTracker<OT>> left_tracker, right_tracker;

        std::vector<SolutionTrackerP> solutions;
        SolType next_solution_value = OT::worst;

        bool solution_list_complete = false;
        bool initialized = false;
        size_t solution_index = 0;

        std::priority_queue<HeapEntry, std::vector<HeapEntry>, CompareHeapEntry> heap;
        std::vector<HeapEntry> heap_buffer;
        double max_heap_value = 0.0;
        double max_heap_buffer_value = 0.0;
        Visited visited_heap_indices;
    };

}

#endif