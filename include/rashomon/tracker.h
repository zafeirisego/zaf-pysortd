
#ifndef SORTD_RASHOMON_TRACKER_H
#define SORTD_RASHOMON_TRACKER_H

#include "rashomon/solution_tracker.h"
#include "solver/solver.h"

namespace SORTD {

	/*
	* Abstract base class for the trackers
	*/
	template <class OT>
	struct AbstractTracker {
		using SolType = typename OT::SolType;
		using Context = typename Solver<OT>::Context;
		using SolutionTracker = typename Solver<OT>::SolutionTracker;
		using SolutionTrackerP = std::shared_ptr<SolutionTracker>;

		/*
		* Construct the next (complete) solution and add it to the solution list. 
		* Update the next-solution-value to the solution after the popped solution
		*/
		virtual void Pop() = 0;
		
		/*
		* Return true if this tracker has or can construct another solution within the bound.
		* Return false if this tracker is exhausted.
		*/
		virtual bool HasNext() = 0;
		
		/*
		* Return true if this tracker has a solution with index n.
		* If it is not constructed yet, the tracker attempts to construct it
		*/
		bool HasNSolution(size_t n) {
			if (n < GetSolutions().size()) return true;
			while (n >= GetSolutions().size() && HasNext()) Pop();
			return n < GetSolutions().size();
		}
		
		/*
		* Get the value of the first (optimal) solution
		*/
		SolType GetFirstSolutionValue() {
			if (GetSolutions().empty()) return GetNextSolutionValue();
			return GetSolutionN(0)->obj;
		}

		/*
		* Get the value of the next solution value.
		* If the solution list is already fully constructed, return the value of the solution not yet used
		* Otherwise, return the value of the solution not yet constructed (not yet added to the solution list)
		*/
		virtual SolType GetNextSolutionValue() = 0;
		
		/*
		* Return the solution with index n (assuming it exists, otherwise throw an error)
		* If the solution is not yet constructed, it is constructed first
		*/
		inline const SolutionTrackerP& GetSolutionN(size_t n) {
			if (HasNSolution(n)) {
				return GetSolutions()[n];
			} else {
				throw std::runtime_error("Request non-existing solution n");
			}
		}

		/*
		* Return the last unused solution
		* If the solution list is fully constructed, this is last solution of that list not used
		* If the solutions are incrementally constructed, return the last solution that was constructed (popped)
		*/
		virtual SolutionTrackerP GetLastUnusedSolution() = 0;
		
		/*
		* Get the list of solutions (constructed up to this point)
		*/
		virtual const std::vector<SolutionTrackerP>& GetSolutions() const = 0;

		/*
		* Free the memory used by this tracker
		*/
		virtual void Erase() {}

		/*
		* Return true if the solution list of this tracker is complete
		*/
		virtual bool IsSolutionListComplete() const { return false; }

		/*
		* Create the appropriate tracker for the given subproblem.
		* LeafTracker if the subproblem is a leaf node subproblem,
		* TerminalTracker if the subproblem is a terminal subproblem (depth <= 2),
		* BranchTracker otherwise.
		*/ 
		static std::shared_ptr<AbstractTracker<OT>> CreateTracker(Solver<OT>* solver, Cache<OT>* cache, const ADataView& data, const Context& context, int max_depth, int max_num_nodes, SolType UB);
	};

}

#endif