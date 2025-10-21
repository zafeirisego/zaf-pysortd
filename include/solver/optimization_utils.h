/**
From Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
 */

#pragma once
#include "base.h"
#include "model/data.h"
#include "model/branch.h"
#include "model/node.h"
#include "model/container.h"


#define DBL_DIFF 1e-6

namespace SORTD {

	template <class OT>
	using SolContainer =  Node<OT>;

	/*
	* Initialize an empty solution. 
	* Default is a worst-case solution
	* if lb=true, it is a best-case solution
	*/
	template <class OT>
	SolContainer<OT> InitializeSol(bool lb = false) {
		if (lb) return Node<OT>(OT::best);
		return Node<OT>();
	}

	/*
	* Initialize an empty lower bound.
	*/
	template <class OT>
	SolContainer<OT> InitializeLB() {
		return InitializeSol<OT>(true);
	}


    /*
	* Return a copy of the solution
	*/
    template <class OT>
    std::shared_ptr<Container<OT>> CopySol(const std::shared_ptr<Container<OT>>& sol) {
        return std::make_shared<Container<OT>>(*(sol.get()));
    }

	/*
	* Return a copy of the solution
	*/
	template <class OT>
	SolContainer<OT> CopySol
	(const SolContainer<OT>& sol) {
		return sol;
	}


	/*
	* Check if the solution is empty
	* single solution: empty feature, empty label
	* solution set: nullptr or zero solutions
	*/
	template <class OT>
	bool CheckEmptySol(const SolContainer<OT>& sol) {
		return sol.feature == INT32_MAX && sol.label == OT::worst_label;
	}

    template <class OT>
    bool CheckEmptySol(const std::shared_ptr<Container<OT>>& sol) {
        return sol.get() == nullptr || sol->Size() == 0;
    }

	/*
	* Add a solution to the solution set (if not dominated) or replace the solution if better
	*/
	template <class OT>
	inline void AddSol(SolContainer<OT>& container, const Node<OT>& sol) {
		if (sol.solution < container.solution) container = sol;
	}

    template <class OT>
    inline bool AddSubOptimalSols(std::shared_ptr<Container<OT>>& container, const Node<OT>& sol, const SolContainer<OT>& UB) {
        if (sol.solution <= UB.solution) {
            container->Push(sol);
            return true;
        }
        return false;
    }

    template <class OT>
    inline void AddSubOptimalSols(std::shared_ptr<Container<OT>>& container, const std::shared_ptr<Container<OT>>& sols, SolContainer<OT>& UB) {
        container->Push(sols);
        return;
    }

    template <class OT>
    inline bool AddSubOptimalSols(SolContainer<OT>& container, const Node<OT>& sol, const SolContainer<OT>& UB) {
        if (sol.solution < container.solution) container = sol;
            return true;
    }

	/*
	* Add a solution to the solution set (if not dominated) or replace the solution if better
	* If the current node is the root node, use the root-node comparator to determine dominance
	*/
	template <class OT>
	inline void AddSol(OT* task, const int depth, SolContainer<OT>& container, const Node<OT>& sol) {
		if (sol.solution < container.solution) container = sol;
	}

	/*
	* Add set of solutions to the solution container
	*/
	template <class OT>
	inline void AddSols(SolContainer<OT>& container, const SolContainer<OT>& sols) {
		if (sols.solution < container.solution) container = sols;
	}

	/*
	* Add set of solutions to the solution container
	* If the current node is the root node, use the root-node comparator to determine dominance
	*/
	template <class OT>
	inline void AddSols(OT* task, const int depth, SolContainer<OT>& container, const SolContainer<OT>& sols) {
		if (sols.solution < container.solution) container = sols;
	}

	/*
	* Add set of solutions to the solution container
	* Use the inverted dominance operator to determine dominance
	*/
	template <class OT>
	inline void AddSolsInv(SolContainer<OT>& container, const SolContainer<OT>& sols) {
		if (sols.solution > container.solution) container = sols;
	}

	/*
	* Combine two solutions and store the combined solution in out
	*/
	template <class OT>
	inline void CombineSols(int feature, const Node<OT>& left, const Node<OT>& right, const typename OT::SolType& branching_costs, Node<OT>& out) {
		if constexpr (OT::has_branching_costs) {
			out = Node<OT>(feature, OT::Add(branching_costs, OT::Add(left.solution, right.solution)), left.NumNodes(), right.NumNodes());
		} else {
			out = Node<OT>(feature, OT::Add(left.solution, right.solution), left.NumNodes(), right.NumNodes());
		}
//        out.solution = std::ceil(out.solution / 1e-4) * 1e-4;
	}

    template <class OT>
    inline void CombineSols(typename OT::SolType left, typename OT::SolType right, typename OT::SolType branching_costs, typename OT::SolType& out) {
        if constexpr (OT::has_branching_costs) {
            out = left + right + branching_costs;
        } else {
            out =  left + right;
        }
//        out = std::round(out / 1e-4) * 1e-4;
    }

	/*
	* Return true iff the left solution value dominates the rigt solution value
	*/
	template <class OT>
	bool LeftDominatesRight(const typename OT::SolType& left, const typename OT::SolType& right) {
		if constexpr (std::is_same<typename OT::SolType, double>::value) {
				return left * (1 + DBL_DIFF) <= right || std::abs(left - right) <= DBL_DIFF * left;
			}
			return left <= right;
	}

	/*
	* Return true iff the left solution value strictly dominates the rigt solution value
	*/
	template <class OT>
	bool LeftStrictDominatesRight(const typename OT::SolType& left, const typename OT::SolType& right) {
		if constexpr (std::is_same<typename OT::SolType, double>::value) {
				return left * (1 + DBL_DIFF)  < right;
			}
			return left < right;
	}

	/*
	* Return true iff the left solution value + the rashomon bound delta strictly dominates the rigt solution value
	*/
	template <class OT>
	bool UBStrictDominatesRight(const typename OT::SolType& left, const typename OT::SolType& right, const typename OT::SolType& rash_delta) {
		if constexpr (std::is_same<typename OT::SolType, double>::value) {
				return (left + rash_delta) * (1 + DBL_DIFF) < right;
			}
			return left + rash_delta < right;
	}

	/*
	* Return true iff at least one left solution dominates the right solution
	*/
	template <class OT>
	inline bool LeftDominatesRightSol(const SolContainer<OT>& left, const Node<OT>& right) {
		return LeftDominatesRight<OT>(left.solution, right.solution);
	}

	/*
	* Return true iff at least one left solution strictly dominates the right solution
	*/
	template <class OT>
	inline bool LeftStrictDominatesRightSol(const SolContainer<OT>& left, const Node<OT>& right) {
		return LeftStrictDominatesRight<OT>(left.solution, right.solution);
	}

	/*
	* Return true iff for all left solutions there is at least one right solution that inverse dominates it
	*/
	template <class OT>
	inline bool LeftDominatesRight(const SolContainer<OT>& left, const SolContainer<OT>& right) {
		return LeftDominatesRight<OT>(left.solution, right.solution);
	}

	/*
	* Return true iff for all left solutions there is at least one right solution that inverse strictly dominates it
	*/
	template <class OT>
	bool LeftStrictDominatesRight(const SolContainer<OT>& left, const SolContainer<OT>& right) {
		return LeftStrictDominatesRight<OT>(left.solution, right.solution);
	}

	/*
	* Return true iff for all left solutions + rashomon bound delta there is at least one right solution that inverse strictly dominates it
	*/
	template <class OT>
	bool UBStrictDominatesRight(const SolContainer<OT>& left, const SolContainer<OT>& right, const typename OT::SolType& rash_delta) {
		return UBStrictDominatesRight<OT>(left.solution, right.solution, rash_delta);
	}

	/*
	* Return true if and only if the left and right solution (set) are equal
	*/
	template <class OT>
	bool SolutionsEqual(const SolContainer<OT>& left, const SolContainer<OT>& right) {
		if constexpr (std::is_same<typename OT::SolType, double>::value) {
				return std::abs(left.solution - right.solution) <= DBL_DIFF * left.solution;
			}
			return left.solution == right.solution;
	}

	/*
	* return true iff if the left and right solution and the branching costs equal the solution value of tree->parent
	* If so, store the solution for left and right nodes in tree
	*/
	template<class OT>
	bool CheckReconstructSolution(const Node<OT>& left, const Node<OT>& right, const typename OT::SolType& branching_costs, TreeNode<OT>* tree) {
		auto combi_sol = OT::Add(left.solution, right.solution);
		if constexpr (OT::has_branching_costs) {
			combi_sol = OT::Add(combi_sol, branching_costs);
		}
		if constexpr (std::is_same<typename OT::SolType, double>::value) {
			if (std::abs(tree->parent.solution - combi_sol) > DBL_DIFF * combi_sol) return false;
		} else {
			if (!(tree->parent.solution == combi_sol)) return false;
		}
		tree->left_child = left;
		tree->right_child = right;
		return true;
	}

	template<class OT>
	void AddValue(SolContainer<OT>& solutions, const typename OT::SolType& value) {
		OT::Add(solutions.solution, value, solutions.solution);
	}

}