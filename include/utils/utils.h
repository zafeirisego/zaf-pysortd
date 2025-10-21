/**
From Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/

#pragma once
#include "base.h"


namespace SORTD {

	inline int int_log2(unsigned int x) {
		if (x == 0) return INT32_MAX;
		int power = 0;
		while (x >>= 1) { power++; } // compute the log2
		return power;
	}

}