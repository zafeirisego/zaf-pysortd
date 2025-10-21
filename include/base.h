/**
From Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/

#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <limits>
#include <deque>
#include <chrono>
#include <time.h>
#include <algorithm>
#include <iterator> 
#include <numeric>
#include <cmath>
#include <cstddef> 
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <random>
#include <cfloat>
#include <cstring>
#include <type_traits>
#include <array>
#include <queue>

#ifdef DEBUG
#define runtime_assert(cond)                                             \
        do {                                                                      \
            if (!(cond)) {                                                        \
                std::cerr << "Assertion failed: " << #cond                        \
                          << "\nFile: " << __FILE__                               \
                          << "\nLine: " << __LINE__ << std::endl;                 \
                std::terminate();                                                 \
            }                                                                     \
        } while (0)
#else
#define runtime_assert(x) {}
#endif

#define SOLUTION_PRECISION 1e-4
#define STOI(T, s) ((std::is_same<T, double>::value) ? static_cast<int64_t>(std::round(s / SOLUTION_PRECISION)) : static_cast<int64_t>(s))
#define SOL_EQUAL(T, a, b) STOI(T, a) == STOI(T, b)