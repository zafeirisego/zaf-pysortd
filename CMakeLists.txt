# From Jacobus G.M. van der Linden “STreeD”
# https://github.com/AlgTUDelft/pystreed 

# CMakeList.txt : CMake project for SORTD, include source and define
# project specific logic here.

cmake_minimum_required (VERSION 3.9)

project ("SORTD")
include(CTest)
set (CMAKE_CXX_STANDARD 17)

# Set release build.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
	message(STATUS "Build type not specified: Use Release by default")
endif ()
if (CMAKE_BUILD_TYPE STREQUAL Release)
	message(STATUS "Compiling in release mode")
else()
	message(STATUS "Compiling in debug mode")
	add_compile_definitions(DEBUG=1)
endif()
if (MSVC)
	add_compile_options(/bigobj)
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")

else ()
#  add_compile_options(-Wa,-mbig-obj)
endif ()

# Headers
set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include")

set(SORTD_HEADER_FILES
	${PROJECT_INCLUDE_DIR}/base.h

	${PROJECT_INCLUDE_DIR}/utils/debug.h
	${PROJECT_INCLUDE_DIR}/utils/parameter_handler.h
	${PROJECT_INCLUDE_DIR}/utils/stopwatch.h
	${PROJECT_INCLUDE_DIR}/utils/file_reader.h
	${PROJECT_INCLUDE_DIR}/utils/key_value_heap.h
	${PROJECT_INCLUDE_DIR}/utils/dynamic_bitset.h
	${PROJECT_INCLUDE_DIR}/utils/pointer_pool.h
	${PROJECT_INCLUDE_DIR}/utils/utils.h
	
	${PROJECT_INCLUDE_DIR}/model/data.h
	${PROJECT_INCLUDE_DIR}/model/feature_vector.h
	${PROJECT_INCLUDE_DIR}/model/branch.h
	${PROJECT_INCLUDE_DIR}/model/node.h
	${PROJECT_INCLUDE_DIR}/model/container.h

	${PROJECT_INCLUDE_DIR}/solver/result.h
	${PROJECT_INCLUDE_DIR}/solver/feature_selector.h
	${PROJECT_INCLUDE_DIR}/solver/difference_computer.h
	${PROJECT_INCLUDE_DIR}/solver/similarity_lowerbound.h
	${PROJECT_INCLUDE_DIR}/solver/cost_combiner.h
	${PROJECT_INCLUDE_DIR}/solver/cost_storage.h
	${PROJECT_INCLUDE_DIR}/solver/counter.h
	${PROJECT_INCLUDE_DIR}/solver/tree.h
	${PROJECT_INCLUDE_DIR}/solver/cache.h
	${PROJECT_INCLUDE_DIR}/solver/dataset_cache.h
	${PROJECT_INCLUDE_DIR}/solver/branch_cache.h
	${PROJECT_INCLUDE_DIR}/solver/terminal_solver.h
	${PROJECT_INCLUDE_DIR}/solver/statistics.h
	${PROJECT_INCLUDE_DIR}/solver/solver.h
	${PROJECT_INCLUDE_DIR}/solver/optimization_utils.h
	${PROJECT_INCLUDE_DIR}/solver/data_splitter.h
	${PROJECT_INCLUDE_DIR}/solver/tuning.h
	
	${PROJECT_INCLUDE_DIR}/rashomon/tracker.h
	${PROJECT_INCLUDE_DIR}/rashomon/leaf_tracker.h
	${PROJECT_INCLUDE_DIR}/rashomon/branch_tracker.h
	${PROJECT_INCLUDE_DIR}/rashomon/split_tracker.h
	${PROJECT_INCLUDE_DIR}/rashomon/terminal_tracker.h
	${PROJECT_INCLUDE_DIR}/rashomon/rashomon_terminal_solver.h
	${PROJECT_INCLUDE_DIR}/rashomon/solution_tracker.h
	${PROJECT_INCLUDE_DIR}/rashomon/solution_value_heap.h
	${PROJECT_INCLUDE_DIR}/rashomon/rashomon_utils.h
	${PROJECT_INCLUDE_DIR}/rashomon/small_solution_cache.h

	${PROJECT_INCLUDE_DIR}/tasks/tasks.h
	${PROJECT_INCLUDE_DIR}/tasks/optimization_task.h
	${PROJECT_INCLUDE_DIR}/tasks/accuracy/cost_complex_accuracy.h
	${PROJECT_INCLUDE_DIR}/tasks/regression/cost_complex_regression.h
	${PROJECT_INCLUDE_DIR}/tasks/regression/ckmeans/Ckmeans.1d.dp.h
	${PROJECT_INCLUDE_DIR}/tasks/regression/ckmeans/precision.h
	${PROJECT_INCLUDE_DIR}/tasks/regression/ckmeans/within_cluster.h
	
)

set(SORTD_SRC_FILES

	${PROJECT_SOURCE_DIR}/utils/parameter_handler.cpp
	${PROJECT_SOURCE_DIR}/utils/file_reader.cpp
	${PROJECT_SOURCE_DIR}/utils/key_value_heap.cpp

	${PROJECT_SOURCE_DIR}/model/feature_vector.cpp
	${PROJECT_SOURCE_DIR}/model/data.cpp
	${PROJECT_SOURCE_DIR}/model/branch.cpp

	${PROJECT_SOURCE_DIR}/solver/feature_selector_gini.cpp
	${PROJECT_SOURCE_DIR}/solver/define_parameters.cpp
	${PROJECT_SOURCE_DIR}/solver/solver.cpp
	${PROJECT_SOURCE_DIR}/solver/cache.cpp
	${PROJECT_SOURCE_DIR}/solver/branch_cache.cpp
	${PROJECT_SOURCE_DIR}/solver/dataset_cache.cpp
	${PROJECT_SOURCE_DIR}/solver/cost_combiner.cpp
	${PROJECT_SOURCE_DIR}/solver/terminal_solver.cpp
	${PROJECT_SOURCE_DIR}/solver/cost_storage.cpp
	${PROJECT_SOURCE_DIR}/solver/counter.cpp
	${PROJECT_SOURCE_DIR}/solver/result.cpp
	${PROJECT_SOURCE_DIR}/solver/similarity_lowerbound.cpp
	${PROJECT_SOURCE_DIR}/solver/difference_computer.cpp
	${PROJECT_SOURCE_DIR}/solver/data_splitter.cpp
	
	${PROJECT_SOURCE_DIR}/rashomon/branch_tracker.cpp
	${PROJECT_SOURCE_DIR}/rashomon/split_tracker.cpp
	${PROJECT_SOURCE_DIR}/rashomon/terminal_tracker.cpp
	${PROJECT_SOURCE_DIR}/rashomon/rashomon_terminal_solver.cpp
	${PROJECT_SOURCE_DIR}/rashomon/solution_tracker.cpp
	
	${PROJECT_SOURCE_DIR}/tasks/optimization_task.cpp
	${PROJECT_SOURCE_DIR}/tasks/accuracy/cost_complex_accuracy.cpp
	${PROJECT_SOURCE_DIR}/tasks/regression/cost_complex_regression.cpp
	${PROJECT_SOURCE_DIR}/tasks/regression/ckmeans/Ckmeans.1d.dp.cpp
	${PROJECT_SOURCE_DIR}/tasks/regression/ckmeans/fill_log_linear.cpp
	${PROJECT_SOURCE_DIR}/tasks/regression/ckmeans/fill_SMAWK.cpp

	${SORTD_HEADER_FILES}
)

if (CMAKE_BUILD_TYPE STREQUAL Release)
	#set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Add source to this project's executable.
add_library(SORTDLib ${SORTD_SRC_FILES})
add_executable (SORTD ${PROJECT_SOURCE_DIR}/main.cpp)

target_include_directories(SORTDLib PUBLIC ${PROJECT_INCLUDE_DIR})
target_link_libraries (SORTD SORTDLib)


if (CMAKE_BUILD_TYPE STREQUAL Release)
else()
	if (MSVC)
		message(STATUS "Enabling profiler")
		#set_target_properties(SORTDLib SORTD PROPERTIES LINK_FLAGS "/PROFILE")
		SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /PROFILE")
	else ()
		message(STATUS "No profiler")
	endif()
endif()

# for release mode
# set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")
# set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-Wl")
#for debug mode
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
#set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -Wl,-dead_strip_dylibs -Wl,-keep_dwarf_unwind")

# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ")
# Tests

enable_testing()
set(PROJECT_TEST_DIR "test")
add_executable(CorrectSolutionTest ${PROJECT_TEST_DIR}/correct_solution_test.cpp)
target_link_libraries (CorrectSolutionTest SORTDLib)
add_test(NAME "CorrectSolutionTest" COMMAND CorrectSolutionTest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
