/**
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/

#pragma once

namespace SORTD {

	struct Statistics {
		Statistics() {
			num_terminal_nodes_with_node_budget_one = 0;
			num_terminal_nodes_with_node_budget_two = 0;
			num_terminal_nodes_with_node_budget_three = 0;

			total_time = 0;
			time_in_terminal_node = 0;
			time_merging = 0;
			time_lb_merging = 0;
			time_ub_subtracting = 0;
			time_reconstructing = 0;
			
			time_rashomon_total = 0;
			time_rashomon_init = 0;
			time_rashomon_terminal = 0;
			time_rashomon_combine_terminal = 0;
			time_rashomon_split_heap = 0;

			num_cache_hit_nonzero_bound = 0;
			num_cache_hit_optimality = 0;
		}

		void Print() {
			std::cout << "Total time elapsed: " << total_time << std::endl;
//			std::cout << "\tTerminal time: " << time_in_terminal_node << std::endl;
//			std::cout << "\tMerging time: " << time_merging << std::endl;
//			std::cout << "\tLB Merging time: " << time_lb_merging << std::endl;
//			std::cout << "\tUB Substracting time: " << time_ub_subtracting << std::endl;
//			std::cout << "\tReconstructing time: " << time_reconstructing << std::endl;
			std::cout << "\tRashomon set total construction time: " << time_rashomon_total << std::endl;
			std::cout << "\tRashomon set initialization time: " << time_rashomon_init << std::endl;
			std::cout << "\tRashomon set terminal time: " << time_rashomon_terminal << std::endl;
			std::cout << "\tRashomon set combine terminal time: " << time_rashomon_combine_terminal << std::endl;
			std::cout << "\tRashomon set split heap time: " << time_rashomon_split_heap << std::endl;
			
//			std::cout << "Terminal calls: " << num_terminal_nodes_with_node_budget_one + num_terminal_nodes_with_node_budget_two + num_terminal_nodes_with_node_budget_three << std::endl;
//			std::cout << "\tTerminal 1 node: " << num_terminal_nodes_with_node_budget_one << std::endl;
//			std::cout << "\tTerminal 2 node: " << num_terminal_nodes_with_node_budget_two << std::endl;
//			std::cout << "\tTerminal 3 node: " << num_terminal_nodes_with_node_budget_three << std::endl;
		}

		size_t num_terminal_nodes_with_node_budget_one;
		size_t num_terminal_nodes_with_node_budget_two;
		size_t num_terminal_nodes_with_node_budget_three;

		size_t num_cache_hit_optimality;
		size_t num_cache_hit_nonzero_bound;


		double total_time;
		double time_in_terminal_node;
		double time_lb_merging;
		double time_ub_subtracting;
		double time_merging;
		double time_reconstructing;

		double time_rashomon_total;
		double time_rashomon_init;
		double time_rashomon_terminal;
		double time_rashomon_combine_terminal;
		double time_rashomon_split_heap;
	};
}