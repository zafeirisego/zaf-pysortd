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
		int depth = context.GetBranch().Depth();
		int portion data.Size();
		double penalty = ((double) (depth * portion)) / ((double) (train_summary.size));
		error += cost_complexity_parameter * penalty;
                return error;
        }

	int AverageDepthAccuracy::GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
		int error = 0;
	        for (int k = 0; k < data.NumLabels(); k++) {
			if (k == label) continue;
	                error += data.NumInstancesForLabel(k);		
		}	
		return error;
	}

	Node<CostComplexAccuracy> CostComplexAccuracy::ComputeLowerBound(const ADataView& data, const Branch& branch, int max_depth, int num_nodes) {
				// Equivalent Points Bound adapted from Angelino, E., Larus-Stone, N., Alabi, D., Seltzer, M., & Rudin, C. (2018). 
				// 		// Learning certifiably optimal rule lists for categorical data. Journal of Machine Learning Research, 18(234), 1-78.
				auto lb = Node<CostComplexAccuracy>(best);
				auto& hashmap = lower_bound_cache[branch.Depth()];
				auto it = hashmap.find(branch);
				if (it != hashmap.end()) {
				             return hashmap[branch];
				}
				lb.solution = 0;
				lb.num_nodes_left = 0;
				lb.num_nodes_right = 0;
				const int num_labels = data.NumLabels();
				std::vector<std::vector<const AInstance*>::const_iterator> iterators;
				std::vector<std::vector<const AInstance*>::const_iterator> ends;
				for (int k = 0; k < num_labels; k++) {
					iterators.push_back(data.GetInstancesForLabel(k).begin());
				        ends.push_back(data.GetInstancesForLabel(k).end());
				}
				std::vector<int> labels(num_labels);
				std::iota(labels.begin(), labels.end(), 0);
				auto comp_with_check_end = [&iterators, &ends](const int k1, const int k2) {
					if (iterators[k1] == ends[k1]) return false;
					if (iterators[k2] == ends[k2]) return true;
					return (*iterators[k1])->GetID() < (*iterators[k2])->GetID();
				};
				auto comp = [&iterators, &ends](const int k1, const int k2) {
					return (*iterators[k1])->GetID() < (*iterators[k2])->GetID();
				};
				std::sort(labels.begin(), labels.end(), comp_with_check_end);
				// pop empty labels 
				while (iterators[labels[labels.size() - 1]] == ends[labels[labels.size() - 1]])
					labels.pop_back();
				// Initialize prev with the first instance in order of the features
				std::vector<int> class_counts(num_labels, 0);
				int current_label = labels[0];
				const AInstance* prev = *iterators[current_label];
				class_counts[current_label]++;
				iterators[current_label]++;
				int n = 1;
				// while the first iterator still has instances
				while (labels.size() > 0 && iterators[labels[0]] != ends[labels[0]]) {
				        current_label = labels[0];
					if (static_cast<const Instance<int, CCAccExtraData>*>(prev)->GetExtraData().unique_feature_vector_id ==
				            static_cast<const Instance<int, CCAccExtraData>*>(*iterators[current_label])->GetExtraData().unique_feature_vector_id) {
						class_counts[current_label]++;
				         	n++;
					} else {
						if (n > 1) {
				                     //check if the label count is unique
				          		int largest = class_counts[0];
							for (int k = 1; k < num_labels; k++) {
								if (class_counts[k] > largest) {
									largest = class_counts[k];
								}
							}
							int min_error = n - largest;
							lb.solution += min_error;
						}
				                std::fill(class_counts.begin(), class_counts.end(), 0);
						class_counts[current_label]++;
						n = 1;
					}
				        prev = *iterators[current_label];
					iterators[current_label]++;
					if (iterators[current_label] == ends[current_label]) {
						labels.erase(labels.begin());
						continue;
					}
				        //std::sort(labels.begin(), labels.end(), comp);
					for (int k = 1; k < labels.size(); k++) {
						if (comp(labels[k], current_label)) {
				          		std::swap(labels[k], labels[k-1]);
						} else {
				          		break;
				        	}
				        }			
				}
				// Compute the error from the last group
				if (n > 1) {
				      // Check if the label count is unique
				      int largest = class_counts[0];
				      for (int k = 1; k < num_labels; k++) {
						if (class_counts[k] > largest) {
							largest = class_counts[k];
						}
				      }
				      int min_error = n - largest;
				      lb.solution += min_error;
				}
				hashmap[branch] = lb;
				return lb;
		}

}
