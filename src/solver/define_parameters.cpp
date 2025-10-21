/**
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/
#include "utils/parameter_handler.h"

namespace SORTD {

	ParameterHandler ParameterHandler::DefineParameters() {
		ParameterHandler parameters;

		parameters.DefineNewCategory("Main Parameters");
		parameters.DefineNewCategory("Algorithmic Parameters");
		parameters.DefineNewCategory("Objective Parameters");
		parameters.DefineNewCategory("Task-specific Parameters");

		parameters.DefineStringParameter
		(
			"task",
			"Task to optimize.",
			"accuracy",
			"Main Parameters",
			{ "cost-complex-accuracy", "cost-complex-regression" }
		);

		parameters.DefineStringParameter
		(
			"file",
			"Location to the (training) dataset.",
			"", //default value
			"Main Parameters"
		);

		parameters.DefineStringParameter
		(
			"test-file",
			"Location to the test dataset.",
			"", //default value
			"Main Parameters",
			{},
			true
		);

		parameters.DefineFloatParameter
		(
			"time",
			"Maximum runtime given in seconds.",
			600, //default value
			"Main Parameters",
			0, //min value
			INT32_MAX //max value
		);

		parameters.DefineIntegerParameter
		(
			"max-depth",
			"Maximum allowed depth of the tree, where the depth is defined as the largest number of *decision/feature nodes* from the root to any leaf. Depth greater than four is usually time consuming.",
			3, //default value
			"Main Parameters",
			0, //min value
			20 //max value
		);

		parameters.DefineIntegerParameter
		(
			"max-num-nodes",
			"Maximum number of *decision/feature nodes* allowed. Note that a tree with k feature nodes has k+1 leaf nodes.",
			INT32_MAX, //default value
			"Main Parameters",
			0,
			INT32_MAX
		);

		parameters.DefineIntegerParameter
		(
			"max-num-features",
			"Maximum number of features that are considered from the dataset (in order of appearance).",
			INT32_MAX, // default value,
			"Main Parameters",
			1,
			INT32_MAX
		);

		parameters.DefineIntegerParameter
		(
			"num-instances",
			"Number of instances that are considered from the dataset (in order of appearance).",
			INT32_MAX, // default value,
			"Main Parameters",
			1,
			INT32_MAX
		);

		parameters.DefineIntegerParameter
		(
			"num-extra-cols",
			"Number of extra columns that need to be read after the label and before the binary feature vector.",
			0, // default value,
			"Main Parameters",
			0,
			INT32_MAX
		);

		parameters.DefineBooleanParameter
		(
			"verbose",
			"Determines if the solver should print logging information to the standard output.",
			true,
			"Main Parameters"
		);

		parameters.DefineFloatParameter
		(
			"train-test-split",
			"The percentage of instances for the test set",
			0.0, //default value
			"Main Parameters",
			0, //min value
			1.0 //max value
		);

		parameters.DefineBooleanParameter
		(
			"stratify",
			"Stratify the train-test split",
			true,
			"Main Parameters"
		);

		parameters.DefineIntegerParameter
		(
			"min-leaf-node-size",
			"The minimum size of leaf nodes",
			1, // default value
			"Main Parameters",
			1, //min value
			INT32_MAX // max value
		);

		parameters.DefineStringParameter
		(
			"feature-ordering",
			"Feature ordering strategy used to determine the order in which features will be inspected in each node.",
			"in-order", //default value
			"Algorithmic Parameters",
			{ "in-order", "gini" }
		);

		parameters.DefineIntegerParameter
		(
			"random-seed",
			"Random seed used only if the feature-ordering is set to random. A seed of -1 assings the seed based on the current time.",
			4,
			"Algorithmic Parameters",
			-1,
			INT32_MAX
		);

		parameters.DefineBooleanParameter
		(
			"use-branch-caching",
			"Use branch caching to store computed subtrees.",
			//\"Dataset\" is more powerful than \"branch\" but may required more computational time. Need to be determined experimentally. \"Closure\" is experimental and typically slower than other options.",
			false, //default value
			"Algorithmic Parameters"
		);

		parameters.DefineBooleanParameter
		(
			"use-dataset-caching",
			"Use dataset caching to store computed subtrees. Dataset-caching is more powerful than branch-caching but may required more computational time.",
			true, //default value
			"Algorithmic Parameters"
		);

		parameters.DefineIntegerParameter
		(
			"duplicate-factor",
			"Duplicates the instances the given amount of times. Used for stress-testing the algorithm, not a practical parameter.",
			1,
			"Algorithmic Parameters",
			1,
			INT32_MAX
		);


		parameters.DefineFloatParameter
		(
			"cost-complexity",
			"The cost for adding an extra node to the tree. 0.01 means one extra node is only jusitified if it results in at least one percent better training accuracy score.",
			0.00, // default value
			"Objective Parameters",
			0.0, //min value
			1.0 //max value
		);


		parameters.DefineStringParameter
		(
			"regression-bound",
			"The type of bound to use, only for cost-complex-regression task.",
			"equivalent", //default value
			"Task-specific Parameters",
			{ "equivalent", "kmeans" },
			true
		);

        parameters.DefineBooleanParameter
        (
                "use-rashomon-multiplier",
                "Use rashomon multiplier to determine how worse a rashomon tree can be compared to the optimal tree",
                true, //default value
                "Algorithmic Parameters"
        );

        parameters.DefineFloatParameter(
                "rashomon-multiplier",
                "Upper bound of the rashomon set trees calculated by best_obj * (1 + rashomon-multiplier).",
                0.2, // default
                "Algorithmic Parameters",
                0.0, // min
                DBL_MAX // max
        );

        parameters.DefineBooleanParameter
                (
                    "track-tree-statistics",
                    "Calculates feature and node number statistics.",
                    false, //default value
                    "Algorithmic Parameters"
                );

        parameters.DefineIntegerParameter
                (
                        "max-num-trees",
                        "Upper bound for the number of trees in the rashomon set.",
                        INT64_MAX,
                        "Algorithmic Parameters",
                        1,
                        INT64_MAX
                );

        parameters.DefineBooleanParameter
                (
                        "ignore-trivial-extensions",
                        "Ignores the solutions where a split cannot improve accuracy",
                        false, //default value
                        "Algorithmic Parameters"
                );

		return parameters;
	}
}