# SORTeD Rashomon Sets of Sparse Decision Trees: Scalable Anytime Enumeration

SORTD is a framework for Rashomon set of decision trees with totally ordered optimization tasks. Currently SORTD has the following tasks:
* Cost-Complex Classification (Decision trees regularized with the number of leaf nodes)
* Cost-Complex Regression (Regression trees regularized with the number of leaf nodes)

## Paper

Elif Arslan, Jacobus G. M. van der Linden, Serge P. Hoogendoorn, Marco Rinaldi, Emir Demirović, “SORTeD Rashomon Sets of Sparse Decision Trees: Anytime Enumeration”, The Thirty-Ninth Annual Conference on Neural Information Processing Systems. 2025.

## Python usage

### Install from source using pip
To compile the `pysortd` Python package, navigate to the codebase directory and run the following command:

```sh
cd pysortd
pip install . 
```

### Example usage
`pysortd` can be used, for example, as follows:

```python
from pysortd import SORTDClassifier
import pandas as pd

df = pd.read_csv('data/accuracy/monk2.csv',delimiter=" ", header=None)
X, y = df.iloc[:, 1:], df.iloc[:, 0]

model = SORTDClassifier("cost-complex-accuracy",max_depth = 3, verbose=True,cost_complexity=0.01,
                        use_rashomon_multiplier=True,rashomon_multiplier=0.1, max_num_trees=100)
model.fit(X,y)
rashomon_set_size = model.rashomon_set_size
```


## C++ usage

### Compiling
The code can be compiled on Windows or Linux by using cmake. For Windows users, cmake support can be installed as an extension of Visual Studio and then this repository can be imported as a CMake project.

For Linux users, they can use the following commands:

```sh
mkdir build
cd build
cmake ..
cmake --build .
```
The compiler must support the C++17 standard.

### Running
After SORTD is built, the following command can be used (for example):
```sh
./SORTD -task cost-complex-accuracy -file ../data/accuracy/monk2.csv -max-depth 3 -cost-complexity 0.01 -use-rashomon-multiplier true -rashomon-multiplier 0.1 -max-num-trees 100
```

Run the program without any parameters to see a full list of the available parameters.

## Applications
Currently, SORTD implements the following optimization tasks:
 
* `cost-complex-accuracy`: `SORTDClassifier` minimizes the misclassification score plus the cost for adding a leaf node by the parameter `cost_complexity`.
* `cost-complex-regression`: `SORTDRegressor` minimizes the _sum of squared errors_ plus the cost for adding a leaf node by the parameter `cost_complexity`. 

See [examples/regression_example.py](examples/regression_example.py) for a regression example.

## Parameters
SORTD can be configured by the following parameters:
* `max_depth` : The maximum depth of the trees. Note that a tree of depth zero has a single leaf node. A tree of depth one has one branching node and two leaf nodes.
* `cost_complexity`: The cost of adding one more leaf node to the tree.
* `max_num_trees`: The limit on the number of trees in the Rashomon set. 
* `use_rashomon_multiplier`: Enables or disables the use of the Rashomon multiplier. When disabled, the Rashomon bound is set to the root leaf solution value. In this case, `max_num_trees` should also be set.
* `rashomon_multiplier`: The Rashomon multiplier value to bound the Rashomon set.
* `ignore_trivial_extensions`: Enables or disables discarding trees with trivial extensions.
* `min_leaf_node_size` : The minimum number of samples required in each leaf node.
* `time_limit` : The run time limit in seconds. If the time limit is exceeded a possibly non-optimal tree is returned.
* `feature_ordering` : The order in which the features are considered for branching. Default is `"gini"` which sorts the features by gini-impurity decrease. The alternative (and default for regression and survival analysis) is `"in-order"` which considers the feature in order of appearance.
* `use_branch_caching` : Enables or disables the use of branch caching.
* `use_dataset_caching` : Enables or disables the use of dataset caching.
* `verbose` : Enables or disables verbose output.
* `random_seed` : The random seed.

## Miscellaneous 
* SORTD assumes features and cost-complex classification labels are binary.
* The label is in the first column in all datasets.
