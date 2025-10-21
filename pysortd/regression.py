"""
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
"""

from pysortd.base import BaseSORTDSolver
from pysortd.binarizer import get_column_types
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import Interval, StrOptions
from typing import Optional
import numpy as np
import pandas as pd
import numbers
from pathlib import Path
import os

class SORTDRegressor(BaseSORTDSolver):

    _parameter_constraints: dict = {**BaseSORTDSolver._parameter_constraints, 
        "regression_lower_bound": [StrOptions({"equivalent", "kmeans"})]
    }

    def __init__(self,
                 optimization_task : str = "cost-complex-regression",
                 max_depth : int = 3,
                 min_leaf_node_size : int = 1,
                 time_limit : float = 600,
                 cost_complexity : float = 0.01,
                 regression_lower_bound : str = "kmeans",
                 use_branch_caching: bool = False,
                 use_dataset_caching: bool = True,
                 verbose : bool = False,
                 random_seed: int = 27, 
                 continuous_binarize_strategy: str = 'quantile',
                 n_thresholds: int = 5,
                 n_categories: int = 5,
                 max_num_binary_features: int = None,
                 use_rashomon_multiplier : bool = True,
                 rashomon_multiplier: float = 1.0,
                 track_tree_statistics : bool = False,
                 ignore_trivial_extensions : bool = False,
                 max_num_trees : int = 2**63 -1):
        """
        Construct a SORTDRegressor

        Parameters:
            optimization_task: the objective used for optimization. Default = cost-complex-regression
            max_depth: the maximum depth of the tree
            min_leaf_node_size: the minimum number of training instance that should end up in every leaf node
            time_limit: the time limit in seconds for fitting the tree
            cost_complexity: the cost of adding a branch node, expressed as a percentage. E.g., 0.01 means a branching node may be added if it increases the training accuracy by at least 1%.
                only used when optimization_task == "cost-complex-regression'
            regression_lower_bound: the lower bound used by the cost-complex-regression task: kmeans or equivalent
            use_branch_caching: Enable/Disable branch caching (typically the slower caching strategy. May be faster in some scenario's)
            use_dataset_caching: Enable/Disable dataset caching (typically the faster caching strategy)
            verbose: Enable/Disable verbose output
            random_seed: the random seed used by the solver (for example when creating folds)
            continuous_binarization_strategy: the strategy used for binarizing continuous features
            n_thresholds: the number of thresholds to use per continuous feature
            n_categories: the number of categories to use per categorical feature
            max_num_binary_features: the maximum number of binary features (selected by random forest feature importance)
            use_rashomon_multiplier: Enable/Disable the rashomon multiplier use. If enabled, max_num_trees should be nonzero.
            rashomon_multiplier: the Rashomon multiplier limits the Rashomon set to this factor above the optimal solution
            track_tree_statistics: Enable/Disable the use of tracking statistics related to the rashomon trees
            ignore_trivial_extensions: Enable/Disable removing depth-1 solutions with same left and right label assignment
            max_num_trees : Upper bound for the number of trees in the Rashomon set
        """
        self._extra_permitted_params = ["regression_lower_bound"]
        if not optimization_task in ["cost-complex-regression"]:
            raise ValueError(f"Invalid value for optimization_task: {optimization_task}")
        BaseSORTDSolver.__init__(self, optimization_task,
            max_depth=max_depth,
            min_leaf_node_size=min_leaf_node_size,
            time_limit=time_limit,
            cost_complexity=cost_complexity,
            feature_ordering="in-order",
            use_branch_caching=use_branch_caching,
            use_dataset_caching=use_dataset_caching,
            verbose=verbose,
            random_seed=random_seed,
            continuous_binarize_strategy=continuous_binarize_strategy,
            n_thresholds=n_thresholds,
            n_categories=n_categories,
            use_rashomon_multiplier=use_rashomon_multiplier,
            rashomon_multiplier=rashomon_multiplier,
            max_num_binary_features=max_num_binary_features,
            track_tree_statistics=track_tree_statistics,
            ignore_trivial_extensions = ignore_trivial_extensions,
            max_num_trees=max_num_trees)
        self._label_type = np.double
        self.regression_lower_bound = regression_lower_bound

    def _initialize_param_handler(self):
        super()._initialize_param_handler()
        self._params.regression_lower_bound = self.regression_lower_bound
        # self._params.use_task_lower_bound = True
        return self._params

