"""
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
"""

from pysortd.base import BaseSORTDSolver
from typing import Optional
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted
from pysortd.utils import _color_brew
import numpy as np
import numbers
import warnings

class SORTDClassifier(BaseSORTDSolver):
    """
    SORTDClassifier returns optimal classification trees.
    It supports several objectives, as specified by the optimization task parameter
    """

    _parameter_constraints: dict = {**BaseSORTDSolver._parameter_constraints }

    def __init__(self, 
                 optimization_task : str = "cost-complex-accuracy",
                 max_depth : int = 3,
                 min_leaf_node_size: int = 1,
                 time_limit : float = 600,
                 cost_complexity : float = 0.01,
                 feature_ordering : str = "gini", 
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
                 ignore_trivial_extensions: bool = False,
                 max_num_trees : int = 2**63 -1):
        """
        Construct a SORTDClassifier

        Parameters:
            optimization_task: the objective used for optimization. Default = accuracy
            max_depth: the maximum depth of the tree
            min_leaf_node_size: the minimum number of training instance that should end up in every leaf node
            time_limit: the time limit in seconds for fitting the tree
            cost_complexity: the cost of adding a branch node, expressed as a percentage. E.g., 0.01 means a branching node may be added if it increases the training accuracy by at least 1%.
                only used when optimization_task == "cost-complex-accuracy'
            feature_ordering: heuristic for the order that features are checked. Default: "gini", alternative: "in-order": the order in the given data
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
        if not optimization_task in ["accuracy", "cost-complex-accuracy",]:
            raise ValueError(f"Invalid value for optimization_task: {optimization_task}")
        BaseSORTDSolver.__init__(self, optimization_task, 
            max_depth=max_depth,
            min_leaf_node_size=min_leaf_node_size,
            time_limit=time_limit,
            cost_complexity=cost_complexity,
            feature_ordering=feature_ordering,
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
        
    def _initialize_param_handler(self):
        super()._initialize_param_handler()
        #self._params.use_task_lower_bound = self.use_task_lower_bound
        return self._params

    def fit(self, X, y, extra_data=None, categorical=None):
        """
        Fits a SORTD Classification model to the given training data.

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix

            y : array-like, shape = (n_samples)
            Target vector

            extra_data : array-like, shape = (n_samples, n_data_items)
            An array (optional) that represents extra data per instance

            categorical : array-like, 
            List of column names that are categorical

        Returns:
            BaseSORTDSolver

        Raises:
            ValueError: If x or y is None or if they have different number of rows.
        """
        self.n_classes_ = len(np.unique(y))
        return super().fit(X, y, extra_data, categorical)

    def predict_proba(self, X, extra_data=None):
        """
        Predicts the probabilities of the target class for the given input feature data.

        Args:
            X : array-like, shape = (n_samples, n_features)
            Data matrix
            extra_data : array-like, shape = (n_samples)
            Extra data (if required)

        Returns:
            numpy.ndarray: A 2D array that represents the predicted class probabilities of the test data.
                The i-j-th element in this array corresponds to the predicted class probablity for the j-th class of the i-th instance in `X`.
        """
        check_is_fitted(self, "fit_result")
        X = self._binarize_data(X, reset=False)
        X = self._process_predict_data(X)
        extra_data = self._process_extra_data(X, extra_data)
        probabilities = np.zeros((len(X), self.n_classes_))
        train_data = (self.train_X_, self.train_y_)
        self._recursive_predict_proba(self.tree_, probabilities, np.array(range(0, len(X))), X, train_data)
        # Check that all rows sum to proability 1 (account for floating errors)
        assert (probabilities.sum(axis=1).min() >= 1-1e-4)
        return probabilities
    
    def _recursive_predict_proba(self, tree, probabilities, indices, X, train_data):
        train_X = train_data[0]
        train_y = train_data[1]
        if tree.is_leaf_node():
            n = len(train_y)
            assert(n > 0)
            all_counts = np.zeros(self.n_classes_)
            unique, counts = np.unique(train_y, return_counts=True)
            for label, count in zip(unique, counts):
                all_counts[label] = count
            probs = all_counts / n
            probabilities[indices] = probs
            assert(probs[tree.label] >= max(probs) - 1e-5)
        else:
            indices_left  = np.intersect1d(np.argwhere(~X[:, tree.feature]), indices)
            indices_right = np.intersect1d(np.argwhere( X[:, tree.feature]), indices)
            sel = train_X[:, tree.feature].astype(bool)
            train_data_left  = (train_X[~sel, :], train_y[~sel])
            train_data_right = (train_X[ sel, :], train_y[ sel])
            self._recursive_predict_proba(tree.left_child,  probabilities, indices_left,  X, train_data_left)
            self._recursive_predict_proba(tree.right_child, probabilities, indices_right, X, train_data_right)

    def _export_dot_leaf_node(self, fh, node, node_id, label_names, train_data):
        if not hasattr(self, "_colors"):
            self._colors = _color_brew(self.n_classes_)
        color = self._colors[node.label]
        return super()._export_dot_leaf_node(fh, node, node_id, label_names, train_data, color=color)