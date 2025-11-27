from typing import Literal
import numpy as np
from collections import Counter
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier

#KB - Exercise 9.1
class RandomForestClassifier(Model):
    """
    Random Forest Classifier for classification tasks.
    
    An ensemble learning method that constructs multiple decision trees during training
    and outputs the class that is the mode of the classes output by individual trees.
    Each tree is trained on a bootstrap sample of the data with a random subset of features,
    which reduces overfitting and improves generalization.
    
    The prediction is made by majority voting across all trees in the forest.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of decision trees in the forest
    max_features : int, optional
        The maximum number of features to consider when building each tree
        If None, defaults to sqrt(n_features)
    min_sample_split : int, default=2
        The minimum number of samples required to split an internal node in each tree
    max_depth : int, default=10
        The maximum depth of each decision tree
    mode : Literal['gini', 'entropy'], default='gini'
        The criterion used to measure the quality of a split
        - 'gini': Gini impurity
        - 'entropy': Information gain
    seed : int, default=42
        Random seed for reproducibility of bootstrap sampling and feature selection
    **kwargs : dict
        Additional keyword arguments passed to the parent Model class
    
    Attributes
    ----------
    trees : list of tuples
        List containing (feature_indices, trained_tree) for each tree in the forest
        - feature_indices: np.ndarray of selected feature indices for that tree
        - trained_tree: fitted DecisionTreeClassifier instance
    """
    def __init__(self, n_estimators: int = 100, max_features: int = None,min_sample_split: int = 2, max_depth: int = 10,
                 mode: Literal['gini', 'entropy'] = 'gini', seed: int = 42, **kwargs) -> None:
        
        """
        Initialize the Random Forest Classifier.
        
        Parameters
        ----------
        n_estimators : int, default=100
            Number of decision trees to create
        max_features : int, optional
            Maximum features per tree (defaults to sqrt(n_features) if None)
        min_sample_split : int, default=2
            Minimum samples needed to split a node
        max_depth : int, default=10
            Maximum depth of each tree
        mode : Literal['gini', 'entropy'], default='gini'
            Impurity measure for splitting
        seed : int, default=42
            Random seed for reproducibility
        **kwargs : dict
            Additional arguments for the parent Model class
        """
        
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []     # list of tuples: (feature_indices, fitted_tree)

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Train the Random Forest by building multiple decision trees.
        
        The fitting process for each tree:
        1. Creates a bootstrap sample (sampling with replacement)
        2. Randomly selects a subset of features
        3. Trains a decision tree on the bootstrapped data with selected features
        4. Stores the feature indices and trained tree
        
        This process is repeated n_estimators times to create the forest.
        
        Parameters
        ----------
        dataset : Dataset
            Training dataset containing features (X) and labels (y)
        
        Returns
        -------
        self : RandomForestClassifier
            The fitted Random Forest model
        """
        np.random.seed(self.seed)
        n_samples, n_features = dataset.shape()

        # if max_features not given, use sqrt(n_features)
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        self.trees = []

        for _ in range(self.n_estimators):

            #Bootstrap samples (with replacement)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = dataset.X[bootstrap_indices]
            y_boot = dataset.y[bootstrap_indices]

            # Select max_features random features
            feature_indices = np.random.choice(
                n_features, size=self.max_features, replace=False
            )

            X_boot_feat = X_boot[:, feature_indices]

            # Build a Dataset with selected features
            boot_dataset = Dataset(
                X=X_boot_feat,
                y=y_boot,
                features=[dataset.features[i] for i in feature_indices],
                label=dataset.label
            )

            #Train a decision tree
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(boot_dataset)

            #(features_used, trained_tree)
            self.trees.append((feature_indices, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels using majority voting across all trees.
        
        The prediction process:
        1. Each tree makes predictions using only its selected features
        2. Collects all predictions from all trees
        3. For each sample, selects the most common predicted class (majority vote)
        
        Parameters
        ----------
        dataset : Dataset
            Dataset containing features (X) to make predictions on
        
        Returns
        -------
        np.ndarray
            Predicted class labels for each sample, shape (n_samples,)
        """
        all_predictions = []

        for feature_indices, tree in self.trees:
            # extract only the features the tree was trained on
            X_subset = dataset.X[:, feature_indices]
            sub_dataset = Dataset(
                X=X_subset,
                y=dataset.y,
                features=[dataset.features[i] for i in feature_indices],
                label=dataset.label
            )
            preds = tree.predict(sub_dataset)
            all_predictions.append(preds)

        # shape: (n_estimators, n_samples)
        all_predictions = np.array(all_predictions)

        # ---- majority vote ----
        final_predictions = []
        for i in range(dataset.shape()[0]):
            sample_preds = all_predictions[:, i]
            most_common = Counter(sample_preds).most_common(1)[0][0]
            final_predictions.append(most_common)

        return np.array(final_predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the accuracy of predictions.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset containing true labels (y)
        predictions : np.ndarray
            Predicted labels
        
        Returns
        -------
        float
            Accuracy score (proportion of correct predictions)
        """
        return accuracy(dataset.y, predictions)
