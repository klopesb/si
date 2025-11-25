#Implement the RandomForestClassifier class by taking into consideration the structure of the class shown in the next slides.
from typing import Literal, Tuple, Union
import numpy as np
from collections import Counter
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier(Model):
    def __init__(self, n_estimators: int = 100, max_features: int = None,min_sample_split: int = 2, max_depth: int = 10,
                 mode: Literal['gini', 'entropy'] = 'gini', seed: int = 42, **kwargs) -> None:
        
        """
           Random Forest Classifier
    
            An ensemble learning method that constructs multiple decision trees during training
            and outputs the class that is the mode of the classes output by individual trees.
            
            Parameters
            ----------
            n_estimators : int, default=100
                The number of decision trees in the forest
            max_features : int, optional
                The maximum number of features to consider per tree
                If None, defaults to sqrt(n_features)
            min_sample_split : int, default=2
                The minimum number of samples required to split an internal node
            max_depth : int, default=10
                The maximum depth of each tree
            mode : Literal['gini', 'entropy'], default='gini'
                The impurity calculation mode
            seed : int, default=42
                Random seed for reproducibility
            
            Attributes
            ----------
            trees : list
        List of tuples containing (features_used, trained_tree) for each tree in the forest
        """
        
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        # estimated parameter
        self.trees = []     # list of tuples: (feature_indices, fitted_tree)

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Train all trees in the forest.
        """
        np.random.seed(self.seed)
        n_samples, n_features = dataset.shape()

        # if max_features not given, use sqrt(n_features)
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        self.trees = []

        for _ in range(self.n_estimators):

            # ---- 1. Bootstrap samples (with replacement) ----
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = dataset.X[bootstrap_indices]
            y_boot = dataset.y[bootstrap_indices]

            # ---- 2. Select max_features random features ----
            feature_indices = np.random.choice(
                n_features, size=self.max_features, replace=False
            )

            X_boot_feat = X_boot[:, feature_indices]

            # ---- 3. Build a Dataset with selected features ----
            boot_dataset = Dataset(
                X=X_boot_feat,
                y=y_boot,
                features=[dataset.features[i] for i in feature_indices],
                label=dataset.label
            )

            # ---- 4. Train a decision tree ----
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(boot_dataset)

            # ---- 5. Store (features_used, trained_tree) ----
            self.trees.append((feature_indices, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict using all trees (majority vote).
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
        return accuracy(dataset.y, predictions)
