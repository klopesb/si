from typing import Callable
import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
from sklearn.feature_selection import SelectPercentile as SklearnSelectPercentile

#KB Exercise 
class SelectPercentile(Transformer):
    """
    Select features according to a percentile of the highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function.

    Parameters
    ----------
    score_func: callable, default=f_classification
        Function taking dataset and returning a pair of arrays (scores, p_values)
    percentile: int, default=10
        Percentile of top features to select (between 0 and 100).

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """

    def __init__(self, score_func: Callable = f_classification, percentile: int = 10, **kwargs):
        """
        Select features according to a percentile of the highest scores.

        Parameters
        ----------
        score_func: callable, default=f_classification
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: int, default=10
            Percentile of top features to select (between 0 and 100).
        """
        super().__init__(**kwargs)
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        It fits SelectPercentile to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting features based on the specified percentile.
        Handles ties at the threshold to maintain the correct number of features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the selected features based on percentile.
        """
        
        # Create sklearn's SelectPercentile with a dummy score function
        # We'll use our pre-computed F scores instead
        sklearn_selector = SklearnSelectPercentile(percentile=self.percentile)
        
        # Manually set the scores in the sklearn selector
        sklearn_selector.scores_ = self.F
        sklearn_selector.pvalues_ = self.p
        
        # Use sklearn's get_support to get the mask of selected features
        mask = sklearn_selector.get_support()
        
        # Get indices of selected features
        idxs = np.where(mask)[0]
        
        # Select features by name
        features = np.array(dataset.features)[idxs]
        
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)


if __name__ == '__main__':


    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                   [0, 1, 4, 3],
                                   [0, 1, 1, 3]]),
                       y=np.array([0, 1, 0]),
                       features=["f1", "f2", "f3", "f4"],
                       label="y")

    selector = SelectPercentile(percentile=50)
    selector = selector.fit(dataset)
    print("\nOriginal example - F-scores:", selector.F)
    print("Selected features (50%):", selector.transform(dataset).features)
    print("Number of features:", len(selector.transform(dataset).features))