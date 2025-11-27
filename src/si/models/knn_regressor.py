# KB exercise 7.2

from typing import Callable
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    K-Nearest Neighbors Regressor
    
    The KNNRegressor predicts the value of a sample by averaging the values
    of its k nearest neighbors in the training dataset.
    
    Parameters
    ----------
    k : int, default=1
        The number of nearest neighbors to consider
    distance : Callable, default=euclidean_distance
        The distance function to calculate similarity between samples
    
    Attributes
    ----------
    dataset : Dataset
        The training dataset stored for making predictions
    """
    
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN Regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Fit the model to the given dataset by storing it.
        
        Parameters
        ----------
        dataset : Dataset
            The training dataset
        
        Returns
        -------
        self : KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict values for all samples in the dataset.
        
        Parameters
        ----------
        dataset : Dataset
            The test dataset to predict values for
        
        Returns
        -------
        predictions : np.ndarray
            An array of predicted values for the testing dataset (y_pred)
        
        Algorithm
        ---------
        1. Calculate the distance between each sample and various samples in the training dataset
        2. Obtain the indexes of the k most similar examples (shortest distance)
        3. Use the previous indexes to retrieve the corresponding values in Y
        4. Calculate the average of the values obtained in step 3
        5. Apply steps 1, 2, 3, and 4 to all samples in the testing dataset
        """
        # Initialize array to store predictions
        predictions = np.zeros(dataset.shape()[0])
        
        # Loop through each sample in the test dataset
        for i, sample in enumerate(dataset.X):
            # 1. Calculate distances between the sample and all training samples
            distances = self.distance(sample, self.dataset.X)
            
            # 2. Obtain the indexes of the k most similar examples (shortest distance)
            k_nearest_indices = np.argsort(distances)[:self.k]
            
            # 3. Use the previous indexes to retrieve the corresponding values in Y
            k_nearest_values = self.dataset.y[k_nearest_indices]
            
            # 4. Calculate the average of the values obtained in step 3
            predictions[i] = np.mean(k_nearest_values)
        
        # 5. Return predictions for all samples
        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the RMSE between predictions and actual values.
        
        Parameters
        ----------
        dataset : Dataset
            The dataset with true values
        predictions : np.ndarray
            The predicted values
        
        Returns
        -------
        error : float
            The RMSE error between predictions and actual values
        """
        return rmse(dataset.y, predictions)