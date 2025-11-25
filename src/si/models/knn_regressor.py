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


if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    
    print("=" * 70)
    print("TEST 1: Simple synthetic regression problem")
    print("=" * 70)
    
    # Create a simple dataset: y = 2*x1 + 3*x2 + noise
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples) * 0.1
    
    dataset = Dataset(X=X, y=y, features=["x1", "x2"], label="y")
    
    # Split the dataset
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    
    print(f"Training samples: {train_dataset.shape()[0]}")
    print(f"Testing samples: {test_dataset.shape()[0]}")
    print()
    
    # Test different values of k
    k_values = [1, 3, 5, 10, 20]
    
    print("Testing different k values:")
    print("-" * 70)
    for k in k_values:
        knn = KNNRegressor(k=k)
        knn.fit(train_dataset)
        score = knn.score(test_dataset)
        print(f"k={k:2d} -> RMSE: {score:.4f}")
    print()
    
    print("=" * 70)
    print("TEST 2: Detailed prediction example")
    print("=" * 70)
    
    # Use k=3 for detailed example
    knn_best = KNNRegressor(k=3)
    knn_best.fit(train_dataset)
    
    # Get predictions
    predictions = knn_best.predict(test_dataset)
    
    # Show first 10 predictions vs actual values
    print("First 10 predictions vs actual values:")
    print(f"{'Actual':>10s} {'Predicted':>10s} {'Error':>10s}")
    print("-" * 35)
    for i in range(min(10, len(predictions))):
        actual = test_dataset.y[i]
        predicted = predictions[i]
        error = abs(actual - predicted)
        print(f"{actual:10.4f} {predicted:10.4f} {error:10.4f}")
    
    print()
    print(f"Overall RMSE on test set: {knn_best.score(test_dataset):.4f}")
    print()
    
    print("=" * 70)
    print("TEST 3: Understanding the prediction process")
    print("=" * 70)
    
    # Take a single test sample and explain the prediction
    test_sample = test_dataset.X[0]
    print(f"Test sample: {test_sample}")
    print(f"Actual value: {test_dataset.y[0]:.4f}")
    print()
    
    # Calculate distances manually
    distances = euclidean_distance(test_sample, train_dataset.X)
    k_nearest_indices = np.argsort(distances)[:3]
    
    print("3 Nearest neighbors:")
    print(f"{'Index':>6s} {'Distance':>10s} {'Value':>10s}")
    print("-" * 30)
    for idx in k_nearest_indices:
        print(f"{idx:6d} {distances[idx]:10.4f} {train_dataset.y[idx]:10.4f}")
    
    print()
    k_nearest_values = train_dataset.y[k_nearest_indices]
    predicted = np.mean(k_nearest_values)
    print(f"Average of neighbor values: {predicted:.4f}")
    print(f"Predicted value: {predicted:.4f}")
    print(f"Actual value: {test_dataset.y[0]:.4f}")
    print(f"Prediction error: {abs(predicted - test_dataset.y[0]):.4f}")
    print()
    
    print("=" * 70)
    print("TEST 4: Comparison - Classification vs Regression")
    print("=" * 70)
    print("KNN Classifier:")
    print("  - Predicts: Most common CLASS among k neighbors")
    print("  - Output: Discrete categories (e.g., 0, 1, 2)")
    print("  - Metric: Accuracy")
    print()
    print("KNN Regressor:")
    print("  - Predicts: AVERAGE VALUE of k neighbors")
    print("  - Output: Continuous values (e.g., 2.5, 3.7, 10.2)")
    print("  - Metric: RMSE")
    print()
    print("✓ KNN Regressor implementation complete!")