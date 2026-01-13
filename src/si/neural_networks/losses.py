from abc import abstractmethod

import numpy as np


class LossFunction:

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the derivative of the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean squared error loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true.
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Cross entropy loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p) + (1 - y_true) / (1 - p)

import numpy as np
from si.neural_networks.losses import LossFunction


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical cross-entropy loss function.
    
    The categorical cross-entropy loss function is applied to multi-class
    classification problems. It measures the dissimilarity between predicted
    class probabilities and true one-hot encoded class labels.
    
    This loss function is typically used with softmax activation in the output
    layer for multi-class classification.
    
    Formula: L = -sum(y_true * log(y_pred))
    
    where:
    - y_true is the one-hot encoded true labels
    - y_pred is the predicted probability distribution (from softmax)
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the categorical cross-entropy loss function.
        
        Parameters
        ----------
        y_true : np.ndarray
            The true labels (one-hot encoded). Shape: (n_samples, n_classes)
        y_pred : np.ndarray
            The predicted probabilities. Shape: (n_samples, n_classes)
        
        Returns
        -------
        float
            The loss value
        """
        # Clip predictions to avoid log(0) and numerical instability
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Compute categorical cross-entropy loss
        # Only the probability of the true class contributes to the loss
        loss = -np.sum(y_true * np.log(y_pred_clipped))
        
        return loss

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the categorical cross-entropy loss function.
        
        Parameters
        ----------
        y_true : np.ndarray
            The true labels (one-hot encoded). Shape: (n_samples, n_classes)
        y_pred : np.ndarray
            The predicted probabilities. Shape: (n_samples, n_classes)
        
        Returns
        -------
        np.ndarray
            The derivative of the loss function. Shape: (n_samples, n_classes)
        """
        # Clip predictions to avoid division by zero
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Compute the derivative: -y_true / y_pred
        return -y_true / y_pred_clipped


