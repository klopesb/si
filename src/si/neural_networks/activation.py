from abc import abstractmethod
from typing import Union

import numpy as np

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.
    """

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0
    
class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input))


class ReLUActivation(ActivationLayer):
    """
    ReLU activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input >= 0, 1, 0)

class TanhActivation(ActivationLayer):
    """
    Tanh (Hyperbolic Tangent) activation function.
    
    The tanh activation layer applies the hyperbolic tangent function to the output
    of neurons, squashing the values to the range of -1 to 1.
    
    Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Derivative: tanh'(x) = 1 - tanh(x)^2
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Tanh activation function.
        
        Applies the hyperbolic tangent function, which squashes input values
        to the range [-1, 1].

        Parameters
        ----------
        input : np.ndarray
            The input to the layer

        Returns
        -------
        np.ndarray
            The output of the layer, with values in range [-1, 1]
        """
        return np.tanh(input)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Derivative of the tanh activation function.
        
        The derivative of tanh(x) is 1 - tanh(x)^2

        Parameters
        ----------
        input : np.ndarray
            The input to the layer

        Returns
        -------
        np.ndarray
            The derivative of the activation function
        """
        tanh_output = self.activation_function(input)
        return 1 - np.power(tanh_output, 2)
class SoftmaxActivation(ActivationLayer):
    """
    Softmax activation function.
    
    The softmax activation layer transforms the raw output scores into a probability
    distribution (that sums to 1), making it suitable for multi-class classification
    problems.
    
    Formula: softmax(x_i) = e^(x_i) / sum(e^(x_j)) for all j
    
    The output is a probability distribution where all values are in [0, 1] and sum to 1.
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Softmax activation function.
        
        Transforms the input into a probability distribution. Each element represents
        the probability of that class, and all elements sum to 1.
        
        Uses the stable softmax implementation: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        This prevents numerical overflow.

        Parameters
        ----------
        input : np.ndarray
            The input to the layer (logits/scores)

        Returns
        -------
        np.ndarray
            The output probability distribution, with values in [0, 1] that sum to 1
        """
        # Subtract max for numerical stability (prevents overflow)
        exp_values = np.exp(input - np.max(input, axis=-1, keepdims=True))
        # Normalize to get probabilities
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Derivative of the softmax activation function.
        
        The derivative of softmax is more complex:
        - If i == j: softmax(x_i) * (1 - softmax(x_i))
        - If i != j: -softmax(x_i) * softmax(x_j)
        
        For simplicity in backpropagation with cross-entropy loss,
        we return the softmax output itself, as the full Jacobian
        is typically combined with the loss function derivative.

        Parameters
        ----------
        input : np.ndarray
            The input to the layer

        Returns
        -------
        np.ndarray
            The derivative approximation (softmax output)
        """
        softmax_output = self.activation_function(input)
        return softmax_output * (1 - softmax_output)