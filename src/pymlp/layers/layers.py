
from abc import ABC, abstractmethod

from pymlp.activation_funcs import *
from pymlp.cost_funcs import *
from pymlp.normalizers import *
from pymlp.regularizers import *
from pymlp.typing import *


class Layer(ABC):
    @abstractmethod
    def forward_propagation(self, inputs: NDArray) -> NDArray:
        pass

    @abstractmethod
    def back_propagation(self, prev_derivative: NDArray) -> NDArray:
        pass

    def _assert_shape(self, inputs: NDArray) -> None:
        if inputs.shape[0] == 0:
            raise ValueError("Inputs array is empty")

    def _assert_positive(self, nodes: Int64) -> None:
        if nodes <= 0:
            raise ValueError("Number of nodes is non-positive")


class InputLayer(Layer):
    nodes: Int64
    normalizer: Normalizer | None

    def __init__(self, nodes: Int64, normalizer: Normalizer | None) -> None:
        self._assert_positive(nodes)
        self.nodes = nodes
        self.normalizer = normalizer

    def forward_propagation(self, inputs: NDArray) -> NDArray:
        self._assert_shape(inputs)
        if self.normalizer is None:
            return inputs
        return self.normalizer.transform(inputs)

    def back_propagation(self, prev_derivative: NDArray) -> NDArray:
        return np.zeros(self.nodes)  # No back propagation for input layer


class DenseLayer(Layer):
    nodes: Int64
    activation_func: ActivationFunc
    regularizer: Regularizer | None

    weights: NDArray
    biases: NDArray
    inputs: NDArray
    weighted_inputs: NDArray
    weights_gradient: NDArray
    biases_gradient: NDArray

    _parameters_initialized: bool = False
    _gradients_initialized: bool = False
    _inputs_stored: bool = False

    def __init__(self, nodes: Int64, activation_func: ActivationFunc, regularizer: Regularizer | None) -> None:
        self._assert_positive(nodes)
        self.nodes = nodes
        self.activation_func = activation_func
        self.regularizer = regularizer

    def initialize_parameters(self, prev_nodes: Int64) -> None:
        self._assert_positive(prev_nodes)
        self._parameters_initialized = True
        self.weights = np.random.normal(size=(self.nodes, prev_nodes))
        self.biases = np.random.normal(size=self.nodes)

    def initialize_gradients(self) -> None:
        self._assert_parameters_initialized()
        self._gradients_initialized = True
        self.weights_gradient = np.zeros(self.weights.shape)
        self.biases_gradient = np.zeros(self.biases.shape)

    def forward_propagation(self, inputs: NDArray) -> NDArray:
        self._assert_parameters_initialized()
        self._inputs_stored = True
        self.inputs = inputs
        self.weighted_inputs = self.weights @ inputs + self.biases
        return self.activation_func.compute(self.weighted_inputs)

    def back_propagation(self, prev_derivative: NDArray) -> NDArray:
        self._assert_parameters_initialized()
        self._assert_gradients_initialized()
        self._assert_inputs_stored()
        # Derivative of activation function w.r.t weighted inputs (times previous derivative due to chain rule)
        activation_derivative: NDArray = prev_derivative @ self.activation_func.derivative(self.weighted_inputs)
        # Derivative of activation w.r.t. weights and biases
        self.weights_gradient += np.outer(activation_derivative, self.inputs)
        if self.regularizer is not None:
            self.weights_gradient += self.regularizer.derivative(self.weights)
        self.biases_gradient += activation_derivative
        return activation_derivative @ self.weights  # Derivative of activation w.r.t. inputs

    def _assert_parameters_initialized(self) -> None:
        if not self._parameters_initialized:
            raise ValueError("Parameters have not been initialized")

    def _assert_gradients_initialized(self) -> None:
        if not self._gradients_initialized:
            raise ValueError("Gradients have not been initialized")

    def _assert_inputs_stored(self) -> None:
        if not self._inputs_stored:
            raise ValueError("Inputs have not been stored through forward propagation")


class DenseOutputLayer(DenseLayer):
    cost_func: CostFunc

    def __init__(self, nodes: Int64, activation_func: ActivationFunc, regularizer: Regularizer | None, cost_func: CostFunc) -> None:
        self._assert_positive(nodes)
        super().__init__(nodes, activation_func, regularizer)
        self.cost_func = cost_func

    def start_back_propagation(self, targets: NDArray) -> NDArray:
        self._assert_parameters_initialized()
        self._assert_gradients_initialized()
        self._assert_inputs_stored()
        # Compute derivative of cost function w.r.t. activation
        activation: NDArray = self.activation_func.compute(self.weighted_inputs)
        cost_derivative: NDArray = self.cost_func.derivative(activation, targets)
        return self.back_propagation(cost_derivative)
