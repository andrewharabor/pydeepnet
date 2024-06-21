
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


class InputLayer(Layer):
    nodes: int
    normalizer: Normalizer

    def __init__(self, nodes: int, normalizer: Normalizer) -> None:
        self.nodes = nodes
        self.normalizer = normalizer

    def forward_propagation(self, inputs: NDArray) -> NDArray:
        return self.normalizer.transform(inputs)

    def back_propagation(self, prev_derivative: NDArray) -> NDArray:
        return np.zeros(self.nodes)  # No back propagation for input layer


class DenseLayer(Layer):
    nodes: int
    activation_func: ActivationFunc
    regularizer: Regularizer
    weights: NDArray
    biases: NDArray
    inputs: NDArray
    weighted_inputs: NDArray
    weights_gradient: NDArray
    biases_gradient: NDArray

    def __init__(self, nodes: int, activation_func: ActivationFunc, regularizer: Regularizer) -> None:
        self.nodes = nodes
        self.activation_func = activation_func
        self.regularizer = regularizer

    def initialize_weights(self, prev_nodes: int) -> None:
        self.weights = np.random.normal(size=(self.nodes, prev_nodes))
        self.biases = np.random.normal(size=self.nodes)

    def initialize_gradients(self) -> None:
        self.weights_gradient = np.zeros(self.weights.shape)
        self.biases_gradient = np.zeros(self.biases.shape)

    def forward_propagation(self, inputs: NDArray) -> NDArray:
        self.inputs = inputs
        self.weighted_inputs = self.weights @ inputs + self.biases
        return self.activation_func.compute(self.weighted_inputs)

    def back_propagation(self, prev_derivative: NDArray) -> NDArray:
        # Derivative of activation function w.r.t weighted inputs (times previous derivative due to chain rule)
        activation_derivative: NDArray = prev_derivative @ self.activation_func.derivative(self.weighted_inputs)
        # Derivative of activation w.r.t. weights and biases
        self.weights_gradient += np.outer(activation_derivative, self.inputs) + self.regularizer.derivative(self.weights)
        self.biases_gradient += activation_derivative
        return activation_derivative @ self.weights  # Derivative of activation w.r.t. inputs


class DenseOutputLayer(DenseLayer):
    cost_func: CostFunc

    def __init__(self, nodes: int, activation_func: ActivationFunc, regularizer: Regularizer, cost_func: CostFunc) -> None:
        super().__init__(nodes, activation_func, regularizer)
        self.cost_func = cost_func

    def start_back_propagation(self, targets: NDArray) -> NDArray:
        # Compute derivative of cost function w.r.t. activation
        activation: NDArray = self.activation_func.compute(self.weighted_inputs)
        cost_derivative: NDArray = self.cost_func.derivative(activation, targets)
        return self.back_propagation(cost_derivative)
