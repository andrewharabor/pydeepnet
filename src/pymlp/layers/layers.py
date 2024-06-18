
from pymlp.activation_funcs import *
from pymlp.cost_funcs import *
from pymlp.typing import *


class Layer(ABC):
    @abstractmethod
    def forward_propagation(self, inputs: NDArray) -> NDArray:
        pass

    @abstractmethod
    def back_propagation(self, prev_derivative: NDArray) -> NDArray:
        pass


class InputLayer(Layer):
    size: int

    def __init__(self, size: int):
        self.size = size

    def forward_propagation(self, inputs: NDArray) -> NDArray:
        return inputs

    def back_propagation(self, prev_derivative: NDArray) -> NDArray:
        return np.zeros(self.size)  # No back propagation for input layer


class DenseHiddenLayer(Layer):
    nodes: int
    activation_func: ActivationFunc
    weights: NDArray
    biases: NDArray
    inputs: NDArray
    weighted_inputs: NDArray
    weights_gradient: NDArray
    biases_gradient: NDArray

    def __init__(self, nodes: int, activation_func: ActivationFunc):
        self.nodes = nodes
        self.activation_func = activation_func

    def initialize_weights(self, prev_nodes: int, to_zeros: bool = False) -> None:
        if to_zeros:
            self.weights = np.zeros((self.nodes, prev_nodes))
            self.biases = np.zeros(self.nodes)
        else:
            self.weights = np.random.randn(self.nodes, prev_nodes)
            self.biases = np.random.randn(self.nodes)

    def forward_propagation(self, inputs: NDArray) -> NDArray:
        self.inputs = inputs
        self.weighted_inputs = self.weights @ inputs + self.biases
        return self.activation_func.compute(self.weighted_inputs)

    def back_propagation(self, prev_derivative: NDArray) -> NDArray:
        # FIXME: average across all derivatives from next layer
        # Derivative of activation function w.r.t weighted inputs (times previous derivative due to chain rule)
        activation_derivative: NDArray = prev_derivative @ self.activation_func.derivative(self.weighted_inputs)
        # Derivative of activation w.r.t. weights and biases
        self.weights_gradient = np.outer(activation_derivative, self.inputs)
        self.biases_gradient = activation_derivative
        return activation_derivative @ self.weights  # Derivative of activation w.r.t. inputs


class DenseOutputLayer(DenseHiddenLayer):
    cost_func: CostFunc

    def __init__(self, nodes: int, activation_func: ActivationFunc, cost_func: CostFunc):
        super().__init__(nodes, activation_func)
        self.cost_func = cost_func

    def start_back_propagation(self, targets: NDArray) -> NDArray:
        # Compute derivative of cost function w.r.t. activation
        activation: NDArray = self.activation_func.compute(self.weighted_inputs)
        cost_derivative: NDArray = self.cost_func.derivative(activation, targets)
        return self.back_propagation(cost_derivative)
