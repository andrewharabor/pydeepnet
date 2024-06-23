
from pymlp.layers import *
from pymlp.typing import *


class Optimizer(ABC):
    @abstractmethod
    def initialize(self, layers: list[DenseLayer]) -> None:
        pass

    @abstractmethod
    def update_parameters(self, layers: list[DenseLayer]) -> None:
        pass


class GradientDescent(Optimizer):
    learning_rate: Float64

    def __init__(self, learning_rate: Float64) -> None:
        self.learning_rate: Float64 = learning_rate

    def initialize(self, layers: list[DenseLayer]) -> None:
        return

    def update_parameters(self, layers: list[DenseLayer]) -> None:
        for layer in layers:
            layer.weights -= self.learning_rate * layer.weights_gradient
            layer.biases -= self.learning_rate * layer.biases_gradient


class Adam(Optimizer):
    learning_rate: Float64
    decay_rate1: Float64
    decay_rate2: Float64
    weights_momentums: list[NDArray]
    biases_momentums: list[NDArray]
    weights_velocities: list[NDArray]
    biases_velocities: list[NDArray]
    error: Float64 = Float64(1e-9)

    def __init__(self, learning_rate: Float64, decay_rate1: Float64, decay_rate2: Float64) -> None:
        self.learning_rate: Float64 = learning_rate
        self.decay_rate1: Float64 = decay_rate1
        self.decay_rate2: Float64 = decay_rate2
        self.weights_momentums: list[NDArray] = []
        self.biases_momentums: list[NDArray] = []
        self.weights_velocities: list[NDArray] = []
        self.biases_velocities: list[NDArray] = []

    def initialize(self, layers: list[DenseLayer]) -> None:
        # Ensure momentum and velocity arrays are empty before appending
        self.weights_momentums = []
        self.biases_momentums = []
        self.weights_velocities = []
        self.biases_velocities = []
        for layer in layers:
            self.weights_momentums.append(np.zeros(layer.weights.shape))
            self.biases_momentums.append(np.zeros(layer.biases.shape))
            self.weights_velocities.append(np.zeros(layer.weights.shape))
            self.biases_velocities.append(np.zeros(layer.biases.shape))

    def update_parameters(self, layers: list[DenseLayer]) -> None:
        for i, layer in enumerate(layers):
            self.weights_momentums[i] = self.decay_rate1 * self.weights_momentums[i] + (1 - self.decay_rate1) * layer.weights_gradient
            self.biases_momentums[i] = self.decay_rate1 * self.biases_momentums[i] + (1 - self.decay_rate1) * layer.biases_gradient
            self.weights_velocities[i] = self.decay_rate2 * self.weights_velocities[i] + (1 - self.decay_rate2) * np.square(layer.weights_gradient)
            self.biases_velocities[i] = self.decay_rate2 * self.biases_velocities[i] + (1 - self.decay_rate2) * np.square(layer.biases_gradient)
            layer.weights -= self.learning_rate * (self.weights_momentums[i] / (1 - self.decay_rate1)) / (np.sqrt(self.weights_velocities[i] / (1 - self.decay_rate2)) + self.error)
            layer.biases -= self.learning_rate * (self.biases_momentums[i] / (1 - self.decay_rate1)) / (np.sqrt(self.biases_velocities[i] / (1 - self.decay_rate2)) + self.error)


class RMSProp(Optimizer):
    learning_rate: Float64
    decay_rate: Float64
    weights_history: list[NDArray]
    biases_history: list[NDArray]
    error: Float64 = Float64(1e-9)

    def __init__(self, learning_rate: Float64, decay_rate: Float64) -> None:
        self.learning_rate: Float64 = learning_rate
        self.decay_rate: Float64 = decay_rate
        self.weights_history: list[NDArray] = []
        self.biases_history: list[NDArray] = []

    def initialize(self, layers: list[DenseLayer]) -> None:
        # Ensure history arrays are empty before appending
        self.weights_history = []
        self.biases_history = []
        for layer in layers:
            self.weights_history.append(np.zeros(layer.weights.shape))
            self.biases_history.append(np.zeros(layer.biases.shape))

    def update_parameters(self, layers: list[DenseLayer]) -> None:
        for i, layer in enumerate(layers):
            self.weights_history[i] = self.decay_rate * self.weights_history[i] + (1 - self.decay_rate) * np.square(layer.weights_gradient)
            self.biases_history[i] = self.decay_rate * self.biases_history[i] + (1 - self.decay_rate) * np.square(layer.biases_gradient)
            layer.weights -= self.learning_rate * layer.weights_gradient / (np.sqrt(self.weights_history[i]) + self.error)
            layer.biases -= self.learning_rate * layer.biases_gradient / (np.sqrt(self.biases_history[i]) + self.error)


class Momentum(Optimizer):
    learning_rate: Float64
    momentum: Float64
    weights_velocities: list[NDArray]
    biases_velocities: list[NDArray]

    def __init__(self, learning_rate: Float64, momentum: Float64) -> None:
        self.learning_rate: Float64 = learning_rate
        self.momentum: Float64 = momentum
        self.weights_velocities: list[NDArray] = []
        self.biases_velocities: list[NDArray] = []

    def initialize(self, layers: list[DenseLayer]) -> None:
        # Ensure velocity arrays are empty before appending
        self.weights_velocities = []
        self.biases_velocities = []
        for layer in layers:
            self.weights_velocities.append(np.zeros(layer.weights.shape))
            self.biases_velocities.append(np.zeros(layer.biases.shape))

    def update_parameters(self, layers: list[DenseLayer]) -> None:
        for i, layer in enumerate(layers):
            self.weights_velocities[i] = self.momentum * self.weights_velocities[i] + self.learning_rate * layer.weights_gradient
            self.biases_velocities[i] = self.momentum * self.biases_velocities[i] + self.learning_rate * layer.biases_gradient
            layer.weights -= self.weights_velocities[i]
            layer.biases -= self.biases_velocities[i]
