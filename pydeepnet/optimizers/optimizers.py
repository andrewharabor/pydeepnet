
from pydeepnet._typing import *
from pydeepnet.layers import *


class Optimizer(ABC):
    _initialized: bool = False

    @abstractmethod
    def initialize(self, layers: list[DenseLayer]) -> None:
        pass

    @abstractmethod
    def update_parameters(self, layers: list[DenseLayer]) -> None:
        pass

    def _assert_initialized(self) -> None:
        if not self._initialized:
            raise Exception("Optimizer not initialized")

    def _assert_positive(self, learning_rate: Float64) -> None:
        if learning_rate <= 0:
            raise ValueError("Learning rate is non-positive")

    def _assert_fractional(self, decay_rate: Float64) -> None:
        if decay_rate <= 0 or decay_rate >= 1:
            raise ValueError("Decay rate is not between 0 and 1")


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
        self._assert_positive(learning_rate)
        self._assert_fractional(decay_rate1)
        self._assert_fractional(decay_rate2)
        self.learning_rate: Float64 = learning_rate
        self.decay_rate1 = decay_rate1
        self.decay_rate2 = decay_rate2

    def initialize(self, layers: list[DenseLayer]) -> None:
        self._initialized = True
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
        self._assert_initialized()
        for i, layer in enumerate(layers):
            self.weights_momentums[i] = self.decay_rate1 * self.weights_momentums[i] + (1 - self.decay_rate1) * layer.weights_gradient
            self.biases_momentums[i] = self.decay_rate1 * self.biases_momentums[i] + (1 - self.decay_rate1) * layer.biases_gradient
            self.weights_velocities[i] = self.decay_rate2 * self.weights_velocities[i] + (1 - self.decay_rate2) * np.square(layer.weights_gradient)
            self.biases_velocities[i] = self.decay_rate2 * self.biases_velocities[i] + (1 - self.decay_rate2) * np.square(layer.biases_gradient)
            layer.weights -= self.learning_rate * (self.weights_momentums[i] / (1 - self.decay_rate1)) / (np.sqrt(self.weights_velocities[i] / (1 - self.decay_rate2)) + self.error)
            layer.biases -= self.learning_rate * (self.biases_momentums[i] / (1 - self.decay_rate1)) / (np.sqrt(self.biases_velocities[i] / (1 - self.decay_rate2)) + self.error)


class GradientDescent(Optimizer):
    learning_rate: Float64

    def __init__(self, learning_rate: Float64) -> None:
        self._assert_positive(learning_rate)
        self.learning_rate = learning_rate

    def initialize(self, layers: list[DenseLayer]) -> None:
        self._initialized = True
        return

    def update_parameters(self, layers: list[DenseLayer]) -> None:
        self._assert_initialized()
        for layer in layers:
            layer.weights -= self.learning_rate * layer.weights_gradient
            layer.biases -= self.learning_rate * layer.biases_gradient


class Momentum(Optimizer):
    learning_rate: Float64
    decay_rate: Float64
    weights_velocities: list[NDArray]
    biases_velocities: list[NDArray]

    def __init__(self, learning_rate: Float64, decay_rate: Float64) -> None:
        self._assert_positive(learning_rate)
        self._assert_fractional(decay_rate)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

    def initialize(self, layers: list[DenseLayer]) -> None:
        self._initialized = True
        self.weights_velocities = []
        self.biases_velocities = []
        for layer in layers:
            self.weights_velocities.append(np.zeros(layer.weights.shape))
            self.biases_velocities.append(np.zeros(layer.biases.shape))

    def update_parameters(self, layers: list[DenseLayer]) -> None:
        self._assert_initialized()
        for i, layer in enumerate(layers):
            self.weights_velocities[i] = self.decay_rate * self.weights_velocities[i] + self.learning_rate * layer.weights_gradient
            self.biases_velocities[i] = self.decay_rate * self.biases_velocities[i] + self.learning_rate * layer.biases_gradient
            layer.weights -= self.weights_velocities[i]
            layer.biases -= self.biases_velocities[i]


class RMSProp(Optimizer):
    learning_rate: Float64
    decay_rate: Float64
    weights_history: list[NDArray]
    biases_history: list[NDArray]
    error: Float64 = Float64(1e-9)

    def __init__(self, learning_rate: Float64, decay_rate: Float64) -> None:
        self._assert_positive(learning_rate)
        self._assert_fractional(decay_rate)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

    def initialize(self, layers: list[DenseLayer]) -> None:
        self._initialized = True
        self.weights_history = []
        self.biases_history = []
        for layer in layers:
            self.weights_history.append(np.zeros(layer.weights.shape))
            self.biases_history.append(np.zeros(layer.biases.shape))

    def update_parameters(self, layers: list[DenseLayer]) -> None:
        self._assert_initialized()
        for i, layer in enumerate(layers):
            self.weights_history[i] = self.decay_rate * self.weights_history[i] + (1 - self.decay_rate) * np.square(layer.weights_gradient)
            self.biases_history[i] = self.decay_rate * self.biases_history[i] + (1 - self.decay_rate) * np.square(layer.biases_gradient)
            layer.weights -= self.learning_rate * layer.weights_gradient / (np.sqrt(self.weights_history[i]) + self.error)
            layer.biases -= self.learning_rate * layer.biases_gradient / (np.sqrt(self.biases_history[i]) + self.error)
