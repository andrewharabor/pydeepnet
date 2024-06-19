
from pymlp.typing import *


class ActivationFunc(ABC):
    @abstractmethod
    def compute(self, vector: NDArray) -> NDArray:
        pass

    @abstractmethod
    def derivative(self, vector: NDArray) -> NDArray:
        pass


class ReLU(ActivationFunc):
    leak: Float64  # for leaky ReLU

    def __init__(self, leak: Float64 = Float64(0.0)):
        self.leak = leak

    def compute(self, vector: NDArray) -> NDArray:
        return np.maximum(vector, self.leak * vector)

    def derivative(self, vector: NDArray) -> NDArray:
        return np.diag(np.where(vector > 0, 1, self.leak))


class Softmax(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        exp: NDArray = np.exp(vector - np.max(vector))
        return exp / np.sum(exp)

    def derivative(self, vector: NDArray) -> NDArray:
        softmax: NDArray = self.compute(vector).reshape(-1, 1)
        return np.diagflat(softmax) - np.dot(softmax, softmax.T)  # Jacobian matrix


class Sigmoid(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-vector))

    def derivative(self, vector: NDArray) -> NDArray:
        sigmoid: NDArray = self.compute(vector)
        return np.diag(sigmoid * (1 - sigmoid))


class Tanh(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        return np.tanh(vector)

    def derivative(self, vector: NDArray) -> NDArray:
        return np.diag(1 - np.tanh(vector) ** 2)


class SiLU(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        return vector / (1 + np.exp(-vector))

    def derivative(self, vector: NDArray) -> NDArray:
        sigmoid: NDArray = 1 / (1 + np.exp(-vector))
        return np.diag(sigmoid + (vector * sigmoid * (1 - sigmoid)))


class Softplus(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        return np.log(1 + np.exp(vector))

    def derivative(self, vector: NDArray) -> NDArray:
        return np.diag(1 / (1 + np.exp(-vector)))


class BinaryStep(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        return np.where(vector >= 0, 1, 0)

    def derivative(self, vector: NDArray) -> NDArray:
        return np.diag(np.zeros_like(vector))


class Linear(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        return vector

    def derivative(self, vector: NDArray) -> NDArray:
        return np.diag(np.ones_like(vector))
