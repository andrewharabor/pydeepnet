
from pymlp.typing import *


class ActivationFunc(ABC):
    @abstractmethod
    def compute(self, vector: NDArray) -> NDArray:
        pass

    @abstractmethod
    def derivative(self, vector: NDArray) -> NDArray:
        pass

    @staticmethod
    def _assert_shape(vector: NDArray) -> None:
        if vector.shape[0] == 0:
            raise ValueError("`vector` array is empty")


class BinaryStep(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        BinaryStep._assert_shape(vector)
        return np.where(vector >= 0, 1, 0)

    def derivative(self, vector: NDArray) -> NDArray:
        BinaryStep._assert_shape(vector)
        return np.diag(np.zeros_like(vector))


class Linear(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        Linear._assert_shape(vector)
        return vector

    def derivative(self, vector: NDArray) -> NDArray:
        Linear._assert_shape(vector)
        return np.diag(np.ones_like(vector))


class ReLU(ActivationFunc):
    leak: Float64  # for leaky ReLU

    def __init__(self, leak: Float64 = Float64(0.0)) -> None:
        self.leak = leak

    def compute(self, vector: NDArray) -> NDArray:
        ReLU._assert_shape(vector)
        return np.maximum(vector, self.leak * vector)

    def derivative(self, vector: NDArray) -> NDArray:
        ReLU._assert_shape(vector)
        return np.diag(np.where(vector > 0, 1, self.leak))


class Sigmoid(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        Sigmoid._assert_shape(vector)
        return 1 / (1 + np.exp(-vector))

    def derivative(self, vector: NDArray) -> NDArray:
        Sigmoid._assert_shape(vector)
        sigmoid: NDArray = self.compute(vector)
        return np.diag(sigmoid * (1 - sigmoid))


class SiLU(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        SiLU._assert_shape(vector)
        return vector / (1 + np.exp(-vector))

    def derivative(self, vector: NDArray) -> NDArray:
        SiLU._assert_shape(vector)
        sigmoid: NDArray = 1 / (1 + np.exp(-vector))
        return np.diag(sigmoid + (vector * sigmoid * (1 - sigmoid)))


class Softmax(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        Softmax._assert_shape(vector)
        exp: NDArray = np.exp(vector - np.max(vector))
        return exp / np.sum(exp)

    def derivative(self, vector: NDArray) -> NDArray:
        Softmax._assert_shape(vector)
        softmax: NDArray = self.compute(vector)
        return np.diag(softmax) - np.outer(softmax, softmax)


class Softplus(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        Softplus._assert_shape(vector)
        return np.log(1 + np.exp(vector))

    def derivative(self, vector: NDArray) -> NDArray:
        Softplus._assert_shape(vector)
        return np.diag(1 / (1 + np.exp(-vector)))


class Tanh(ActivationFunc):
    def compute(self, vector: NDArray) -> NDArray:
        Tanh._assert_shape(vector)
        return np.tanh(vector)

    def derivative(self, vector: NDArray) -> NDArray:
        Tanh._assert_shape(vector)
        return np.diag(1 - np.tanh(vector) ** 2)
