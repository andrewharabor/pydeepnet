
from pymlp.typing import *


class Regularizer(ABC):
    @abstractmethod
    def compute(self, weights: NDArray) -> Float64:
        pass

    @abstractmethod
    def derivative(self, weights: NDArray) -> NDArray:
        pass


class L2Norm(Regularizer):
    penalty: Float64

    def __init__(self, penalty: Float64) -> None:
        self.penalty: Float64 = penalty

    def compute(self, weights: NDArray) -> Float64:
        return self.penalty / 2 * np.sum(weights ** 2)

    def derivative(self, weights: NDArray) -> NDArray:
        return self.penalty * weights


class L1Norm(Regularizer):
    penalty: Float64

    def __init__(self, penalty: Float64) -> None:
        self.penalty: Float64 = penalty

    def compute(self, weights: NDArray) -> Float64:
        return self.penalty * np.sum(np.abs(weights))

    def derivative(self, weights: NDArray) -> NDArray:
        return self.penalty * np.sign(weights)


class ElasticNet(Regularizer):
    l1_regularizer: L1Norm
    l2_regularizer: L2Norm

    def __init__(self, l1_penalty: Float64, l2_penalty: Float64) -> None:
        self.l1_regularizer: L1Norm = L1Norm(l1_penalty)
        self.l2_regularizer: L2Norm = L2Norm(l2_penalty)

    def compute(self, weights: NDArray) -> Float64:
        return self.l1_regularizer.compute(weights) + self.l2_regularizer.compute(weights)

    def derivative(self, weights: NDArray) -> NDArray:
        return self.l1_regularizer.derivative(weights) + self.l2_regularizer.derivative(weights)


class NoReg(Regularizer):
    def compute(self, weights: NDArray) -> Float64:
        return Float64(0.0)

    def derivative(self, weights: NDArray) -> NDArray:
        return np.zeros(weights.shape)
