
from pydeepnet._typing import *


class Regularizer(ABC):
    @abstractmethod
    def compute(self, weights: NDArray) -> Float64:
        pass

    @abstractmethod
    def derivative(self, weights: NDArray) -> NDArray:
        pass

    def _assert_shape(self, weights: NDArray) -> None:
        if weights.shape[0] == 0:
            raise ValueError("Weights array is empty")

    def _assert_non_negative(self, penalty: Float64) -> None:
        if penalty < 0:
            raise ValueError("Penalty is negative")


class Lasso(Regularizer):
    penalty: Float64

    def __init__(self, penalty: Float64) -> None:
        self._assert_non_negative(penalty)
        self.penalty = penalty

    def compute(self, weights: NDArray) -> Float64:
        self._assert_shape(weights)
        return self.penalty * np.sum(np.abs(weights))

    def derivative(self, weights: NDArray) -> NDArray:
        self._assert_shape(weights)
        return self.penalty * np.sign(weights)


class Ridge(Regularizer):
    penalty: Float64

    def __init__(self, penalty: Float64) -> None:
        self._assert_non_negative(penalty)
        self.penalty = penalty

    def compute(self, weights: NDArray) -> Float64:
        self._assert_shape(weights)
        return self.penalty / 2 * np.sum(weights ** 2)

    def derivative(self, weights: NDArray) -> NDArray:
        self._assert_shape(weights)
        return self.penalty * weights


class ElasticNet(Regularizer):
    lasso: Lasso
    ridge: Ridge

    def __init__(self, lasso_penalty: Float64, ridge_penalty: Float64) -> None:
        self._assert_non_negative(lasso_penalty)
        self._assert_non_negative(ridge_penalty)
        self.lasso = Lasso(lasso_penalty)
        self.ridge = Ridge(ridge_penalty)

    def compute(self, weights: NDArray) -> Float64:
        self._assert_shape(weights)
        return self.lasso.compute(weights) + self.ridge.compute(weights)

    def derivative(self, weights: NDArray) -> NDArray:
        self._assert_shape(weights)
        return self.lasso.derivative(weights) + self.ridge.derivative(weights)
