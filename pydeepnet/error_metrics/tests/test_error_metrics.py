
from pydeepnet._typing import *
from pydeepnet.error_metrics import *


class TestMeanAbsolutePercentageError():
    mean_absolute_percentage_error: MeanAbsolutePercentageError = MeanAbsolutePercentageError()

    def test_compute(self) -> None:
        test_cases: list[tuple[tuple[NDArray, NDArray], Float64]] = [
            ((np.array([[0], [10]]), np.array([[0.2], [8.7]])), Float64(57.47126770019531)),
            ((np.array([[-5.0, 1.0], [0.1, 3.1415]]), np.array([[-4.3, 0.9], [0.0985, 3.125]])), Float64(7.360256195068359)),
            ((np.array([[0.1, 0.2, 0.3], [-4.0, 0.5, 6.7]]), np.array([[-11.0, 0.75, 13.0], [-4.8, 0.6, 6.0]])), Float64(52.822452545166016)),
            ((np.array([[10.0, 214.0, 9.0, 0.0], [13.0, 42.0, -0.49, -8.1428], [0.96, 0.75, 0.13, 0.16]]), np.array([[-10.0, 19.0, 89.9, -0.43], [13.0, 42.0, -0.49, -8.1428], [0.84, 0.6, 0.12, -0.24]])), Float64(135.8825225830078)),
            ((np.array([[0.1, 0.05, 0.75, 0.05, 0.15], [1.0, -1.0, 0.0, 0.001, -0.001]]), np.array([[0, 0, 1, 0, 0], [2, -2, 25, 0, 0]])), Float64(3520000024.0)),
        ]
        for (predictions, targets), expected in test_cases:
            assert np.isclose(self.mean_absolute_percentage_error.compute(predictions, targets), expected)
        with pytest.raises(ValueError):
            self.mean_absolute_percentage_error.compute(np.array([]), np.array([[0.2, 1], [3, 4.5]]))
            self.mean_absolute_percentage_error.compute(np.array([11, 0.1]), np.array([]))
            self.mean_absolute_percentage_error.compute(np.array([]), np.array([]))
            self.mean_absolute_percentage_error.compute(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2], [3, 4], [5, 6]]))
            self.mean_absolute_percentage_error.compute(np.array([1, 2]), np.array([1, 29, 7.3, 4]))


class TestPercentageAccuracy():
    percentage_accuracy: PercentageAccuracy = PercentageAccuracy()

    def test_compute(self) -> None:
        test_cases: list[tuple[tuple[NDArray, NDArray], Float64]] = [
            ((np.array([[1]]), np.array([[1]])), Float64(100.0)),
            ((np.array([[0.5, 0.5], [0.1, 0.9], [0.001, 0.999]]), np.array([[1, 0], [0, 1], [1, 0]])), Float64(66.66666667)),
            ((np.array([[0.01, 0.54, 0.37, 0.08], [0.1, 0.2, 0.3, 0.4]]), np.array([[0, 0, 1, 0], [0, 0, 0, 1]])), Float64(50.0)),
            ((np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]), np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])), Float64(0.0)),
            ((np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.3184, 0.1292, 0.5524], [0.05, 0.95, 0.0], [0.1, 0.8, 0.1]]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])), Float64(60.0)),
        ]
        for (predictions, targets), expected in test_cases:
            assert np.isclose(self.percentage_accuracy.compute(predictions, targets), expected)
        with pytest.raises(ValueError):
            self.percentage_accuracy.compute(np.array([]), np.array([[1, 0], [0, 1]]))
            self.percentage_accuracy.compute(np.array([[0.1, 0.9]]), np.array([]))
            self.percentage_accuracy.compute(np.array([]), np.array([]))
            self.percentage_accuracy.compute(np.array([[0.8, 0.2], [0.35, 0.65]]), np.array([[1, 0], [0, 1], [1, 0]]))
            self.percentage_accuracy.compute(np.array([[0.8, 0.2]]), np.array([[1, 0, 0, 0]]))
            self.percentage_accuracy.compute(np.array([[-0.5, 2], [0.1, 0.9]]), np.array([[0, 1], [1, 0]]))
            self.percentage_accuracy.compute(np.array([[0.5, 0.5], [0.5, 0.8]]), np.array([[0, 1], [0, 1]]))
            self.percentage_accuracy.compute(np.array([[0.6, 0.4], [0.75, 0.25]]), np.array([[0, 10], [0, 1]]))
            self.percentage_accuracy.compute(np.array([[0.1, 0.9], [0.7, 0.3]]), np.array([[0, 0], [0, 0]]))
