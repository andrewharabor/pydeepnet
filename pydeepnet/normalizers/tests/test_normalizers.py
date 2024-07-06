
from pydeepnet._typing import *
from pydeepnet.normalizers import *


class TestDecimalScaling:
    decimal_scaling: DecimalScaling

    def test_fit(self):
        test_cases: list[tuple[NDArray, Float64]] = [
            (np.array([10]), Float64(1)),
            (np.array([-2]), Float64(1)),
            (np.array([-5, 51]), Float64(2)),
            (np.array([1, 2]), Float64(1)),
            (np.array([-0.5, 7.6]), Float64(1)),
            (np.array([2.718, 0.577, 3.141]), Float64(1)),
            (np.array([1, 2, 3, -4, 5, 60, -7, 8, 957, 10]), Float64(3)),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), Float64(1)),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), Float64(2)),
            (np.array([[0.0, 0.0], [0.0, 0.0]]), Float64(0)),
            (np.array([[0.1, 0.2, 0.3], [-400.0, 0.5, 6.7]]), Float64(3)),
        ]
        for inputs, expected in test_cases:
            self.decimal_scaling = DecimalScaling()
            self.decimal_scaling.fit(inputs)
            assert self.decimal_scaling.scale == expected
        decimal_scaling = DecimalScaling()
        with pytest.raises(ValueError):
            decimal_scaling.fit(np.array([]))

    def test_transform(self):
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([10]), np.array([1.0])),
            (np.array([-2]), np.array([-0.2])),
            (np.array([-5, 51]), np.array([-0.05, 0.51])),
            (np.array([1, 2]), np.array([0.1, 0.2])),
            (np.array([-0.5, 7.6]), np.array([-0.05, 0.76])),
            (np.array([2.718, 0.577, 3.141]), np.array([0.2718, 0.057699999999999994, 0.3141])),
            (np.array([1, 2, 3, -4, 5, 60, -7, 8, 957, 10]), np.array([0.001, 0.002, 0.003, -0.004, 0.005, 0.06, -0.007, 0.008, 0.957, 0.01])),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), np.array([[0.075, -0.9], [0.01, -0.00030000000000000003]])),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), np.array([[0.01, 0.12, 0.03], [0.44, 0.05, 0.15], [0.27, 0.008, -0.09]])),
            (np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 0.0]])),
            (np.array([[0.1, 0.2, 0.3], [-400.0, 0.5, 6.7]]), np.array([[0.0001, 0.0002, 0.0003], [-0.4, 0.0005, 0.0067]])),
        ]
        for inputs, expected in test_cases:
            self.decimal_scaling = DecimalScaling()
            self.decimal_scaling.fit(inputs)
            assert np.allclose(self.decimal_scaling.transform(inputs), expected)
        decimal_scaling = DecimalScaling()
        with pytest.raises(ValueError):
            decimal_scaling.transform(np.array([]))
        with pytest.raises(RuntimeError):
            decimal_scaling.transform(np.array([1.0]))

    def test_undo(self):
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([1.0]), np.array([10])),
            (np.array([-0.2]), np.array([-2])),
            (np.array([-0.05, 0.51]), np.array([-5, 51])),
            (np.array([0.1, 0.2]), np.array([1, 2])),
            (np.array([-0.05, 0.76]), np.array([-0.5, 7.6])),
            (np.array([0.2718, 0.057699999999999994, 0.3141]), np.array([2.718, 0.577, 3.141])),
            (np.array([0.001, 0.002, 0.003, -0.004, 0.005, 0.06, -0.007, 0.008, 0.957, 0.01]), np.array([1, 2, 3, -4, 5, 60, -7, 8, 957, 10])),
            (np.array([[0.075, -0.9], [0.01, -0.00030000000000000003]]), np.array([[0.75, -9.0], [0.1, -0.003]])),
            (np.array([[0.01, 0.12, 0.03], [0.44, 0.05, 0.15], [0.27, 0.008, -0.09]]), np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]])),
            (np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 0.0]])),
            (np.array([[0.0001, 0.0002, 0.0003], [-0.4, 0.0005, 0.0067]]), np.array([[0.1, 0.2, 0.3], [-400.0, 0.5, 6.7]])),
        ]
        for inputs, expected in test_cases:
            self.decimal_scaling = DecimalScaling()
            self.decimal_scaling.fit(expected)
            assert np.allclose(self.decimal_scaling.undo(inputs), expected)
        decimal_scaling = DecimalScaling()
        with pytest.raises(ValueError):
            decimal_scaling.undo(np.array([]))
        with pytest.raises(RuntimeError):
            decimal_scaling.undo(np.array([1.0]))


class TestMinMax:
    min_max: MinMax

    def test_fit(self):
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([[10]]), np.array([10])),
            (np.array([[-2]]), np.array([-2])),
            (np.array([[-5, 51]]), np.array([-5, 51])),
            (np.array([[1, 2]]), np.array([1, 2])),
            (np.array([[-0.5, 7.6]]), np.array([-0.5, 7.6])),
            (np.array([[2.718, 0.577, 3.141]]), np.array([2.718, 0.577, 3.141])),
            (np.array([[1, 2, 3, -4, 5, 60, -7, 8, 957, 10]]), np.array([1, 2, 3, -4, 5, 60, -7, 8, 957, 10])),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), np.array([0.1, -9.0])),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), np.array([1.0, 0.8, -9.0])),
            (np.array([[0, 0], [0, 0]]), np.array([0, 0])),
            (np.array([[0.1, 0.2, 0.3], [-400.0, 0.5, 6.7]]), np.array([-400.0, 0.2, 0.3])),
        ]
        for inputs, expected in test_cases:
            self.min_max = MinMax()
            self.min_max.fit(inputs)
            assert np.allclose(self.min_max.min, expected)
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([[10]]), np.array([10])),
            (np.array([[-2]]), np.array([-2])),
            (np.array([[-5, 51]]), np.array([-5, 51])),
            (np.array([[1, 2]]), np.array([1, 2])),
            (np.array([[-0.5, 7.6]]), np.array([-0.5, 7.6])),
            (np.array([[2.718, 0.577, 3.141]]), np.array([2.718, 0.577, 3.141])),
            (np.array([[1, 2, 3, -4, 5, 60, -7, 8, 957, 10]]), np.array([1, 2, 3, -4, 5, 60, -7, 8, 957, 10])),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), np.array([0.75, -0.003])),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), np.array([44, 12, 15])),
            (np.array([[0, 0], [0, 0]]), np.array([0, 0])),
            (np.array([[0.1, 0.2, 0.3], [-400.0, 0.5, 6.7]]), np.array([0.1, 0.5, 6.7])),
        ]
        for inputs, expected in test_cases:
            self.min_max = MinMax()
            self.min_max.fit(inputs)
            assert np.allclose(self.min_max.max, expected)
        min_max = MinMax()
        with pytest.raises(ValueError):
            min_max.fit(np.array([]))

    def test_transform(self):
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([[10]]), np.array([[0.0]])),
            (np.array([[-2]]), np.array([[0.0]])),
            (np.array([[-5, 51]]), np.array([[0.0, 0.0]])),
            (np.array([[1, 2]]), np.array([[0.0, 0.0]])),
            (np.array([[-0.5, 7.6]]), np.array([[0.0, 0.0]])),
            (np.array([[2.718, 0.577, 3.141]]), np.array([[0.0, 0.0, 0.0]])),
            (np.array([[1, 2, 3, -4, 5, 60, -7, 8, 957, 10]]), np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), np.array([[0.9999999999999999, 0.0], [0.0, 1.0000000000000002]])),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), np.array([[0.0, 1.0, 0.5], [1.0, 0.375, 1.0], [0.6046511627906976, 0.0, 0.0]])),
            (np.array([[0, 0], [0, 0]]), np.array([[0.0, 0.0], [0.0, 0.0]])),
            (np.array([[0.1, 0.2, 0.3], [-400.0, 0.5, 6.7]]), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])),
        ]
        for inputs, expected in test_cases:
            self.min_max = MinMax()
            self.min_max.fit(inputs)
            assert np.allclose(self.min_max.transform(inputs), expected)
        min_max = MinMax()
        with pytest.raises(ValueError):
            min_max.transform(np.array([]))
        with pytest.raises(RuntimeError):
            min_max.transform(np.array([1.0]))

    def test_undo(self):
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([[0.0]]), np.array([[10]])),
            (np.array([[0.0]]), np.array([[-2]])),
            (np.array([[0.0, 0.0]]), np.array([[-5, 51]])),
            (np.array([[0.0, 0.0]]), np.array([[1, 2]])),
            (np.array([[0.0, 0.0]]), np.array([[-0.5, 7.6]])),
            (np.array([[0.0, 0.0, 0.0]]), np.array([[2.718, 0.577, 3.141]])),
            (np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), np.array([[1, 2, 3, -4, 5, 60, -7, 8, 957, 10]])),
            (np.array([[0.9999999999999999, 0.0], [0.0, 1.0000000000000002]]), np.array([[0.75, -9.0], [0.1, -0.003]])),
            (np.array([[0.0, 1.0, 0.5], [1.0, 0.375, 1.0], [0.6046511627906976, 0.0, 0.0]]), np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]])),
            (np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[0, 0], [0, 0]])),
            (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]), np.array([[0.1, 0.2, 0.3], [-400.0, 0.5, 6.7]])),
        ]
        for inputs, expected in test_cases:
            self.min_max = MinMax()
            self.min_max.fit(expected)
            assert np.allclose(self.min_max.undo(inputs), expected)
        min_max = MinMax()
        with pytest.raises(ValueError):
            min_max.undo(np.array([]))
        with pytest.raises(RuntimeError):
            min_max.undo(np.array([1.0]))


class TestZScore():
    zscore: ZScore

    def test_fit(self):
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([[10]]), np.array([10.0])),
            (np.array([[-2]]), np.array([-2.0])),
            (np.array([[-5, 51]]), np.array([-5.0, 51.0])),
            (np.array([[1, 2]]), np.array([1.0, 2.0])),
            (np.array([[-0.5, 7.6]]), np.array([-0.5, 7.6])),
            (np.array([[2.718, 0.577, 3.141]]), np.array([2.718, 0.577, 3.141])),
            (np.array([[1, 2, 3, -4, 5, 60, -7, 8, 957, 10]]), np.array([1.0, 2.0, 3.0, -4.0, 5.0, 60.0, -7.0, 8.0, 957.0, 10.0])),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), np.array([0.425, -4.5015])),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), np.array([24.0, 5.933333333333334, 3.0])),
            (np.array([[0, 0], [0, 0]]), np.array([0.0, 0.0])),
            (np.array([[0.1, 0.2, 0.3], [-400.0, 0.5, 6.7]]), np.array([-199.95, 0.35, 3.5])),
        ]
        for inputs, expected in test_cases:
            self.zscore = ZScore()
            self.zscore.fit(inputs)
            assert np.allclose(self.zscore.mean, expected)
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([[10]]), np.array([1.0])),
            (np.array([[-2]]), np.array([1.0])),
            (np.array([[-5, 51]]), np.array([1.0, 1.0])),
            (np.array([[1, 2]]), np.array([1.0, 1.0])),
            (np.array([[-0.5, 7.6]]), np.array([1.0, 1.0])),
            (np.array([[2.718, 0.577, 3.141]]), np.array([1.0, 1.0, 1.0])),
            (np.array([[1, 2, 3, -4, 5, 60, -7, 8, 957, 10]]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), np.array([0.325, 4.4985])),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), np.array([17.682382946499793, 4.619764303752111, 9.797958971132712])),
            (np.array([[0, 0], [0, 0]]), np.array([1.0, 1.0])),
            (np.array([[0.1, 0.2, 0.3], [-400.0, 0.5, 6.7]]), np.array([200.05, 0.15, 3.2])),
        ]
        for inputs, expected in test_cases:
            self.zscore = ZScore()
            self.zscore.fit(inputs)
            assert np.allclose(self.zscore.std_dev, expected)
        zscore = ZScore()
        with pytest.raises(ValueError):
            zscore.fit(np.array([]))

    def test_transform(self):
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([[10]]), np.array([[0.0]])),
            (np.array([[-2]]), np.array([[0.0]])),
            (np.array([[-5, 51]]), np.array([[0.0, 0.0]])),
            (np.array([[1, 2]]), np.array([[0.0, 0.0]])),
            (np.array([[-0.5, 7.6]]), np.array([[0.0, 0.0]])),
            (np.array([[2.718, 0.577, 3.141]]), np.array([[0.0, 0.0, 0.0]])),
            (np.array([[1, 2, 3, -4, 5, 60, -7, 8, 957, 10]]), np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), np.array([[1.0, -1.0], [-0.9999999999999998, 1.0]])),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), np.array([[-1.3007296623757842, 1.313198307917874, 0.0], [1.1310692716311168, -0.2020305089104422, 1.2247448713915892], [0.16966039074466752, -1.1111677990074318, -1.2247448713915892]])),
            (np.array([[0, 0], [0, 0]]), np.array([[0.0, 0.0], [0.0, 0.0]])),
            (np.array([[0.1, 0.2, 0.3], [-400.0, 0.5, 6.7]]), np.array([[0.9999999999999999, -0.9999999999999998, -1.0], [-1.0, 1.0000000000000002, 1.0]])),
        ]
        for inputs, expected in test_cases:
            self.zscore = ZScore()
            self.zscore.fit(inputs)
            assert np.allclose(self.zscore.transform(inputs), expected)
        zscore = ZScore()
        with pytest.raises(ValueError):
            zscore.transform(np.array([]))
        with pytest.raises(RuntimeError):
            zscore.transform(np.array([1.0]))

    def test_undo(self):
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([[0.0]]), np.array([[10]])),
            (np.array([[0.0]]), np.array([[-2]])),
            (np.array([[0.0, 0.0]]), np.array([[-5, 51]])),
            (np.array([[0.0, 0.0]]), np.array([[1, 2]])),
            (np.array([[0.0, 0.0]]), np.array([[-0.5, 7.6]])),
            (np.array([[0.0, 0.0, 0.0]]), np.array([[2.718, 0.577, 3.141]])),
            (np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), np.array([[1, 2, 3, -4, 5, 60, -7, 8, 957, 10]])),
            (np.array([[1.0, -1.0], [-0.9999999999999998, 1.0]]), np.array([[0.75, -9.0], [0.1, -0.003]])),
            (np.array([[-1.3007296623757842, 1.313198307917874, 0.0], [1.1310692716311168, -0.2020305089104422, 1.2247448713915892], [0.16966039074466752, -1.1111677990074318, -1.2247448713915892]]), np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]])),
            (np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[0, 0], [0, 0]])),
            (np.array([[0.9999999999999999, -0.9999999999999998, -1.0], [-1.0, 1.0000000000000002, 1.0]]), np.array([[0.1, 0.2, 0.3], [-400.0, 0.5, 6.7]])),
        ]
        for inputs, expected in test_cases:
            self.zscore = ZScore()
            self.zscore.fit(expected)
            assert np.allclose(self.zscore.undo(inputs), expected)
        zscore = ZScore()
        with pytest.raises(ValueError):
            zscore.undo(np.array([]))
        with pytest.raises(RuntimeError):
            zscore.undo(np.array([1.0]))
