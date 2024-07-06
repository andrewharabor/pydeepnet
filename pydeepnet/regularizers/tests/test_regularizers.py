
from pydeepnet._typing import *
from pydeepnet.regularizers import *


class TestLasso():
    lasso: Lasso = Lasso(Float64(0.5))

    def test_init(self):
        with pytest.raises(ValueError):
            Lasso(Float64(-0.5))

    def test_compute(self):
        test_cases: list[tuple[NDArray, Float64]] = [
            (np.array([10]), Float64(5.0)),
            (np.array([-2]), Float64(1.0)),
            (np.array([-5, 1]), Float64(3.0)),
            (np.array([1, 2]), Float64(1.5)),
            (np.array([-0.5, 7.6]), Float64(4.05)),
            (np.array([2.718, 0.577, 3.141]), Float64(3.218)),
            (np.array([1, 2, 3, -4, 5, 6, -7, 8, 9, 10]), Float64(27.5)),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), Float64(4.9265)),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), Float64(58.4)),
            (np.array([[0, 0], [0, 0]]), Float64(0.0)),
            (np.array([[0.1, 0.2, 0.3], [-4.0, 0.5, 6.7]]), Float64(5.9)),
        ]
        for inputs, expected in test_cases:
            assert np.isclose(self.lasso.compute(inputs), expected)
        with pytest.raises(ValueError):
            self.lasso.compute(np.array([]))

    def test_derivative(self):
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([10.0]), np.array([0.5])),
            (np.array([-2.0]), np.array([-0.5])),
            (np.array([-5.0, 1.0]), np.array([-0.5, 0.5])),
            (np.array([1.0, 2.0]), np.array([0.5, 0.5])),
            (np.array([-0.5, 7.6]), np.array([-0.5, 0.5])),
            (np.array([2.718, 0.577, 3.141]), np.array([0.5, 0.5, 0.5])),
            (np.array([1.0, 2.0, 3.0, -4.0, 5.0, 6.0, -7.0, 8.0, 9.0, 10.0]), np.array([0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5])),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), np.array([[0.5, -0.5], [0.5, -0.5]])),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]])),
            (np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 0.0]])),
            (np.array([[0.1, 0.2, 0.3], [-4.0, 0.5, 6.7]]), np.array([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]])),
        ]
        for inputs, expected in test_cases:
            assert np.allclose(self.lasso.derivative(inputs), expected)
        with pytest.raises(ValueError):
            self.lasso.derivative(np.array([]))


class TestRidge():
    ridge: Ridge = Ridge(Float64(0.5))

    def test_init(self):
        with pytest.raises(ValueError):
            Ridge(Float64(-0.5))

    def test_compute(self):
        test_cases: list[tuple[NDArray, Float64]] = [
            (np.array([10]), Float64(25.0)),
            (np.array([-2]), Float64(1.0)),
            (np.array([-5, 1]), Float64(6.5)),
            (np.array([1, 2]), Float64(1.25)),
            (np.array([-0.5, 7.6]), Float64(14.5024995803833)),
            (np.array([2.718, 0.577, 3.141]), Float64(4.396583557128906)),
            (np.array([1, 2, 3, -4, 5, 6, -7, 8, 9, 10]), Float64(96.25)),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), Float64(20.39312744140625)),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), Float64(787.6600341796875)),
            (np.array([[0, 0], [0, 0]]), Float64(0.0)),
            (np.array([[0.1, 0.2, 0.3], [-4.0, 0.5, 6.7]]), Float64(15.319998741149902)),
        ]
        for inputs, expected in test_cases:
            assert np.isclose(self.ridge.compute(inputs), expected)
        with pytest.raises(ValueError):
            self.ridge.compute(np.array([]))

    def test_derivative(self):
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([10.0]), np.array([5.0])),
            (np.array([-2.0]), np.array([-1.0])),
            (np.array([-5.0, 1.0]), np.array([-2.5, 0.5])),
            (np.array([1.0, 2.0]), np.array([0.5, 1.0])),
            (np.array([-0.5, 7.6]), np.array([-0.25, 3.799999952316284])),
            (np.array([2.718, 0.577, 3.141]), np.array([1.3589999675750732, 0.28850001096725464, 1.5705000162124634])),
            (np.array([1.0, 2.0, 3.0, -4.0, 5.0, 6.0, -7.0, 8.0, 9.0, 10.0]), np.array([0.5, 1.0, 1.5, -2.0, 2.5, 3.0, -3.5, 4.0, 4.5, 5.0])),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), np.array([[0.375, -4.5], [0.05000000074505806, -0.001500000013038516]])),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), np.array([[0.5, 6.0, 1.5], [22.0, 2.5, 7.5], [13.5, 0.4000000059604645, -4.5]])),
            (np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 0.0]])),
            (np.array([[0.1, 0.2, 0.3], [-4.0, 0.5, 6.7]]), np.array([[0.05000000074505806, 0.10000000149011612, 0.15000000596046448], [-2.0, 0.25, 3.3499999046325684]])),
        ]
        for inputs, expected in test_cases:
            assert np.allclose(self.ridge.derivative(inputs), expected)
        with pytest.raises(ValueError):
            self.ridge.derivative(np.array([]))


class TestElasticNet():
    elastic_net: ElasticNet = ElasticNet(Float64(0.5), Float64(0.5))

    def test_init(self):
        with pytest.raises(ValueError):
            ElasticNet(Float64(-0.5), Float64(0.5))
            ElasticNet(Float64(0.5), Float64(-0.5))
            ElasticNet(Float64(-0.5), Float64(-0.5))

    def test_compute(self):
        test_cases: list[tuple[NDArray, Float64]] = [
            (np.array([10]), Float64(30.0)),
            (np.array([-2]), Float64(2.0)),
            (np.array([-5, 1]), Float64(9.5)),
            (np.array([1, 2]), Float64(2.75)),
            (np.array([-0.5, 7.6]), Float64(18.552499771118164)),
            (np.array([2.718, 0.577, 3.141]), Float64(7.614583492279053)),
            (np.array([1, 2, 3, -4, 5, 6, -7, 8, 9, 10]), Float64(123.75)),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), Float64(25.31962776184082)),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), Float64(846.06005859375)),
            (np.array([[0, 0], [0, 0]]), Float64(0.0)),
            (np.array([[0.1, 0.2, 0.3], [-4.0, 0.5, 6.7]]), Float64(21.21999740600586)),
        ]
        for inputs, expected in test_cases:
            assert np.isclose(self.elastic_net.compute(inputs), expected)
        with pytest.raises(ValueError):
            self.elastic_net.compute(np.array([]))

    def test_derivative(self):
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([10.0]), np.array([5.5])),
            (np.array([-2.0]), np.array([-1.5])),
            (np.array([-5.0, 1.0]), np.array([-3.0, 1.0])),
            (np.array([1.0, 2.0]), np.array([1.0, 1.5])),
            (np.array([-0.5, 7.6]), np.array([-0.75, 4.300000190734863])),
            (np.array([2.718, 0.577, 3.141]), np.array([1.8589999675750732, 0.7885000109672546, 2.070499897003174])),
            (np.array([1.0, 2.0, 3.0, -4.0, 5.0, 6.0, -7.0, 8.0, 9.0, 10.0]), np.array([1.0, 1.5, 2.0, -2.5, 3.0, 3.5, -4.0, 4.5, 5.0, 5.5])),
            (np.array([[0.75, -9.0], [0.1, -0.003]]), np.array([[0.875, -5.0], [0.550000011920929, -0.5015000104904175]])),
            (np.array([[1.0, 12.0, 3.0], [44.0, 5.0, 15.0], [27.0, 0.8, -9.0]]), np.array([[1.0, 6.5, 2.0], [22.5, 3.0, 8.0], [14.0, 0.8999999761581421, -5.0]])),
            (np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 0.0]])),
            (np.array([[0.1, 0.2, 0.3], [-4.0, 0.5, 6.7]]), np.array([[0.550000011920929, 0.6000000238418579, 0.6499999761581421], [-2.5, 0.75, 3.8499999046325684]])),
        ]
        for inputs, expected in test_cases:
            assert np.allclose(self.elastic_net.derivative(inputs), expected)
        with pytest.raises(ValueError):
            self.elastic_net.derivative(np.array([]))
