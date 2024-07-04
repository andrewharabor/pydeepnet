
from pydeepnet._typing import *
from pydeepnet.cost_funcs import *


class TestCrossEntropy():
    cross_entropy: CrossEntropy = CrossEntropy()

    def test_compute(self) -> None:
        test_cases: list[tuple[tuple[NDArray, NDArray], Float64]] = [
            ((np.array([1]), np.array([1])), Float64(0.0)),
            ((np.array([0.5, 0.5]), np.array([1, 0])), Float64(0.6931471824645996)),
            ((np.array([0.1, 0.9]), np.array([0, 1])), Float64(0.10536054521799088)),
            ((np.array([0.001, 0.999]), np.array([1, 0])), Float64(6.907755374908447)),
            ((np.array([0.01, 0.54, 0.37, 0.08]), np.array([0, 0, 1, 0])), Float64(0.9942522644996643)),
            ((np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 0, 1])), Float64(0.9162907004356384)),
            ((np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])), Float64(2.3025851249694824)),
            ((np.array([0.0, 0.0, 1.0]), np.array([1, 0, 0])), Float64(20.72326583694641)),
            ((np.array([0.0, 1.0, 0.0]), np.array([0, 1, 0])), Float64(1.1920928955078125e-09)),
            ((np.array([0.3184, 0.1292, 0.5524]), np.array([0, 0, 1])), Float64(0.5934828519821167)),
            ((np.array([0.05, 0.95, 0.0]), np.array([0, 1, 0])), Float64(0.051293306052684784)),
            ((np.array([0.1, 0.8, 0.1]), np.array([0, 0, 1])), Float64(2.3025851249694824)),
        ]
        for (predictions, targets), expected in test_cases:
            assert np.isclose(self.cross_entropy.compute(predictions, targets), expected)
        with pytest.raises(ValueError):
            self.cross_entropy.compute(np.array([]), np.array([1]))
            self.cross_entropy.compute(np.array([1]), np.array([]))
            self.cross_entropy.compute(np.array([]), np.array([]))
            self.cross_entropy.compute(np.array([1]), np.array([1, 0]))
            self.cross_entropy.compute(np.array([1, 0]), np.array([1]))
            self.cross_entropy.compute(np.array([0.5, 0.5, 0.5]), np.array([1, 0, 0]))
            self.cross_entropy.compute(np.array([-0.5, 2]), np.array([0, 1]))
            self.cross_entropy.compute(np.array([0.1, 0.9]), np.array([0, 2]))
            self.cross_entropy.compute(np.array([0.7, 0.3]), np.array([0, 0]))

    def test_derivative(self) -> None:
        test_cases: list[tuple[tuple[NDArray, NDArray], NDArray]] = [
            ((np.array([1]), np.array([1])), np.array([-0.9999999989999999])),
            ((np.array([0.5, 0.5]), np.array([1, 0])), np.array([-1.9999999960000001, 0.0])),
            ((np.array([0.1, 0.9]), np.array([0, 1])), np.array([0.0, -1.1111111098765432])),
            ((np.array([0.001, 0.999]), np.array([1, 0])), np.array([-999.9990000010001, 0.0])),
            ((np.array([0.01, 0.54, 0.37, 0.08]), np.array([0, 0, 1, 0])), np.array([0.0, 0.0, -2.702702695398101, 0.0])),
            ((np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 0, 1])), np.array([0.0, 0.0, 0.0, -2.49999999375])),
            ((np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])), np.array([0.0, 0.0, 0.0, 0.0, -9.9999999, 0.0, 0.0, 0.0, 0.0, 0.0])),
            ((np.array([0.0, 0.0, 1.0]), np.array([1, 0, 0])), np.array([-999999999.9999999, 0.0, 0.0])),
            ((np.array([0.0, 1.0, 0.0]), np.array([0, 1, 0])), np.array([0.0, -0.9999999989999999, 0.0])),
            ((np.array([0.3184, 0.1292, 0.5524]), np.array([0, 0, 1])), np.array([0.0, 0.0, -1.8102824007779104])),
            ((np.array([0.05, 0.95, 0.0]), np.array([0, 1, 0])), np.array([0.0, -1.0526315778393354, 0.0])),
            ((np.array([0.1, 0.8, 0.1]), np.array([0, 0, 1])), np.array([0.0, 0.0, -9.9999999])),
        ]
        for (predictions, targets), expected in test_cases:
            assert np.all(np.isclose(self.cross_entropy.derivative(predictions, targets), expected))
        with pytest.raises(ValueError):
            self.cross_entropy.derivative(np.array([]), np.array([1]))
            self.cross_entropy.derivative(np.array([1]), np.array([]))
            self.cross_entropy.derivative(np.array([]), np.array([]))
            self.cross_entropy.derivative(np.array([1]), np.array([1, 0]))
            self.cross_entropy.derivative(np.array([1, 0]), np.array([1]))
            self.cross_entropy.derivative(np.array([0.5, 0.5, 0.5]), np.array([1, 0, 0]))
            self.cross_entropy.derivative(np.array([-0.5, 2]), np.array([0, 1]))
            self.cross_entropy.derivative(np.array([0.1, 0.9]), np.array([0, 2]))
            self.cross_entropy.derivative(np.array([0.7, 0.3]), np.array([0, 0]))


class TestHuber():
    huber: Huber = Huber(Float64(0.5))

    def test_init(self) -> None:
        with pytest.raises(ValueError):
            Huber(Float64(-0.5))

    def test_compute(self) -> None:
        test_cases: list[tuple[tuple[NDArray, NDArray], Float64]] = [
            ((np.array([0]), np.array([0.2])), Float64(0.020000001415610313)),
            ((np.array([10]), np.array([8.7])), Float64(0.5250000953674316)),
            ((np.array([-5, 1]), np.array([-4.3, 0.9])), Float64(0.11499994993209839)),
            ((np.array([0.1, 3.1415]), np.array([0.0985, 3.125])), Float64(6.862496957182884e-05)),
            ((np.array([0.1, 0.2, 0.3]), np.array([-11.0, 0.75, 13.0])), Float64(3.933333396911621)),
            ((np.array([-4.0, 0.5, 6.7]), np.array([-4.8, 0.6, 6.0])), Float64(0.16833333671092987)),
            ((np.array([10, 214, 9, 0]), np.array([-10.0, 19.0, 89.9, -0.43])), Float64(36.91686248779297)),
            ((np.array([0.1, 0.05, 0.75, 0.05, 0.15]), np.array([0, 0, 1, 0, 0])), Float64(0.009999999776482582)),
            ((np.array([13.0, 42.0, -0.49, -8.1428]), np.array([13.0, 42.0, -0.49, -8.1428])), Float64(0.0)),
            ((np.array([1.0, -1.0, 0.0, 0.001, -0.001]), np.array([2, -2, 25, 0, 0])), Float64(2.625000476837158)),
        ]
        for (predictions, targets), expected in test_cases:
            assert np.isclose(self.huber.compute(predictions, targets), expected)
        with pytest.raises(ValueError):
            self.huber.compute(np.array([]), np.array([100, 99]))
            self.huber.compute(np.array([8, 1]), np.array([]))
            self.huber.compute(np.array([]), np.array([]))
            self.huber.compute(np.array([1, 2]), np.array([4]))
            self.huber.compute(np.array([1]), np.array([2, 2]))

    def test_derivative(self) -> None:
        test_cases: list[tuple[tuple[NDArray, NDArray], NDArray]] = [
            ((np.array([0.0]), np.array([0.2])), np.array([-0.20000000298023224])),
            ((np.array([10.0]), np.array([8.7])), np.array([0.5])),
            ((np.array([-5.0, 1.0]), np.array([-4.3, 0.9])), np.array([-0.25, 0.050000011920928955])),
            ((np.array([0.1, 3.1415]), np.array([0.0985, 3.125])), np.array([0.0007500015199184418, 0.008249998092651367])),
            ((np.array([0.1, 0.2, 0.3]), np.array([-11.0, 0.75, 13.0])), np.array([0.1666666716337204, -0.1666666716337204, -0.1666666716337204])),
            ((np.array([-4.0, 0.5, 6.7]), np.array([-4.8, 0.6, 6.0])), np.array([0.1666666716337204, -0.033333342522382736, 0.1666666716337204])),
            ((np.array([10.0, 214.0, 9.0, 0.0]), np.array([-10.0, 19.0, 89.9, -0.43])), np.array([0.125, 0.125, -0.125, 0.10750000178813934])),
            ((np.array([0.1, 0.05, 0.75, 0.05, 0.15]), np.array([0, 0, 1, 0, 0])), np.array([0.020000001415610313, 0.010000000707805157, -0.05000000074505806, 0.010000000707805157, 0.030000001192092896])),
            ((np.array([13.0, 42.0, -0.49, -8.1428]), np.array([13.0, 42.0, -0.49, -8.1428])), np.array([0.0, 0.0, 0.0, 0.0])),
            ((np.array([1.0, -1.0, 0.0, 0.001, -0.001]), np.array([2, -2, 25, 0, 0])), np.array([-0.10000000149011612, 0.10000000149011612, -0.10000000149011612, 0.00020000000949949026, -0.00020000000949949026])),
        ]
        for (predictions, targets), expected in test_cases:
            assert np.all(np.isclose(self.huber.derivative(predictions, targets), expected))
        with pytest.raises(ValueError):
            self.huber.derivative(np.array([]), np.array([100, 99]))
            self.huber.derivative(np.array([8, 1]), np.array([]))
            self.huber.derivative(np.array([]), np.array([]))
            self.huber.derivative(np.array([1, 2]), np.array([4]))
            self.huber.derivative(np.array([1]), np.array([2, 2]))


class TestLogCosh():
    log_cosh: LogCosh = LogCosh()

    def test_compute(self) -> None:
        test_cases: list[tuple[tuple[NDArray, NDArray], Float64]] = [
            ((np.array([0]), np.array([0.2])), Float64(0.019868075847625732)),
            ((np.array([10]), np.array([8.7])), Float64(0.6784976720809937)),
            ((np.array([-5, 1]), np.array([-4.3, 0.9])), Float64(0.11613091826438904)),
            ((np.array([0.1, 3.1415]), np.array([0.0985, 3.125])), Float64(6.86219116778623e-05)),
            ((np.array([0.1, 0.2, 0.3]), np.array([-11.0, 0.75, 13.0])), Float64(7.519298076629639)),
            ((np.array([-4.0, 0.5, 6.7]), np.array([-4.8, 0.6, 6.0])), Float64(0.17433850467205048)),
            ((np.array([10, 214, 9, 0]), np.array([-10.0, 19.0, 89.9, -0.43])), Float64(73.47756958007812)),
            ((np.array([0.1, 0.05, 0.75, 0.05, 0.15]), np.array([0, 0, 1, 0, 0])), Float64(0.00992571096867323)),
            ((np.array([13.0, 42.0, -0.49, -8.1428]), np.array([13.0, 42.0, -0.49, -8.1428])), Float64(0.0)),
            ((np.array([1.0, -1.0, 0.0, 0.001, -0.001]), np.array([2, -2, 25, 0, 0])), Float64(5.034882545471191)),
        ]
        for (predictions, targets), expected in test_cases:
            assert np.isclose(self.log_cosh.compute(predictions, targets), expected)
        with pytest.raises(ValueError):
            self.log_cosh.compute(np.array([]), np.array([100, 99]))
            self.log_cosh.compute(np.array([8, 1]), np.array([]))
            self.log_cosh.compute(np.array([]), np.array([]))
            self.log_cosh.compute(np.array([1, 2]), np.array([4]))
            self.log_cosh.compute(np.array([1]), np.array([2, 2]))

    def test_derivative(self) -> None:
        test_cases: list[tuple[tuple[NDArray, NDArray], NDArray]] = [
            ((np.array([0.0]), np.array([0.2])), np.array([-0.19737541675567627])),
            ((np.array([10.0]), np.array([8.7])), np.array([0.8617231845855713])),
            ((np.array([-5.0, 1.0]), np.array([-4.3, 0.9])), np.array([-0.3021838068962097, 0.04983401298522949])),
            ((np.array([0.1, 3.1415]), np.array([0.0985, 3.125])), np.array([0.0007500052452087402, 0.008249253034591675])),
            ((np.array([0.1, 0.2, 0.3]), np.array([-11.0, 0.75, 13.0])), np.array([0.3333333432674408, -0.1668401062488556, -0.3333333432674408])),
            ((np.array([-4.0, 0.5, 6.7]), np.array([-4.8, 0.6, 6.0])), np.array([0.22134563326835632, -0.03322267532348633, 0.20145589113235474])),
            ((np.array([10.0, 214.0, 9.0, 0.0]), np.array([-10.0, 19.0, 89.9, -0.43])), np.array([0.25, 0.25, -0.25, 0.10133033990859985])),
            ((np.array([0.1, 0.05, 0.75, 0.05, 0.15]), np.array([0, 0, 1, 0, 0])), np.array([0.019933611154556274, 0.009991675615310669, -0.04898373782634735, 0.009991675615310669, 0.02977702021598816])),
            ((np.array([13.0, 42.0, -0.49, -8.1428]), np.array([13.0, 42.0, -0.49, -8.1428])), np.array([0.0, 0.0, 0.0, 0.0])),
            ((np.array([1.0, -1.0, 0.0, 0.001, -0.001]), np.array([2, -2, 25, 0, 0])), np.array([-0.15231885015964508, 0.1523188352584839, -0.20000000298023224, 0.00020000338554382324, -0.00020000338554382324])),
        ]
        for (predictions, targets), expected in test_cases:
            assert np.all(np.isclose(self.log_cosh.derivative(predictions, targets), expected))
        with pytest.raises(ValueError):
            self.log_cosh.derivative(np.array([]), np.array([100, 99]))
            self.log_cosh.derivative(np.array([8, 1]), np.array([]))
            self.log_cosh.derivative(np.array([]), np.array([]))
            self.log_cosh.derivative(np.array([1, 2]), np.array([4]))
            self.log_cosh.derivative(np.array([1]), np.array([2, 2]))


class TestMeanAbsoluteError():
    mean_absolute_error: MeanAbsoluteError = MeanAbsoluteError()

    def test_compute(self) -> None:
        test_cases: list[tuple[tuple[NDArray, NDArray], Float64]] = [
            ((np.array([0]), np.array([0.2])), Float64(0.20000000298023224)),
            ((np.array([10]), np.array([8.7])), Float64(1.3000001907348633)),
            ((np.array([-5, 1]), np.array([-4.3, 0.9])), Float64(0.3999999165534973)),
            ((np.array([0.1, 3.1415]), np.array([0.0985, 3.125])), Float64(0.008999999612569809)),
            ((np.array([0.1, 0.2, 0.3]), np.array([-11.0, 0.75, 13.0])), Float64(8.116666793823242)),
            ((np.array([-4.0, 0.5, 6.7]), np.array([-4.8, 0.6, 6.0])), Float64(0.5333333611488342)),
            ((np.array([10, 214, 9, 0]), np.array([-10.0, 19.0, 89.9, -0.43])), Float64(74.0824966430664)),
            ((np.array([0.1, 0.05, 0.75, 0.05, 0.15]), np.array([0, 0, 1, 0, 0])), Float64(0.12000000476837158)),
            ((np.array([13.0, 42.0, -0.49, -8.1428]), np.array([13.0, 42.0, -0.49, -8.1428])), Float64(0.0)),
            ((np.array([1.0, -1.0, 0.0, 0.001, -0.001]), np.array([2, -2, 25, 0, 0])), Float64(5.400399684906006)),
        ]
        for (predictions, targets), expected in test_cases:
            assert np.isclose(self.mean_absolute_error.compute(predictions, targets), expected)
        with pytest.raises(ValueError):
            self.mean_absolute_error.compute(np.array([]), np.array([100, 99]))
            self.mean_absolute_error.compute(np.array([8, 1]), np.array([]))
            self.mean_absolute_error.compute(np.array([]), np.array([]))
            self.mean_absolute_error.compute(np.array([1, 2]), np.array([4]))
            self.mean_absolute_error.compute(np.array([1]), np.array([2, 2]))

    def test_derivative(self) -> None:
        test_cases: list[tuple[tuple[NDArray, NDArray], NDArray]] = [
            ((np.array([0.0]), np.array([0.2])), np.array([-1.0])),
            ((np.array([10.0]), np.array([8.7])), np.array([1.0])),
            ((np.array([-5.0, 1.0]), np.array([-4.3, 0.9])), np.array([-0.5, 0.5])),
            ((np.array([0.1, 3.1415]), np.array([0.0985, 3.125])), np.array([0.5, 0.5])),
            ((np.array([0.1, 0.2, 0.3]), np.array([-11.0, 0.75, 13.0])), np.array([0.3333333432674408, -0.3333333432674408, -0.3333333432674408])),
            ((np.array([-4.0, 0.5, 6.7]), np.array([-4.8, 0.6, 6.0])), np.array([0.3333333432674408, -0.3333333432674408, 0.3333333432674408])),
            ((np.array([10.0, 214.0, 9.0, 0.0]), np.array([-10.0, 19.0, 89.9, -0.43])), np.array([0.25, 0.25, -0.25, 0.25])),
            ((np.array([0.1, 0.05, 0.75, 0.05, 0.15]), np.array([0, 0, 1, 0, 0])), np.array([0.20000000298023224, 0.20000000298023224, -0.20000000298023224, 0.20000000298023224, 0.20000000298023224])),
            ((np.array([13.0, 42.0, -0.49, -8.1428]), np.array([13.0, 42.0, -0.49, -8.1428])), np.array([-0.0, -0.0, -0.0, -0.0])),
            ((np.array([1.0, -1.0, 0.0, 0.001, -0.001]), np.array([2, -2, 25, 0, 0])), np.array([-0.20000000298023224, 0.20000000298023224, -0.20000000298023224, 0.20000000298023224, -0.20000000298023224])),
        ]
        for (predictions, targets), expected in test_cases:
            assert np.all(np.isclose(self.mean_absolute_error.derivative(predictions, targets), expected))
        with pytest.raises(ValueError):
            self.mean_absolute_error.derivative(np.array([]), np.array([100, 99]))
            self.mean_absolute_error.derivative(np.array([8, 1]), np.array([]))
            self.mean_absolute_error.derivative(np.array([]), np.array([]))
            self.mean_absolute_error.derivative(np.array([1, 2]), np.array([4]))
            self.mean_absolute_error.derivative(np.array([1]), np.array([2, 2]))


class TestMeanSquaredError():
    mean_squared_error: MeanSquaredError = MeanSquaredError()

    def test_compute(self) -> None:
        test_cases: list[tuple[tuple[NDArray, NDArray], Float64]] = [
            ((np.array([0]), np.array([0.2])), Float64(0.020000001415610313)),
            ((np.array([10]), np.array([8.7])), Float64(0.8450002670288086)),
            ((np.array([-5, 1]), np.array([-4.3, 0.9])), Float64(0.12499993294477463)),
            ((np.array([0.1, 3.1415]), np.array([0.0985, 3.125])), Float64(6.862496957182884e-05)),
            ((np.array([0.1, 0.2, 0.3]), np.array([-11.0, 0.75, 13.0])), Float64(47.46708297729492)),
            ((np.array([-4.0, 0.5, 6.7]), np.array([-4.8, 0.6, 6.0])), Float64(0.1899999976158142)),
            ((np.array([10, 214, 9, 0]), np.array([-10.0, 19.0, 89.9, -0.43])), Float64(5621.2490234375)),
            ((np.array([0.1, 0.05, 0.75, 0.05, 0.15]), np.array([0, 0, 1, 0, 0])), Float64(0.009999999776482582)),
            ((np.array([13.0, 42.0, -0.49, -8.1428]), np.array([13.0, 42.0, -0.49, -8.1428])), Float64(0.0)),
            ((np.array([1.0, -1.0, 0.0, 0.001, -0.001]), np.array([2, -2, 25, 0, 0])), Float64(62.70000076293945)),
        ]
        for (predictions, targets), expected in test_cases:
            assert np.isclose(self.mean_squared_error.compute(predictions, targets), expected)
        with pytest.raises(ValueError):
            self.mean_squared_error.compute(np.array([]), np.array([100, 99]))
            self.mean_squared_error.compute(np.array([8, 1]), np.array([]))
            self.mean_squared_error.compute(np.array([]), np.array([]))
            self.mean_squared_error.compute(np.array([1, 2]), np.array([4]))
            self.mean_squared_error.compute(np.array([1]), np.array([2, 2]))

    def test_derivative(self) -> None:
        test_cases: list[tuple[tuple[NDArray, NDArray], NDArray]] = [
            ((np.array([0.0]), np.array([0.2])), np.array([-0.20000000298023224])),
            ((np.array([10.0]), np.array([8.7])), np.array([1.3000001907348633])),
            ((np.array([-5.0, 1.0]), np.array([-4.3, 0.9])), np.array([-0.34999990463256836, 0.050000011920928955])),
            ((np.array([0.1, 3.1415]), np.array([0.0985, 3.125])), np.array([0.0007500015199184418, 0.008249998092651367])),
            ((np.array([0.1, 0.2, 0.3]), np.array([-11.0, 0.75, 13.0])), np.array([3.700000286102295, -0.18333333730697632, -4.233333587646484])),
            ((np.array([-4.0, 0.5, 6.7]), np.array([-4.8, 0.6, 6.0])), np.array([0.2666667401790619, -0.033333342522382736, 0.2333332747220993])),
            ((np.array([10.0, 214.0, 9.0, 0.0]), np.array([-10.0, 19.0, 89.9, -0.43])), np.array([5.0, 48.75, -20.225000381469727, 0.10750000178813934])),
            ((np.array([0.1, 0.05, 0.75, 0.05, 0.15]), np.array([0, 0, 1, 0, 0])), np.array([0.020000001415610313, 0.010000000707805157, -0.05000000074505806, 0.010000000707805157, 0.030000001192092896])),
            ((np.array([13.0, 42.0, -0.49, -8.1428]), np.array([13.0, 42.0, -0.49, -8.1428])), np.array([-0.0, -0.0, -0.0, -0.0])),
            ((np.array([1.0, -1.0, 0.0, 0.001, -0.001]), np.array([2, -2, 25, 0, 0])), np.array([-0.20000000298023224, 0.20000000298023224, -5.0, 0.00020000000949949026, -0.00020000000949949026])),
        ]
        for (predictions, targets), expected in test_cases:
            assert np.all(np.isclose(self.mean_squared_error.derivative(predictions, targets), expected))
        with pytest.raises(ValueError):
            self.mean_squared_error.derivative(np.array([]), np.array([100, 99]))
            self.mean_squared_error.derivative(np.array([8, 1]), np.array([]))
            self.mean_squared_error.derivative(np.array([]), np.array([]))
            self.mean_squared_error.derivative(np.array([1, 2]), np.array([4]))
            self.mean_squared_error.derivative(np.array([1]), np.array([2, 2]))
