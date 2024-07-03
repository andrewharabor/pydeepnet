
from pydeepnet._typing import *
from pydeepnet.activation_funcs import *


class TestBinaryStep():
    binary_step: BinaryStep = BinaryStep()

    def test_compute(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0]), np.array([1])),
            (np.array([1]), np.array([1])),
            (np.array([-1]), np.array([0])),
            (np.array([0.5]), np.array([1])),
            (np.array([-0.5]), np.array([0])),
            (np.array([-1, 1]), np.array([0, 1])),
            (np.array([0, 0]), np.array([1, 1])),
            (np.array([-1, 0, 1]), np.array([0, 1, 1])),
            (np.array([0.1, -0.1, 0.01]), np.array([1, 0, 1])),
            (np.array([-0.001, 0.001, 0]), np.array([0, 1, 1])),
            (np.array([1, 2, 3]), np.array([1, 1, 1])),
            (np.array([4, -5, -6]), np.array([1, 0, 0])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.array([1, 0, 0, 1]))
        ]
        for input, expected in test_cases:
            assert np.allclose(self.binary_step.compute(input), expected)
        with pytest.raises(ValueError):
            self.binary_step.compute(np.array([]))

    def test_derivative(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0]), np.zeros((1, 1))),
            (np.array([1]), np.zeros((1, 1))),
            (np.array([-1]), np.zeros((1, 1))),
            (np.array([0.5]), np.zeros((1, 1))),
            (np.array([-0.5]), np.zeros((1, 1))),
            (np.array([-1, 1]), np.zeros((2, 2))),
            (np.array([0, 0]), np.zeros((2, 2))),
            (np.array([-1, 0, 1]), np.zeros((3, 3))),
            (np.array([0.1, -0.1, 0.01]), np.zeros((3, 3))),
            (np.array([-0.001, 0.001, 0]), np.zeros((3, 3))),
            (np.array([1, 2, 3]), np.zeros((3, 3))),
            (np.array([4, -5, -6]), np.zeros((3, 3))),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.zeros((4, 4)))
        ]
        for input, expected in test_cases:
            assert np.allclose(self.binary_step.derivative(input), expected)
        with pytest.raises(ValueError):
            self.binary_step.derivative(np.array([]))


class TestLinear():
    linear: Linear = Linear()

    def test_compute(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0]), np.array([0])),
            (np.array([1]), np.array([1])),
            (np.array([-1]), np.array([-1])),
            (np.array([0.5]), np.array([0.5])),
            (np.array([-0.5]), np.array([-0.5])),
            (np.array([-1, 1]), np.array([-1, 1])),
            (np.array([0, 0]), np.array([0, 0])),
            (np.array([-1, 0, 1]), np.array([-1, 0, 1])),
            (np.array([0.1, -0.1, 0.01]), np.array([0.1, -0.1, 0.01])),
            (np.array([-0.001, 0.001, 0]), np.array([-0.001, 0.001, 0])),
            (np.array([1, 2, 3]), np.array([1, 2, 3])),
            (np.array([4, -5, -6]), np.array([4, -5, -6])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.array([3.14, -2.72, -0.577, 1.62]))
        ]
        for input, expected in test_cases:
            assert np.allclose(self.linear.compute(input), expected)
        with pytest.raises(ValueError):
            self.linear.compute(np.array([]))

    def test_derivative(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0]), np.array([1])),
            (np.array([1]), np.array([1])),
            (np.array([-1]), np.array([1])),
            (np.array([0.5]), np.array([1])),
            (np.array([-0.5]),np.array([1])),
            (np.array([-1, 1]), np.eye(2)),
            (np.array([0, 0]), np.eye(2)),
            (np.array([-1, 0, 1]), np.eye(3)),
            (np.array([0.1, -0.1, 0.01]), np.eye(3)),
            (np.array([-0.001, 0.001, 0]), np.eye(3)),
            (np.array([1, 2, 3]), np.eye(3)),
            (np.array([4, -5, -6]), np.eye(3)),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.eye(4))
        ]
        for input, expected in test_cases:
            assert np.allclose(self.linear.derivative(input), expected)
        with pytest.raises(ValueError):
            self.linear.derivative(np.array([]))


class TestReLU():
    relu: ReLU = ReLU(Float64(0.01))

    def test_compute(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0]), np.array([0])),
            (np.array([1]), np.array([1])),
            (np.array([-1]), np.array([-0.01])),
            (np.array([0.5]), np.array([0.5])),
            (np.array([-0.5]), np.array([-0.005])),
            (np.array([-1, 1]), np.array([-0.01, 1])),
            (np.array([0, 0]), np.array([0, 0])),
            (np.array([-1, 0, 1]), np.array([-0.01, 0, 1])),
            (np.array([0.1, -0.1, 0.01]), np.array([0.1, -0.001, 0.01])),
            (np.array([-0.001, 0.001, 0]), np.array([-0.00001, 0.001, 0])),
            (np.array([1, 2, 3]), np.array([1, 2, 3])),
            (np.array([4, -5, -6]), np.array([4, -0.05, -0.06])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.array([3.14, -0.0272, -0.00577, 1.62]))
        ]
        for input, expected in test_cases:
            assert np.allclose(self.relu.compute(input), expected)
        with pytest.raises(ValueError):
            self.relu.compute(np.array([]))

    def test_derivative(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0]), np.array([0.01])),
            (np.array([1]), np.array([1])),
            (np.array([-1]), np.array([0.01])),
            (np.array([0.5]), np.array([1])),
            (np.array([-0.5]), np.array([0.01])),
            (np.array([-1, 1]), np.diag([0.01, 1])),
            (np.array([0, 0]), np.eye(2) * 0.01),
            (np.array([-1, 0, 1]), np.diag([0.01, 0.01, 1])),
            (np.array([0.1, -0.1, 0.01]), np.diag([1, 0.01, 1])),
            (np.array([-0.001, 0.001, 0]), np.diag([0.01, 1, 0.01])),
            (np.array([1, 2, 3]), np.eye(3)),
            (np.array([4, -5, -6]), np.diag([1, 0.01, 0.01])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.diag([1, 0.01, 0.01, 1]))
        ]
        for input, expected in test_cases:
            assert np.allclose(self.relu.derivative(input), expected)
        with pytest.raises(ValueError):
            self.relu.derivative(np.array([]))


class TestSigmoid():
    sigmoid: Sigmoid = Sigmoid()

    def test_compute(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0.0]), np.array([0.5])),
            (np.array([1.0]), np.array([0.7310585975646973])),
            (np.array([-1.0]), np.array([0.2689414322376251])),
            (np.array([0.5]), np.array([0.622459352016449])),
            (np.array([-0.5]), np.array([0.3775406777858734])),
            (np.array([-1.0, 1.0]), np.array([0.2689414322376251, 0.7310585975646973])),
            (np.array([0.0, 0.0]), np.array([0.5, 0.5])),
            (np.array([-1.0, 0.0, 1.0]), np.array([0.2689414322376251, 0.5, 0.7310585975646973])),
            (np.array([0.1, -0.1, 0.01]), np.array([0.5249791741371155, 0.47502079606056213, 0.5024999976158142])),
            (np.array([-0.001, 0.001, 0.0]), np.array([0.499750018119812, 0.500249981880188, 0.5])),
            (np.array([1.0, 2.0, 3.0]), np.array([0.7310585975646973, 0.8807970285415649, 0.9525741338729858])),
            (np.array([4.0, -5.0, -6.0]), np.array([0.9820137619972229, 0.006692850962281227, 0.0024726230185478926])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.array([0.9585129022598267, 0.061803463846445084, 0.3596231937408447, 0.8347951173782349])),
        ]
        for input, expected in test_cases:
            assert np.allclose(self.sigmoid.compute(input), expected)
        with pytest.raises(ValueError):
            self.sigmoid.compute(np.array([]))

    def test_derivative(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0.0]), np.array([0.25])),
            (np.array([1.0]), np.array([0.1966119259595871])),
            (np.array([-1.0]), np.array([0.1966119408607483])),
            (np.array([0.5]), np.array([0.23500370979309082])),
            (np.array([-0.5]), np.array([0.23500370979309082])),
            (np.array([-1.0, 1.0]), np.diag([0.1966119408607483, 0.1966119259595871])),
            (np.array([0.0, 0.0]), np.diag([0.25, 0.25])),
            (np.array([-1.0, 0.0, 1.0]), np.diag([0.1966119408607483, 0.25, 0.1966119259595871])),
            (np.array([0.1, -0.1, 0.001]), np.diag([0.24937604367733002, 0.2493760585784912, 0.2499999375641345978])),
            (np.array([-0.001, 0.001, 0.0]), np.diag([0.24999994039535522, 0.24999994039535522, 0.25])),
            (np.array([1.0, 2.0, 3.0]), np.diag([0.1966119259595871, 0.10499362647533417, 0.04517665505409241])),
            (np.array([4.0, -5.0, -6.0]), np.diag([0.017662733793258667, 0.006648056674748659, 0.0024665091186761856])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.diag([0.039765916764736176, 0.057983797043561935, 0.2302943468093872, 0.13791222870349884])),
        ]
        for input, expected in test_cases:
            assert np.allclose(self.sigmoid.derivative(input), expected)
        with pytest.raises(ValueError):
            self.sigmoid.derivative(np.array([]))


class TestSiLU():
    silu: SiLU = SiLU()

    def test_compute(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0.0]), np.array([0.0])),
            (np.array([1.0]), np.array([0.7310585975646973])),
            (np.array([-1.0]), np.array([-0.2689414322376251])),
            (np.array([0.5]), np.array([0.3112296760082245])),
            (np.array([-0.5]), np.array([-0.1887703388929367])),
            (np.array([-1.0, 1.0]), np.array([-0.2689414322376251, 0.7310585975646973])),
            (np.array([0.0, 0.0]), np.array([0.0, 0.0])),
            (np.array([-1.0, 0.0, 1.0]), np.array([-0.2689414322376251, 0.0, 0.7310585975646973])),
            (np.array([0.1, -0.1, 0.01]), np.array([0.05249791964888573, -0.04750208184123039, 0.005024999845772982])),
            (np.array([-0.001, 0.001, 0.0]), np.array([-0.0004997500218451023, 0.000500250025652349, 0.0])),
            (np.array([1.0, 2.0, 3.0]), np.array([0.7310585975646973, 1.7615940570831299, 2.857722520828247])),
            (np.array([4.0, -5.0, -6.0]), np.array([3.9280550479888916, -0.033464252948760986, -0.014835738576948643])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.array([3.009730577468872, -0.16810542345046997, -0.2075025886297226, 1.3523681163787842])),
        ]
        for input, expected in test_cases:
            assert np.allclose(self.silu.compute(input), expected)
        with pytest.raises(ValueError):
            self.silu.compute(np.array([]))

    def test_derivative(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0.0]), np.array([0.5])),
            (np.array([1.0]), np.array([0.9276705384254456])),
            (np.array([-1.0]), np.array([0.07232949137687683])),
            (np.array([0.5]), np.array([0.7399612069129944])),
            (np.array([-0.5]), np.array([0.260038822889328])),
            (np.array([-1.0, 1.0]), np.diag([0.07232949137687683, 0.9276705384254456])),
            (np.array([0.0, 0.0]), np.diag([0.5, 0.5])),
            (np.array([-1.0, 0.0, 1.0]), np.diag([0.07232949137687683, 0.5, 0.9276705384254456])),
            (np.array([0.1, -0.1, 0.01]), np.diag([0.5499168038368225, 0.4500831961631775, 0.5049999356269836])),
            (np.array([-0.001, 0.001, 0.0]), np.diag([0.49950000643730164, 0.500499963760376, 0.5])),
            (np.array([1.0, 2.0, 3.0]), np.diag([0.9276705384254456, 1.0907843112945557, 1.0881041288375854])),
            (np.array([4.0, -5.0, -6.0]), np.diag([1.0526647567749023, -0.026547430083155632, -0.012326431460678577])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.diag([1.0833778381347656, -0.09591247141361237, 0.22674335539340973, 1.0582129955291748])),
        ]
        for input, expected in test_cases:
            assert np.allclose(self.silu.derivative(input), expected)
        with pytest.raises(ValueError):
            self.silu.derivative(np.array([]))


class TestSoftmax:
    softmax: Softmax = Softmax()

    def test_compute(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0.0]), np.array([1.0])),
            (np.array([1.0]), np.array([1.0])),
            (np.array([-1.0]), np.array([1.0])),
            (np.array([0.5]), np.array([1.0])),
            (np.array([-0.5]), np.array([1.0])),
            (np.array([-1.0, 1.0]), np.array([0.11920291185379028, 0.8807970285415649])),
            (np.array([0.0, 0.0]), np.array([0.5, 0.5])),
            (np.array([-1.0, 0.0, 1.0]), np.array([0.09003057330846786, 0.2447284758090973, 0.6652409434318542])),
            (np.array([0.1, -0.1, 0.01]), np.array([0.3659435510635376, 0.2996092438697815, 0.3344472348690033])),
            (np.array([-0.001, 0.001, 0.0]), np.array([0.3330000340938568, 0.33366671204566956, 0.33333319425582886])),
            (np.array([1.0, 2.0, 3.0]), np.array([0.09003057330846786, 0.2447284758090973, 0.6652409434318542])),
            (np.array([4.0, -5.0, -6.0]), np.array([0.9998311996459961, 0.00012338896340224892, 4.539226574706845e-05])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.array([0.8026520609855652, 0.0022885561920702457, 0.01950988546013832, 0.17554953694343567])),
        ]
        for input, expected in test_cases:
            assert np.allclose(self.softmax.compute(input), expected)
        with pytest.raises(ValueError):
            self.softmax.compute(np.array([]))

    def test_derivative(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0.0]), np.array([[0.0]])),
            (np.array([1.0]), np.array([[0.0]])),
            (np.array([-1.0]), np.array([[0.0]])),
            (np.array([0.5]), np.array([[0.0]])),
            (np.array([-0.5]), np.array([[0.0]])),
            (np.array([-1.0, 1.0]), np.array([[0.10499357432126999, -0.10499356687068939], [-0.10499356687068939, 0.10499362647533417]])),
            (np.array([0.0, 0.0]), np.array([[0.25, -0.25], [-0.25, 0.25]])),
            (np.array([-1.0, 0.0, 1.0]), np.array([[0.08192507177591324, -0.022033045068383217, -0.05989202484488487], [-0.022033045068383217, 0.1848364621400833, -0.16280339658260345], [-0.05989202484488487, -0.16280339658260345, 0.22269542515277863]])),
            (np.array([0.1, -0.1, 0.01]), np.array([[0.23202887177467346, -0.10964006930589676, -0.1223888099193573], [-0.10964006930589676, 0.20984354615211487, -0.1002034842967987], [-0.1223888099193573, -0.1002034842967987, 0.2225922793149948]])),
            (np.array([-0.001, 0.001, 0.0]), np.array([[0.22211100161075592, -0.11111102998256683, -0.11099996417760849], [-0.11111102998256683, 0.22233325242996216, -0.11122219264507294], [-0.11099996417760849, -0.11122219264507294, 0.22222217917442322]])),
            (np.array([1.0, 2.0, 3.0]), np.array([[0.08192507177591324, -0.022033045068383217, -0.05989202484488487], [-0.022033045068383217, 0.1848364621400833, -0.16280339658260345], [-0.05989202484488487, -0.16280339658260345, 0.22269542515277863]])),
            (np.array([4.0, -5.0, -6.0]), np.array([[0.000168752755, -0.00012336813961155713, -4.5384604163700715e-05], [-0.00012336813961155713, 0.00012337374209892005, -5.600904628977332e-09], [-4.5384604163700715e-05, -5.600904628977332e-09, 4.539020301308483e-05]])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.array([[0.1584017276763916, -0.001836914336308837, -0.015659648925065994, -0.1409052014350891], [-0.001836914336308837, 0.002283318666741252, -4.464947051019408e-05, -0.00040175497997552156], [-0.015659648925065994, -4.464947051019408e-05, 0.019129250198602676, -0.0034249513410031796], [-0.1409052014350891, -0.00040175497997552156, -0.0034249513410031796, 0.14473190903663635]])),
        ]
        for input, expected in test_cases:
            assert np.allclose(self.softmax.derivative(input), expected)
        with pytest.raises(ValueError):
            self.softmax.derivative(np.array([]))


class TestSoftplus:
    softplus: Softplus = Softplus()

    def test_compute(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0.0]), np.array([0.6931471824645996])),
            (np.array([1.0]), np.array([1.31326162815094])),
            (np.array([-1.0]), np.array([0.3132616877555847])),
            (np.array([0.5]), np.array([0.9740769863128662])),
            (np.array([-0.5]), np.array([0.4740769863128662])),
            (np.array([-1.0, 1.0]), np.array([0.3132616877555847, 1.31326162815094])),
            (np.array([0.0, 0.0]), np.array([0.6931471824645996, 0.6931471824645996])),
            (np.array([-1.0, 0.0, 1.0]), np.array([0.3132616877555847, 0.6931471824645996, 1.31326162815094])),
            (np.array([0.1, -0.1, 0.01]), np.array([0.7443966865539551, 0.6443966627120972, 0.6981596946716309])),
            (np.array([-0.001, 0.001, 0.0]), np.array([0.6926472783088684, 0.6936473250389099, 0.6931471824645996])),
            (np.array([1.0, 2.0, 3.0]), np.array([1.31326162815094, 2.1269280910491943, 3.0485873222351074])),
            (np.array([4.0, -5.0, -6.0]), np.array([4.0181498527526855, 0.006715348456054926, 0.0024756852071732283])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.array([3.1823723316192627, 0.06379582732915878, 0.44569849967956543, 1.80056893825531])),
        ]
        for input, expected in test_cases:
            assert np.allclose(self.softplus.compute(input), expected)
        with pytest.raises(ValueError):
            self.softplus.compute(np.array([]))

    def test_derivative(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0.0]), np.array([0.5])),
            (np.array([1.0]), np.array([0.7310585975646973])),
            (np.array([-1.0]), np.array([0.2689414322376251])),
            (np.array([0.5]), np.array([0.622459352016449])),
            (np.array([-0.5]), np.array([0.3775406777858734])),
            (np.array([-1.0, 1.0]), np.diag([0.2689414322376251, 0.7310585975646973])),
            (np.array([0.0, 0.0]), np.diag([0.5, 0.5])),
            (np.array([-1.0, 0.0, 1.0]), np.diag([0.2689414322376251, 0.5, 0.7310585975646973])),
            (np.array([0.1, -0.1, 0.01]), np.diag([0.5249791741371155, 0.47502079606056213, 0.5024999976158142])),
            (np.array([-0.001, 0.001, 0.0]), np.diag([0.499750018119812, 0.500249981880188, 0.5])),
            (np.array([1.0, 2.0, 3.0]), np.diag([0.7310585975646973, 0.8807970285415649, 0.9525741338729858])),
            (np.array([4.0, -5.0, -6.0]), np.diag([0.9820137619972229, 0.006692850962281227, 0.0024726230185478926])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.diag([0.9585129022598267, 0.061803463846445084, 0.3596231937408447, 0.8347951173782349])),
        ]
        for input, expected in test_cases:
            assert np.allclose(self.softplus.derivative(input), expected)
        with pytest.raises(ValueError):
            self.softplus.derivative(np.array([]))


class TestTanh:
    tanh: Tanh = Tanh()

    def test_compute(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0.0]), np.array([0.0])),
            (np.array([1.0]), np.array([0.7615941762924194])),
            (np.array([-1.0]), np.array([-0.7615941762924194])),
            (np.array([0.5]), np.array([0.46211716532707214])),
            (np.array([-0.5]), np.array([-0.46211716532707214])),
            (np.array([-1.0, 1.0]), np.array([-0.7615941762924194, 0.7615941762924194])),
            (np.array([0.0, 0.0]), np.array([0.0, 0.0])),
            (np.array([-1.0, 0.0, 1.0]), np.array([-0.7615941762924194, 0.0, 0.7615941762924194])),
            (np.array([0.1, -0.1, 0.01]), np.array([0.0996679961681366, -0.0996679961681366, 0.00999966636300087])),
            (np.array([-0.001, 0.001, 0.0]), np.array([-0.0009999996982514858, 0.0009999996982514858, 0.0])),
            (np.array([1.0, 2.0, 3.0]), np.array([0.7615941762924194, 0.9640275835990906, 0.9950547814369202])),
            (np.array([4.0, -5.0, -6.0]), np.array([0.9993293285369873, -0.9999092221260071, -0.9999877214431763])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.array([0.9962602257728577, -0.9913585186004639, -0.5204815864562988, 0.9246242046356201])),
        ]
        for input, expected in test_cases:
            assert np.allclose(self.tanh.compute(input), expected)
        with pytest.raises(ValueError):
            self.tanh.compute(np.array([]))

    def test_derivative(self) -> None:
        test_cases: list[tuple[NDArray, NDArray]] = [
            (np.array([0.0]), np.array([1.0])),
            (np.array([1.0]), np.array([0.41997432708740234])),
            (np.array([-1.0]), np.array([0.41997432708740234])),
            (np.array([0.5]), np.array([0.7864477038383484])),
            (np.array([-0.5]), np.array([0.7864477038383484])),
            (np.array([-1.0, 1.0]), np.diag([0.41997432708740234, 0.41997432708740234])),
            (np.array([0.0, 0.0]), np.diag([1.0, 1.0])),
            (np.array([-1.0, 0.0, 1.0]), np.diag([0.41997432708740234, 1.0, 0.41997432708740234])),
            (np.array([0.1, -0.1, 0.01]), np.diag([0.9900662899017334, 0.9900662899017334, 0.9998999834060669])),
            (np.array([-0.001, 0.001, 0.0]), np.diag([0.9999989867210388, 0.9999989867210388, 1.0])),
            (np.array([1.0, 2.0, 3.0]), np.diag([0.41997432708740234, 0.07065081596374512, 0.009865999221801758])),
            (np.array([4.0, -5.0, -6.0]), np.diag([0.0013409506828671875, 0.00018158323198583984, 2.4576547460938e-05])),
            (np.array([3.14, -2.72, -0.577, 1.62]), np.diag([0.007465541362762451, 0.0172082781791687, 0.729098916053772, 0.14507007598876953])),
        ]
        for input, expected in test_cases:
            assert np.allclose(self.tanh.derivative(input), expected)
        with pytest.raises(ValueError):
            self.tanh.derivative(np.array([]))
