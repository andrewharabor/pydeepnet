from pydeepnet._pytest_tester import PytestTester

run_tests = PytestTester(__name__).run
del PytestTester

from .pydeepnet import Float64, Int64, NDArray, NeuralNetwork
