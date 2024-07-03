from pydeepnet._pytest_tester import PytestTester

run_tests = PytestTester(__name__).run
del PytestTester

from .layers import DenseLayer, DenseOutputLayer, InputLayer, Layer
