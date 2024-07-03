from pydeepnet._pytest_tester import PytestTester

run_tests = PytestTester(__name__).run
del PytestTester

from .optimizers import Adam, GradientDescent, Momentum, Optimizer, RMSProp
