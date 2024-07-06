from pydeepnet._pytest_tester import PytestTester

run_tests = PytestTester(__name__).run
del PytestTester

from .regularizers import ElasticNet, Lasso, Regularizer, Ridge
