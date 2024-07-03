from pydeepnet._pytest_tester import PytestTester

run_tests = PytestTester(__name__).run
del PytestTester

from .cost_funcs import (CostFunc, CrossEntropy, Huber, LogCosh,
                         MeanAbsoluteError, MeanSquaredError)
