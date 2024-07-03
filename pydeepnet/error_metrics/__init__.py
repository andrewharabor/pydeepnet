from pydeepnet._pytest_tester import PytestTester

run_tests = PytestTester(__name__).run
del PytestTester

from .error_metrics import (ErrorMetric, MeanAbsolutePercentageError,
                            PercentageAccuracy)
