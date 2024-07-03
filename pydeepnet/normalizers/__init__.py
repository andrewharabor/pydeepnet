from pydeepnet._pytest_tester import PytestTester

run_tests = PytestTester(__name__).run
del PytestTester

from .normalizers import DecimalScaling, MinMax, Normalizer, ZScore
