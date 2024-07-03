from pydeepnet._pytest_tester import PytestTester

run_tests = PytestTester(__name__).run
del PytestTester

from .activation_funcs import (ActivationFunc, BinaryStep, Linear, ReLU,
                               Sigmoid, SiLU, Softmax, Softplus, Tanh)
