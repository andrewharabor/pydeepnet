import os
import sys

import pytest


class PytestTester():
    module_path: str

    def __init__(self, module_name: str) -> None:
        self.module_path = os.path.abspath(sys.modules[module_name].__path__[0])

    def run(self, verbose: bool = False) -> bool:
        pytest_args: list[str] = ["-l"]
        if verbose:
            pytest_args.append("-v")
        pytest_args.append(self.module_path)
        try:
            exit_code: int = pytest.main(pytest_args)
        except SystemExit as exception:
            exit_code = exception.code  # type: ignore
        return exit_code == 0
