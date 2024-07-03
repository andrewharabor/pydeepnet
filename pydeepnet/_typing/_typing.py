from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import numpy as np
import numpy.linalg as npla
import numpy.typing as npt
import pytest

Int64: TypeAlias = np.int64
Float64: TypeAlias = np.float64
NDArray: TypeAlias = np.ndarray[Any, np.dtype[Int64] | np.dtype[Float64]]
