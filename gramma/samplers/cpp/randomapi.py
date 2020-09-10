#!/usr/bin/env python3

import os
import subprocess
from ctypes import c_double, c_void_p, c_uint64, c_size_t, c_int, CFUNCTYPE, CDLL, POINTER, Structure
from typing import List, TypeVar, Union, cast, Any
import numpy as np
from .glf2cpp import INCLUDE_DIR, CXXFLAGS, get_compiler

c_double_p = POINTER(c_double)

SOURCE_PATH = __file__[:-3] + '_.cpp'
DLL_PATH = __file__[:-3] + '_.so'


def get_dll() -> Any:
    """
    (build and) return dll
    """
    if not os.path.exists(DLL_PATH):
        cxx = get_compiler()
        if cxx is None:
            raise OSError('no compiler found')  # pragma: no cover
        cmd_args = [cxx] + CXXFLAGS + ['-O3', '-fPIC', '-shared', '-I', INCLUDE_DIR, '-o', DLL_PATH, SOURCE_PATH]
        retcode = subprocess.call(cmd_args)
        if retcode != 0:
            raise OSError(f'build returned {retcode}')  # pragma: no cover
    dll = CDLL(DLL_PATH)
    dll.get_api.argtypes = ()
    dll.get_api.restype = POINTER(RandomModuleAPI)
    return dll


class RandomModuleAPI(Structure):
    _fields_ = [
        ('new_random', CFUNCTYPE(c_void_p)),
        ('delete_random', CFUNCTYPE(None, c_void_p)),
        ('seed', CFUNCTYPE(None, c_void_p, c_uint64)),
        ('integers', CFUNCTYPE(c_int, c_void_p, c_int, c_int)),
        ('geometric', CFUNCTYPE(c_int, c_void_p, c_double)),
        ('normal', CFUNCTYPE(c_double, c_void_p, c_double, c_double)),
        ('binomial', CFUNCTYPE(c_int, c_void_p, c_int, c_double)),
        ('weighted_select', CFUNCTYPE(c_int, c_void_p, c_double_p, c_size_t)),
    ]


T = TypeVar('T')


class RandomAPI:
    dll = None

    def __init__(self):
        if RandomAPI.dll is None:
            RandomAPI.dll = get_dll()
        self.api = self.dll.get_api().contents  # type: ignore
        self.p = self.api.new_random()

    def __del__(self):
        self.api.delete_random(self.p)

    def seed(self, v: int) -> None:
        self.api.seed(self.p, v)

    def weighted_select(self, weights: np.ndarray) -> int:
        return cast(int, self.api.weighted_select(self.p, weights.ctypes.data_as(c_double_p), weights.size))

    def choice(self, choices: List[T], weights: Union[None, List[Union[int, float]], np.ndarray] = None) -> T:
        if weights is None:
            weights = np.ones(len(choices), dtype=np.double)
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights, dtype=np.double)
        p = weights.astype(np.double)
        return choices[self.weighted_select(p)]

    def integers(self, lo: int, hi: int) -> int:
        return cast(int, self.api.integers(self.p, lo, hi))

    def geometric(self, p: float) -> int:
        return cast(int, self.api.geometric(self.p, p))

    def normal(self, mean: float, std: float) -> float:
        return cast(float, self.api.normal(self.p, mean, std))

    def binomial(self, n: int, p: float) -> int:
        return cast(int, self.api.binomial(self.p, n, p))
