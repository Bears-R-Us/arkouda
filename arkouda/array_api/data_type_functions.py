from __future__ import annotations

from .array_object import Array, implements_numpy
from ._dtypes import (
    _all_dtypes,
    _boolean_dtypes,
    _signed_integer_dtypes,
    _unsigned_integer_dtypes,
    _integer_dtypes,
    _real_floating_dtypes,
    _complex_floating_dtypes,
    _numeric_dtypes,
    _result_type,
)

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from ._typing import Dtype
import arkouda as ak
import numpy as np


def astype(x: Array, dtype: Dtype, /, *, copy: bool = True) -> Array:
    """
    Cast an array to a specified data type.
    """
    if not copy and dtype == x.dtype:
        return x
    return Array._new(ak.akcast(x._array, dtype))


def can_cast(from_: Union[Dtype, Array], to: Dtype, /) -> bool:
    """
    Determine whether an array or dtype can be cast to another dtype.
    """
    if isinstance(from_, Array):
        from_ = from_.dtype
    elif from_ not in _all_dtypes:
        raise TypeError(f"{from_=}, but should be an array_api array or dtype")
    if to not in _all_dtypes:
        raise TypeError(f"{to=}, but should be a dtype")
    try:
        # We promote `from_` and `to` together. We then check if the promoted
        # dtype is `to`, which indicates if `from_` can (up)cast to `to`.
        dtype = _result_type(from_, to)
        return to == dtype
    except TypeError:
        # _result_type() raises if the dtypes don't promote together
        return False


@dataclass
class finfo_object:
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: Dtype


@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int
    dtype: Dtype


def finfo(type, /) -> finfo_object:
    fi = np.finfo(type)
    return finfo_object(
        fi.bits,
        float(fi.eps),
        float(fi.max),
        float(fi.min),
        float(fi.smallest_normal),
        fi.dtype,
    )


def iinfo(type, /) -> iinfo_object:
    ii = np.iinfo(type)
    return iinfo_object(ii.bits, ii.max, ii.min, ii.dtype)


def isdtype(
    dtype: Dtype, kind: Union[Dtype, str, Tuple[Union[Dtype, str], ...]]
) -> bool:
    """
    Returns a boolean indicating whether a provided dtype is of a specified data type ``kind``.
    """
    if isinstance(kind, tuple):
        # Disallow nested tuples
        if any(isinstance(k, tuple) for k in kind):
            raise TypeError("'kind' must be a dtype, str, or tuple of dtypes and strs")
        return any(isdtype(dtype, k) for k in kind)
    elif isinstance(kind, str):
        if kind == "bool":
            return dtype in _boolean_dtypes
        elif kind == "signed integer":
            return dtype in _signed_integer_dtypes
        elif kind == "unsigned integer":
            return dtype in _unsigned_integer_dtypes
        elif kind == "integral":
            return dtype in _integer_dtypes
        elif kind == "real floating":
            return dtype in _real_floating_dtypes
        elif kind == "complex floating":
            return dtype in _complex_floating_dtypes
        elif kind == "numeric":
            return dtype in _numeric_dtypes
        else:
            raise ValueError(f"Unrecognized data type kind: {kind!r}")
    elif kind in _all_dtypes:
        return dtype == kind
    else:
        raise TypeError(
            f"'kind' must be a dtype, str, \
            or tuple of dtypes and strs, not {type(kind).__name__}"
        )


@implements_numpy(np.result_type)
def result_type(*arrays_and_dtypes: Union[Array, Dtype]) -> Dtype:
    """
    Compute the result dtype for a group of arrays and/or dtypes.
    """
    A = []
    for a in arrays_and_dtypes:
        if isinstance(a, Array):
            a = a.dtype
        elif isinstance(a, np.ndarray):
            a = a.dtype
        elif a not in _all_dtypes:
            raise TypeError("result_type() inputs must be array_api arrays or dtypes")
        A.append(a)

    if len(A) == 0:
        raise ValueError("at least one array or dtype is required")
    elif len(A) == 1:
        return A[0]
    else:
        t = A[0]
        for t2 in A[1:]:
            t = _result_type(t, t2)
        return t
