from __future__ import annotations


from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from arkouda.client import generic_msg
import numpy as np
from arkouda.pdarrayclass import create_pdarray

if TYPE_CHECKING:
    from ._typing import (
        Array,
        Device,
        Dtype,
        NestedSequence,
        SupportsBufferProtocol,
    )
    from collections.abc import Sequence
from ._dtypes import _all_dtypes

import arkouda as ak


def _check_valid_dtype(dtype):
    # Note: Only spelling dtypes as the dtype objects is supported.

    # We use this instead of "dtype in _all_dtypes" because the dtype objects
    # define equality with the sorts of things we want to disallow.
    for d in (None,) + _all_dtypes:
        if dtype is d:
            return
    raise ValueError("dtype must be one of the supported dtypes")


def asarray(
    obj: Union[
        Array,
        bool,
        int,
        float,
        NestedSequence[bool | int | float],
        SupportsBufferProtocol,
    ],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    copy: Optional[bool] = None,
) -> Array:
    from ._array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    if isinstance(obj, Array):
        return Array._new(obj._array)
    elif dtype is not None:
        res = ak.full(1, obj, dtype)
        return Array._new(res)
    else:
        res = ak.full(1, obj)
        return Array._new(res)



def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    from ._array_object import Array
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    if stop is None:
        return Array._new(ak.arange(0, start, step, dtype))
    else:
        return Array._new(ak.arange(start, stop, step, dtype))


def empty(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    from ._array_object import Array
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    if isinstance(shape, Tuple):
        size = 1
        for s in shape:
            size *= s
        return Array._new(pdarray("_empty_", dtype, size, len(shape), shape, 0, None))
    else:
        return Array._new(pdarray("_empty_", dtype, shape, 1, shape, 0, None))


def empty_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    from ._array_object import Array
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    return Array._new(
        pdarray("_empty", dtype, x._array.dtype, x._array.size, len(x._array.shape), \
            x._array.shape, x._array.itemsize, x._array.max_bits)
    )


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    from ._array_object import Array
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    cols = n_rows
    if n_cols is not None:
        cols = n_cols

    repMsg = generic_msg(
        cmd="eye",
        args={
            "dtype": np.dtype(dtype).name,
            "rows": n_rows,
            "cols": cols,
            "diag": k,
        },
    )

    return Array._new(create_pdarray(repMsg))


def from_dlpack(x: object, /) -> Array:
    #TODO: What is this?
    raise ValueError(f"Not implemented")


def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    from ._array_object import Array

    a = Array.zeros(shape, dtype=dtype, device=device)
    a._array.fill(fill_value)
    return a


def full_like(
    x: Array,
    /,
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    from ._array_object import Array
    return Array.full(x.shape, fill_value, dtype=dtype, device=device)


def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
) -> Array:
    from ._array_object import Array
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    return Array._new(ak.linspace(start, stop, num))


def meshgrid(*arrays: Array, indexing: str = "xy") -> List[Array]:
    # if indexing not in ['xy', 'ij']:
    #     raise ValueError(
    #         "Valid values for `indexing` are 'xy' and 'ij'.")

    # array_names = "["
    # first = True
    # dim = 1
    # for a in arrays:
    #     if first:
    #         dim = a._array.ndim
    #         first = False
    #     else:
    #         if a._array.dim != dim:
    #             raise ValueError(f"all arrays must have the same dimensionality for 'meshgrid'")
    #         array_names += ","
    #     array_names += x._array.name
    # array_names += "]"

    # repMsg = generic_msg(
    #     cmd=f"meshgrid{dim}D",
    #     args={
    #         "num": len(array_names),
    #         "arrays": array_names,
    #         "indexing": indexing,
    #     },
    # )

    # arrayMsgs = repMsg.split(",")
    # return [Array._new(create_pdarray(msg)) for msg in arrayMsgs]
    raise ValueError(f"Not implemented")


def ones(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    from ._array_object import Array

    a = Array.zeros(shape, dtype=dtype, device=device)
    a._array.fill(1)
    return a



def ones_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    from ._array_object import Array
    return Array.ones(x.shape, dtype=dtype, device=device)


def tril(x: Array, /, *, k: int = 0) -> Array:
    from ._array_object import Array
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    repMsg = generic_msg(
        cmd=f"tril{x._array.ndim}D",
        args={
            "array": x._array.name,
            "diag": k,
        },
    )

    return Array._new(create_pdarray(repMsg))


def triu(x: Array, /, *, k: int = 0) -> Array:
    from ._array_object import Array
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    repMsg = generic_msg(
        cmd=f"triu{x._array.ndim}D",
        args={
            "array": x._array.name,
            "diag": k,
        },
    )

    return Array._new(create_pdarray(repMsg))

def zeros(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    from ._array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    if isinstance(shape, Tuple):
        size = 1
        for s in shape:
            size *= s
        ndim = len(shape)
    else:
        size = shape
        ndim = 1

    repMsg = generic_msg(
        cmd=f"create{ndim}D",
        args={
            "dtype": np.dtype(dtype).name,
            "shape": shape,
        },
    )

    return Array._new(create_pdarray(repMsg))


def zeros_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    from ._array_object import Array
    return Array.zeros(x.shape, dtype=dtype, device=device)
