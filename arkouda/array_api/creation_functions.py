from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union, cast

import numpy as np

import arkouda as ak
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import resolve_scalar_dtype
from arkouda.numpy.pdarrayclass import _to_pdarray, pdarray


__all__ = [
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
]


if TYPE_CHECKING:
    from ._typing import Array, Device, Dtype, NestedSequence, SupportsBufferProtocol


def asarray(
    obj: Union[
        Array,
        bool,
        int,
        float,
        NestedSequence[bool | int | float],
        SupportsBufferProtocol,
        ak.pdarray,
        np.ndarray,
    ],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    copy: Optional[bool] = None,
) -> Array:
    """
    Create a new Array from one of:
    - another Array
    - a scalar value (bool, int, float)
    - a sequence of scalar values (not yet implemented)
    - a buffer (not yet implemented)
    - an arkouda class:`~arkouda.numpy.pdarrayclass.pdarray`
    - a numpy ndarray.

    Parameters
    ----------
    obj : Array, bool, int, float, complex,
          sequence of scalars, buffer, pdarray, or numpy.ndarray
        The object to convert to an Array
    dtype: Optional[Dtype]
        The dtype of the resulting Array. If None, the dtype is inferred from the input object
    device: Optional[Device]
        The device on which to create the Array (not yet implemented)
    copy: Optional[bool]
        Whether to copy the input object (not yet implemented)
    """
    from .array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    if isinstance(obj, ak.pdarray):
        return Array._new(obj)
    elif isinstance(obj, (bool, int, float)):
        if dtype is None:
            xdtype = akdtype(resolve_scalar_dtype(obj))
        else:
            xdtype = akdtype(dtype)
        res = ak.full(1, obj, xdtype)
        return Array._new(res)
    elif isinstance(obj, complex):
        raise TypeError("complex dtype is not supported in Arkouda Arrays")
    elif isinstance(obj, Array):
        return Array._new(ak.array(obj._array))
    elif isinstance(obj, ak.pdarray):
        return Array._new(obj)
    elif isinstance(obj, np.ndarray):
        return Array._new(_to_pdarray(obj, dt=dtype))
    else:
        raise ValueError("asarray not implemented for 'NestedSequence' or 'SupportsBufferProtocol'")


def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Return a 1D of array of evenly spaced values within the half-open interval [start, stop).

    Parameters
    ----------
    start: Union[int, float]
        If `stop` is None, this is the stop value and start is 0. Otherwise,
        this is the start value (inclusive).
    stop: Optional[Union[int, float]]
        The end value of the sequence (exclusive).
    step: Union[int, float]
        Spacing between values (default is 1).
    dtype: Optional[Dtype]
        The data type of the output array. If None, use float64.

    """
    from .array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    if stop is None:
        return Array._new(ak.arange(0, start, step, dtype=dtype))  # type: ignore
    else:
        return Array._new(ak.arange(start, stop, step, dtype=dtype))  # type: ignore


def empty(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Return a new array of given shape and type, without initializing entries.
    """
    from .array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    if isinstance(shape, tuple):
        size = 1
        tshape = cast(Tuple, shape)
        for s in tshape:
            size *= s
        return Array._new(
            pdarray("__empty__", akdtype(dtype), size, len(tshape), tshape, 0, None),
            empty=True,
        )
    else:
        vshape = cast(int, shape)
        return Array._new(
            pdarray("__empty__", akdtype(dtype), vshape, 1, (vshape,), 0, None),
            empty=True,
        )


def empty_like(x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    """
    Return a new array whose shape and dtype match the input array, without initializing entries.
    """
    from .array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    t = x.dtype if dtype is None else akdtype(dtype)

    return Array._new(
        pdarray(
            "__empty__",
            t,
            x._array.size,
            x._array.ndim,
            x._array.shape,
            x._array.itemsize,
            x._array.max_bits,
        ),
        empty=True,
    )


def eye(
    N: int,
    M: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Return a 2D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N: int
        Number of rows in the output.
    M: Optional[int]
        Number of columns in the output. If None, defaults to N.
    k: int
        Index of the diagonal: 0 (the default) refers to the main diagonal, a
        positive value refers to an upper diagonal, and a negative value to a
        lower diagonal.
    dtype: Optional[Dtype]
        Data type of the returned array. If None, use float64.
    """
    from .array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    if M is None:
        M = N

    from arkouda import dtype as akdtype

    return Array._new(ak.eye(N=N, M=M, k=k, dt=akdtype(dtype)))


def from_dlpack(x: object, /) -> Array:
    """
    Construct an Array from a DLPack tensor.

    WARNING: This function is not yet implemented.
    """
    raise ValueError("Not implemented")


def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, bool, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Return a new array of given shape and type, filled with `fill_value`.
    """
    a = zeros(shape, dtype=dtype, device=device)
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
    """
    Return a new array whose shape and dtype match the input array, filled with `fill_value`.
    """
    return full(x.shape, fill_value, dtype=dtype, device=device)


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
    """
    An Array API compliant wrapper for :func:`arkouda.linspace`.
    """
    from .array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    return Array._new(ak.linspace(start, stop, num))


def meshgrid(*arrays: Array, indexing: str = "xy") -> List[Array]:
    """
    Return coordinate matrices from coordinate vectors.

    WARNING: This function is not yet implemented.
    """
    raise ValueError("Not implemented")


def ones(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Return a new array with the specified shape and type, filled with ones.
    """
    a = zeros(shape, dtype=dtype, device=device)
    a._array.fill(1)
    return a


def ones_like(x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    """
    Return a new array whose shape and dtype match the input array, filled with ones.
    """
    return ones(x.shape, dtype=dtype, device=device)


def tril(x: Array, /, *, k: int = 0) -> Array:
    """
    Create a new array with the values from `x` below the `k`-th diagonal, and
    all other elements zero.
    """
    from .array_object import Array

    return Array._new(ak.tril(x._array, diag=k))


def triu(x: Array, /, *, k: int = 0) -> Array:
    """
    Create a new array with the values from `x` above the `k`-th diagonal, and
    all other elements zero.
    """
    from .array_object import Array

    return Array._new(ak.triu(x._array, k))


def zeros(
    shape: Union[int, Tuple[int, ...]],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Return a new array with the specified shape and type, filled with zeros.
    """
    from .array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    return_dtype = akdtype(dtype)
    if dtype is None:
        return_dtype = akdtype(ak.float64)

    return Array._new(ak.zeros(shape, return_dtype))


def zeros_like(x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    """
    Return a new array whose shape and dtype match the input array, filled with zeros.
    """
    return zeros(x.shape, dtype=dtype, device=device)
