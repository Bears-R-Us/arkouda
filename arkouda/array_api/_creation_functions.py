from __future__ import annotations


from typing import TYPE_CHECKING, List, Optional, Tuple, Union

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
    copy: Optional[Union[bool, ak._CopyMode]] = None,
) -> Array:
    raise ValueError(f"Not implemented")


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
    return Array._new(ak.arange(start, stop))


def empty(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    raise ValueError(f"Not implemented")


def empty_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    raise ValueError(f"Not implemented")


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    raise ValueError(f"Not implemented")


def from_dlpack(x: object, /) -> Array:
    raise ValueError(f"Not implemented")


def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    raise ValueError(f"Not implemented")


def full_like(
    x: Array,
    /,
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    raise ValueError(f"Not implemented")


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
    raise ValueError(f"Not implemented")


def meshgrid(*arrays: Array, indexing: str = "xy") -> List[Array]:
    raise ValueError(f"Not implemented")


def ones(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    raise ValueError(f"Not implemented")


def ones_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    raise ValueError(f"Not implemented")


def tril(x: Array, /, *, k: int = 0) -> Array:
    raise ValueError(f"Not implemented")


def triu(x: Array, /, *, k: int = 0) -> Array:
    raise ValueError(f"Not implemented")

def zeros(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    raise ValueError(f"Not implemented")


def zeros_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    raise ValueError(f"Not implemented")
