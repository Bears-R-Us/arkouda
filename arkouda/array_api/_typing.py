"""
Defines the types for type annotations.

These names aren't part of the module namespace, but they are used in the
annotations in the function signatures. The functions in the module are only
valid for inputs that match the given type annotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, TypeVar, Union

import numpy as np


if TYPE_CHECKING:
    from .array_object import Array


__all__ = [
    "Array",
    "Device",
    "Dtype",
    "SupportsDLPack",
    "SupportsBufferProtocol",
    "PyCapsule",
]


_T_co = TypeVar("_T_co", covariant=True)


class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]: ...

    def __len__(self, /) -> int: ...


Device = Literal["cpu"]


Dtype: TypeAlias = Union[
    np.dtype[np.int8],
    np.dtype[np.int16],
    np.dtype[np.int32],
    np.dtype[np.int64],
    np.dtype[np.uint8],
    np.dtype[np.uint16],
    np.dtype[np.uint32],
    np.dtype[np.uint64],
    np.dtype[np.float32],
    np.dtype[np.float64],
    np.dtype[np.complex64],
    np.dtype[np.complex128],
    np.dtype[np.bool_],
]


SupportsBufferProtocol = Any
PyCapsule = Any


class SupportsDLPack(Protocol):
    def __dlpack__(self, /, *, stream: None = ...) -> PyCapsule: ...
