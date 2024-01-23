from __future__ import annotations

from ._array_object import Array

from typing import List, Optional, Tuple, Union

import arkouda as ak
from arkouda.pdarrayclass import broadcast_to_shape


def broadcast_arrays(*arrays: Array) -> List[Array]:
    """
    Array API compatible wrapper for :py:func:`np.broadcast_arrays <numpy.broadcast_arrays>`.

    See its docstring for more information.
    """
    raise ValueError("broadcast_arrays not implemented")


def broadcast_to(x: Array, /, shape: Tuple[int, ...]) -> Array:
    """
    Broadcast the array to the specified shape.
    """
    return Array._new(broadcast_to_shape(x._array, shape))


# Note: the function name is different here
def concat(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: Optional[int] = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.concatenate <numpy.concatenate>`.

    See its docstring for more information.
    """
    # Note: Casting rules here are different from the np.concatenate default
    # (no for scalars with axis=None, no cross-kind casting)
    return Array._new(ak.concatenate(arrays))


def expand_dims(x: Array, /, *, axis: int) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.expand_dims <numpy.expand_dims>`.

    See its docstring for more information.
    """
    # return Array._new(np.expand_dims(x._array, axis))
    raise ValueError("expand_dims not implemented")


def flip(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.flip <numpy.flip>`.

    See its docstring for more information.
    """
    # return Array._new(np.flip(x._array, axis=axis))
    raise ValueError("flip not implemented")


# Note: The function name is different here (see also matrix_transpose).
# Unlike transpose(), the axes argument is required.
def permute_dims(x: Array, /, axes: Tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.transpose <numpy.transpose>`.

    See its docstring for more information.
    """
    # return Array._new(ak.transpose(x._array, axes))
    raise ValueError("permute_dims not implemented")


# Note: the optional argument is called 'shape', not 'newshape'
def reshape(x: Array, /, shape: Tuple[int, ...], *, copy: Optional[bool] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.reshape <numpy.reshape>`.

    See its docstring for more information.
    """

    # data = x._array
    # if copy:
    #     data = ak.copy(data)

    # reshaped = ak.reshape(data, shape)

    # if copy is False and not ak.shares_memory(data, reshaped):
    #     raise AttributeError("Incompatible shape for in-place modification.")

    # return Array._new(reshaped)
    raise ValueError("reshape not implemented")


def roll(
    x: Array,
    /,
    shift: Union[int, Tuple[int, ...]],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.roll <numpy.roll>`.

    See its docstring for more information.
    """
    # return Array._new(np.roll(x._array, shift, axis=axis))
    raise ValueError("roll not implemented")


def squeeze(x: Array, /, axis: Union[int, Tuple[int, ...]]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.squeeze <numpy.squeeze>`.

    See its docstring for more information.
    """
    # return Array._new(np.squeeze(x._array, axis=axis))
    raise ValueError("squeeze not implemented")


def stack(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.stack <numpy.stack>`.

    See its docstring for more information.
    """
    # # Call result type here just to raise on disallowed type combinations
    # result_type(*arrays)
    # arrays = tuple(a._array for a in arrays)
    # return Array._new(np.stack(arrays, axis=axis))
    raise ValueError("stack not implemented")
