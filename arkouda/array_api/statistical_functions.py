from __future__ import annotations

from ._dtypes import (
    _real_floating_dtypes,
    _real_numeric_dtypes,
    _numeric_dtypes,
    # _complex_floating_dtypes,
    _signed_integer_dtypes,
    uint64,
    int64,
    float64,
    # complex128,
)
from .array_object import Array, implements_numpy
from .manipulation_functions import squeeze

from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from ._typing import Dtype

from arkouda.numeric import cast as akcast
from arkouda.client import generic_msg
from arkouda.pdarrayclass import parse_single_value, create_pdarray
from arkouda.pdarraycreation import scalar_array
import numpy as np


def max(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Compute the maximum values of an array along a given axis or axes.

    Parameters
    ----------
    x : Array
        The array to compute the maximum of
    axis : int or Tuple[int, ...], optional
        The axis or axes along which to compute the maximum values. If None, the maximum value of the
        entire array is computed (returning a scalar-array).
    keepdims : bool, optional
        Whether to keep the singleton dimension(s) along `axis` in the result.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in max")

    axis_list = []
    if axis is not None:
        axis_list = list(axis) if isinstance(axis, tuple) else [axis]

    resp = generic_msg(
        cmd=f"reduce{x.ndim}D",
        args={
            "x": x._array,
            "op": "max",
            "nAxes": len(axis_list),
            "axis": axis_list,
            "skipNan": True,
        },
    )

    if axis is None or x.ndim == 1:
        return Array._new(scalar_array(parse_single_value(resp)))
    else:
        arr = Array._new(create_pdarray(resp))

        if keepdims:
            return arr
        else:
            return squeeze(arr, axis)


# this is a temporary fix to get mean working with XArray
# (until a counterpart to np.nanmean is added to the array API
# see: https://github.com/data-apis/array-api/issues/621)
@implements_numpy(np.nanmean)
@implements_numpy(np.mean)
def mean_shim(x: Array, axis=None, dtype=None, out=None, keepdims=False):
    return mean(x, axis=axis, keepdims=keepdims)


def mean(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Compute the minimum values of an array along a given axis or axes.

    Parameters
    ----------
    x : Array
        The array to compute the minimum of
    axis : int or Tuple[int, ...], optional
        The axis or axes along which to compute the mean. If None, the mean of the entire array is
        computed (returning a scalar-array).
    keepdims : bool, optional
        Whether to keep the singleton dimension(s) along `axis` in the result.
    """
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in mean")

    axis_list = []
    if axis is not None:
        axis_list = list(axis) if isinstance(axis, tuple) else [axis]

    resp = generic_msg(
        cmd=f"stats{x.ndim}D",
        args={
            "x": x._array,
            "comp": "mean",
            "nAxes": len(axis_list),
            "axis": axis_list,
            "ddof": 0,
            "skipNan": True,  # TODO: handle all-nan slices
        },
    )

    if axis is None or x.ndim == 1:
        return Array._new(scalar_array(parse_single_value(resp)))
    else:
        arr = Array._new(create_pdarray(resp))

        if keepdims:
            return arr
        else:
            return squeeze(arr, axis)


def min(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Compute the mean of an array along a given axis or axes.

    Parameters
    ----------
    x : Array
        The array to compute the mean of
    axis : int or Tuple[int, ...], optional
        The axis or axes along which to compute the minimum values. If None, the minimum of the entire
        array is computed (returning a scalar-array).
    keepdims : bool, optional
        Whether to keep the singleton dimension(s) along `axis` in the result.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in min")

    axis_list = []
    if axis is not None:
        axis_list = list(axis) if isinstance(axis, tuple) else [axis]

    resp = generic_msg(
        cmd=f"reduce{x.ndim}D",
        args={
            "x": x._array,
            "op": "min",
            "nAxes": len(axis_list),
            "axis": axis_list,
            "skipNan": True,
        },
    )

    if axis is None or x.ndim == 1:
        return Array._new(scalar_array(parse_single_value(resp)))
    else:
        arr = Array._new(create_pdarray(resp))

        if keepdims:
            return arr
        else:
            return squeeze(arr, axis)


def prod(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Dtype] = None,
    keepdims: bool = False,
) -> Array:
    """
    Compute the product of an array along a given axis or axes.

    Parameters
    ----------
    x : Array
        The array to compute the product of
    axis : int or Tuple[int, ...], optional
        The axis or axes along which to compute the product. If None, the product of the entire array
        is computed (returning a scalar-array).
    dtype : Dtype, optional
        The dtype of the returned array. If None, the dtype of the input array is used.
    keepdims : bool, optional
        Whether to keep the singleton dimension(s) along `axis` in the result.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in prod")

    axis_list = []
    if axis is not None:
        axis_list = list(axis) if isinstance(axis, tuple) else [axis]

    # cast to the appropriate dtype if necessary
    cast_to = _prod_sum_dtype(x.dtype) if dtype is None else dtype
    if cast_to != x.dtype:
        x_op = akcast(x._array, cast_to)
    else:
        x_op = x._array

    resp = generic_msg(
        cmd=f"reduce{x.ndim}D",
        args={
            "x": x_op,
            "op": "prod",
            "nAxes": len(axis_list),
            "axis": axis_list,
            "skipNan": True,
        },
    )

    if axis is None or x.ndim == 1:
        return Array._new(scalar_array(parse_single_value(resp)))
    else:
        arr = Array._new(create_pdarray(resp))

        if keepdims:
            return arr
        else:
            return squeeze(arr, axis)


# Not working with XArray yet, pending a fix for:
# https://github.com/pydata/xarray/issues/8566#issuecomment-1870472827
def std(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Array:
    """
    Compute the standard deviation of an array along a given axis or axes.

    Parameters
    ----------
    x : Array
        The array to compute the standard deviation of
    axis : int or Tuple[int, ...], optional
        The axis or axes along which to compute the standard deviation. If None, the standard deviation
        of the entire array is computed (returning a scalar-array).
    correction : int or float, optional
        The degrees of freedom correction to apply. The default is 0.
    keepdims : bool, optional
        Whether to keep the singleton dimension(s) along `axis` in the result.
    """
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in std")
    if correction < 0:
        raise ValueError("Correction must be non-negative in std")

    axis_list = []
    if axis is not None:
        axis_list = list(axis) if isinstance(axis, tuple) else [axis]

    resp = generic_msg(
        cmd=f"stats{x.ndim}D",
        args={
            "x": x._array,
            "comp": "std",
            "ddof": correction,
            "nAxes": len(axis_list),
            "axis": axis_list,
            "skipNan": True,
        },
    )

    if axis is None or x.ndim == 1:
        return Array._new(scalar_array(parse_single_value(resp)))
    else:
        arr = Array._new(create_pdarray(resp))

        if keepdims:
            return arr
        else:
            return squeeze(arr, axis)


def sum(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Dtype] = None,
    keepdims: bool = False,
) -> Array:
    """
    Compute the sum of an array along a given axis or axes.

    Parameters
    ----------
    x : Array
        The array to compute the sum of
    axis : int or Tuple[int, ...], optional
        The axis or axes along which to compute the sum. If None, the sum of the entire array is
        computed (returning a scalar-array).
    dtype : Dtype, optional
        The dtype of the returned array. If None, the dtype of the input array is used.
    keepdims : bool, optional
        Whether to keep the singleton dimension(s) along `axis` in the result.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sum")

    axis_list = []
    if axis is not None:
        axis_list = list(axis) if isinstance(axis, tuple) else [axis]

    # cast to the appropriate dtype if necessary
    cast_to = _prod_sum_dtype(x.dtype) if dtype is None else dtype
    if cast_to != x.dtype:
        x_op = akcast(x._array, cast_to)
    else:
        x_op = x._array

    resp = generic_msg(
        cmd=f"reduce{x.ndim}D",
        args={
            "x": x_op,
            "op": "sum",
            "nAxes": len(axis_list),
            "axis": axis_list,
            "skipNan": True,
        },
    )

    if axis is None or x.ndim == 1:
        return Array._new(scalar_array(parse_single_value(resp)))
    else:
        arr = Array._new(create_pdarray(resp))

        if keepdims:
            return arr
        else:
            return squeeze(arr, axis)


# Not working with XArray yet, pending a fix for:
# https://github.com/pydata/xarray/issues/8566#issuecomment-1870472827
def var(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Array:
    """
    Compute the variance of an array along a given axis or axes.

    Parameters
    ----------
    x : Array
        The array to compute the variance of
    axis : int or Tuple[int, ...], optional
        The axis or axes along which to compute the variance. If None, the variance of the entire array
        is computed (returning a scalar-array).
    correction : int or float, optional
        The degrees of freedom correction to apply. The default is 0.
    keepdims : bool, optional
        Whether to keep the singleton dimension(s) along `axis` in the result.
    """
    # Note: the keyword argument correction is different here
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in var")
    if correction < 0:
        raise ValueError("Correction must be non-negative in std")

    axis_list = []
    if axis is not None:
        axis_list = list(axis) if isinstance(axis, tuple) else [axis]

    resp = generic_msg(
        cmd=f"stats{x.ndim}D",
        args={
            "x": x._array,
            "comp": "var",
            "ddof": correction,
            "nAxes": len(axis_list),
            "axis": axis_list,
            "skipNan": True,
        },
    )

    if axis is None or x.ndim == 1:
        return Array._new(scalar_array(parse_single_value(resp)))
    else:
        arr = Array._new(create_pdarray(resp))

        if keepdims:
            return arr
        else:
            return squeeze(arr, axis)


def _prod_sum_dtype(dtype: Dtype) -> Dtype:
    if dtype == uint64:
        return dtype
    elif dtype in _real_floating_dtypes:
        return float64
    # elif dtype in _complex_floating_dtypes:
    #     return complex128
    elif dtype in _signed_integer_dtypes:
        return int64
    else:
        return uint64


def cumulative_sum(
    x: Array, /, *,
    axis: Optional[int] = None,
    dtype: Optional[Dtype] = None,
    include_initial: bool = False
) -> Array:
    """
    Compute the cumulative sum of the elements of an array along a given axis.

    Parameters
    ----------
    x : Array
        The array to compute the cumulative sum of
    axis : int, optional
        The axis along which to compute the cumulative sum. If x is 1D, this argument is optional,
        otherwise it is required.
    dtype : Dtype, optional
        The dtype of the returned array. If None, the dtype of the input array is used.
    include_initial : bool, optional
        Whether to include the initial value as the first element of the output.
    """
    resp = generic_msg(
        cmd=f"cumSum{x.ndim}D",
        args={
            "x": x._array,
            "axis": axis if axis is not None else 0,
            "include_initial": include_initial,
        },
    )

    return Array._new(create_pdarray(resp))
