from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

from arkouda.numpy import cast as akcast
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.pdarrayclass import create_pdarray
from arkouda.numpy.util import _axis_validation

from ._dtypes import (
    _real_floating_dtypes,
    _real_numeric_dtypes,
    _signed_integer_dtypes,
    float64,
    int64,
    uint64,
)
from ._dtypes import _numeric_dtypes  # _complex_floating_dtypes,; complex128,
from .array_object import Array, implements_numpy
from .manipulation_functions import squeeze


__all__ = [
    "_prod_sum_dtype",
    "cumulative_sum",
    "cumulative_prod",
    "max",
    "mean",
    "mean_shim",
    "min",
    "prod",
    "std",
    "sum",
    "var",
]


if TYPE_CHECKING:
    from ._typing import Dtype


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

    Returns
    -------
    Array
        An array with the maximum values along the given axis, or a one-element
        array with the maximum value, if no axis is given.

    Raises
    ------
    TypeError
        Raised if x is not real numeric.

    Examples
    --------
    >>> import arkouda as ak
    >>> import arkouda.array_api as xp
    >>> a = xp.asarray(ak.arange(10,dtype=ak.float64))
    >>> xp.max(a)
    Arkouda Array ((), float64)9.0
    >>> a = xp.asarray(ak.arange(10,dtype=ak.float64).reshape(2,5))
    >>> xp.max(a,axis=0)
    Arkouda Array ((5,), float64)[5.0 6.0 7.0 8.0 9.0]
    >>> xp.max(a,axis=1)
    Arkouda Array ((2,), float64)[4.0 9.0]

    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in max")

    from arkouda import max as ak_max

    return Array._new(ak_max(x._array, axis=axis, keepdims=keepdims))


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

    Returns
    -------
    Array
        The mean calculated from the pda sum and size, along the axis/axes if
        those are given.

    Raises
    ------
    IndexError
        Raised if axis is not valid for the given array.
    TypeError
        Raised if x is not real numeric.

    Examples
    --------
    >>> import arkouda as ak
    >>> import arkouda.array_api as xp
    >>> a = xp.asarray(ak.arange(10,dtype=ak.float64))
    >>> xp.mean(a)
    Arkouda Array ((1,), float64)[4.5]
    >>> a = xp.asarray(ak.arange(10,dtype=ak.float64).reshape(2,5))
    >>> xp.mean(a,axis=0)
    Arkouda Array ((5,), float64)[2.5 3.5 4.5 5.5 6.5]
    >>> xp.mean(a,axis=1)
    Arkouda Array ((2,), float64)[2.0 7.0]

    """
    from arkouda.client import generic_msg

    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in mean")

    valid, axis_ = _axis_validation(axis, x.ndim)

    if not valid:
        raise IndexError(f"axis = {axis} is not valid for rank {x.ndim}")

    axis_ = axis_ if axis_ is not None else list(range(x.ndim))

    arr = Array._new(
        create_pdarray(
            generic_msg(
                cmd=f"mean<{x.dtype},{x.ndim}>",
                args={
                    "x": x._array,
                    "axis": axis_,
                    "skipNan": True,  # TODO: handle all-nan slices
                },
            )
        )
    )

    if keepdims or axis is None:
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

    Returns
    -------
    Array
        An array with the minimum values along the given axis, or a one-element
        array with the minimum value, if no axis is given.

    Raises
    ------
    TypeError
        Raised if x is not real numeric.

    Examples
    --------
    >>> import arkouda as ak
    >>> import arkouda.array_api as xp
    >>> a = xp.asarray(ak.arange(10,dtype=ak.float64))
    >>> xp.min(a)
    Arkouda Array ((), float64)0.0
    >>> a = xp.asarray(ak.arange(10,dtype=ak.float64).reshape(2,5))
    >>> xp.min(a,axis=0)
    Arkouda Array ((5,), float64)[0.0 1.0 2.0 3.0 4.0]
    >>> xp.min(a,axis=1)
    Arkouda Array ((2,), float64)[0.0 5.0]
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in min")

    from arkouda import min as ak_min

    return Array._new(ak_min(x._array, axis=axis, keepdims=keepdims))


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

    Returns
    -------
    Array
        An array with the product along the given axis, or a one-element
        array with the product of the entire array, if no axis is given.

    Raises
    ------
    TypeError
        Raised if x._array is not real numeric, or can't be cast to a pdarray.

    Examples
    --------
    >>> import arkouda as ak
    >>> import arkouda.array_api as xp
    >>> a = xp.asarray(1 + ak.arange(10,dtype=ak.float64))
    >>> xp.prod(a)
    Arkouda Array ((), float64)3628800.0
    >>> a = xp.asarray((1 + ak.arange(10,dtype=ak.float64)).reshape(2,5))
    >>> xp.prod(a,axis=0)
    Arkouda Array ((5,), float64)[6.0 14.0 24.0 36.0 50.0]
    >>> xp.prod(a,axis=1)
    Arkouda Array ((2,), float64)[120.0 30240.0]

    """
    from arkouda import prod as ak_prod
    from arkouda.numpy import pdarray

    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in prod")

    cast_to = _prod_sum_dtype(x.dtype) if dtype is None else dtype
    x_op = akcast(x._array, cast_to) if cast_to != x.dtype else x._array

    if not isinstance(x_op, pdarray):
        raise TypeError(f"Expected pdarray, got {type(x_op)}")

    return Array._new(ak_prod(x_op, axis=axis, keepdims=keepdims))


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

    Returns
    -------
    Array
        An array with the standard deviation along the given axis, or a one-element
        array with the std of the entire array, if no axis is given.

    Raises
    ------
    TypeError
        Raised if x is not floating point.
    ValueError
        Raised if correction is negative.
    IndexError
        Raised if axis is not valid for given Array.

    Examples
    --------
    >>> import arkouda as ak
    >>> import arkouda.array_api as xp
    >>> a = xp.asarray((1 + ak.arange(10,dtype=ak.float64)))
    >>> xp.std(a)
    Arkouda Array ((1,), float64)[2.87228]
    >>> a = xp.asarray((1 + ak.arange(10,dtype=ak.float64)).reshape(2,5))
    >>> xp.std(a,axis=0)
    Arkouda Array ((5,), float64)[2.5 2.5 2.5 2.5 2.5]
    >>> xp.std(a,axis=1)
    Arkouda Array ((2,), float64)[1.41421 1.41421]

    """
    from arkouda.client import generic_msg

    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in std")
    if correction < 0:
        raise ValueError("Correction must be non-negative in std")

    valid, axis_ = _axis_validation(axis, x.ndim)

    if not valid:
        raise IndexError(f"axis = {axis} is not valid for rank {x.ndim}")

    axis_ = axis_ if axis_ is not None else list(range(x.ndim))

    arr = Array._new(
        create_pdarray(
            generic_msg(
                cmd=f"stdReduce<{x.dtype},{x.ndim}>",
                args={
                    "x": x._array,
                    "ddof": correction,
                    "axis": axis_,
                    "skipNan": True,
                },
            )
        )
    )

    if keepdims or axis is None:
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

    Returns
    -------
    Array
        An array with the sum along the given axis, or a one-element
        array with the sum of the entire array, if no axis is given.

    Raises
    ------
    TypeError
        Raised if x is not numeric.

    Examples
    --------
    >>> import arkouda as ak
    >>> import arkouda.array_api as xp
    >>> a = xp.asarray(1 + ak.arange(10,dtype=ak.float64))
    >>> xp.sum(a)
    Arkouda Array ((), float64)55.0
    >>> a = xp.asarray((1 + ak.arange(10,dtype=ak.float64)).reshape(2,5))
    >>> xp.sum(a,axis=0)
    Arkouda Array ((5,), float64)[7.0 9.0 11.0 13.0 15.0]
    >>> xp.sum(a,axis=1)
    Arkouda Array ((2,), float64)[15.0 40.0]

    """
    from arkouda import sum as ak_sum
    from arkouda.numpy import pdarray

    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sum")

    cast_to = _prod_sum_dtype(x.dtype) if dtype is None else dtype
    x_op = akcast(x._array, cast_to) if cast_to != x.dtype else x._array

    if not isinstance(x_op, pdarray):
        raise TypeError(f"Expected pdarray, got {type(x_op)}")

    return Array._new(ak_sum(x_op, axis=axis, keepdims=keepdims))


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

    Returns
    -------
    Array
        An array with the variance along the given axis, or a one-element
        array with the var of the entire array, if no axis is given.

    Raises
    ------
    TypeError
        Raised if x is not floating point.
    ValueError
        Raised if correction is negative.
    IndexError
        Raised if axis is not valid for given Array.

    Examples
    --------
    >>> import arkouda as ak
    >>> import arkouda.array_api as xp
    >>> a = xp.asarray((1 + ak.arange(10,dtype=ak.float64)))
    >>> xp.var(a)
    Arkouda Array ((1,), float64)[8.25]
    >>> a = xp.asarray((1 + ak.arange(10,dtype=ak.float64)).reshape(2,5))
    >>> xp.var(a,axis=0)
    Arkouda Array ((5,), float64)[6.25 6.25 6.25 6.25 6.25]
    >>> xp.var(a,axis=1)
    Arkouda Array ((2,), float64)[2.0 2.0]

    """
    from arkouda.client import generic_msg

    # Note: the keyword argument correction is different here
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in var")
    if correction < 0:
        raise ValueError("Correction must be non-negative in std")

    valid, axis_ = _axis_validation(axis, x.ndim)

    if not valid:
        raise IndexError(f"axis = {axis} is not valid for rank {x.ndim}")

    axis_ = axis_ if axis_ is not None else list(range(x.ndim))

    arr = Array._new(
        create_pdarray(
            generic_msg(
                cmd=f"varReduce<{x.dtype},{x.ndim}>",
                args={
                    "x": x._array,
                    "ddof": correction,
                    "axis": axis_,
                    "skipNan": True,
                },
            )
        )
    )

    if keepdims or axis is None:
        return arr
    else:
        return squeeze(arr, axis)


def _prod_sum_dtype(dtype: Dtype) -> Dtype:
    if dtype == uint64:
        return akdtype(dtype)
    elif dtype in _real_floating_dtypes:
        return akdtype(float64)
    # elif dtype in _complex_floating_dtypes:
    #     return complex128
    elif dtype in _signed_integer_dtypes:
        return akdtype(int64)
    else:
        return akdtype(uint64)


def cumulative_sum(
    x: Array,
    /,
    *,
    axis: Optional[int] = None,
    dtype: Optional[Dtype] = None,
    include_initial: bool = False,
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

    Returns
    -------
    Array
        A new array holding the result of the cumulative sum along the given axis.

    Raises
    ------
    ValueError
        Raised if x is multi-dim and no axis was supplied.
    IndexError
        Raised if axis is invalid for the given array.

    Examples
    --------
    >>> import arkouda as ak
    >>> import arkouda.array_api as xp
    >>> a = xp.asarray((1 + ak.arange(10,dtype=ak.float64)).reshape(2,5))
    >>> xp.cumulative_sum(a,axis=0,include_initial=True)
    Arkouda Array ((3, 5), float64)[[0.0 0.0 0.0 0.0 0.0] [1.0 2.0 3.0 4.0 5.0] [7.0 9.0 11.0 13.0 15.0]]
    >>> xp.cumulative_sum(a,axis=1,include_initial=False)
    Arkouda Array ((2, 5), float64)[[1.0 3.0 6.0 10.0 15.0] [6.0 13.0 21.0 30.0 40.0]]
    """
    from arkouda.client import generic_msg
    from arkouda.numpy.pdarrayclass import _axis_validation

    x_ = x._array

    if axis is None:
        if x.ndim != 1:
            raise ValueError("multi-dim args to cumulative_sum must supply an axis.")
        else:
            axis = 0

    valid, axis_ = _axis_validation(axis, x.ndim)

    if not valid:
        raise ValueError(f"axis {axis} is not valid for the given array.")

    if dtype is None:
        if x_.dtype == "bool":
            x_ = akcast(x_, int)
    else:
        x_ = akcast(x_, dtype)

    resp = generic_msg(
        cmd=f"cumSum<{x_.dtype},{x.ndim}>",
        args={
            "x": x_,
            "axis": axis_,
            "includeInitial": include_initial,
        },
    )

    return Array._new(create_pdarray(resp))


def cumulative_prod(
    x: Array,
    /,
    *,
    axis: Optional[int] = None,
    dtype: Optional[Dtype] = None,
    include_initial: bool = False,
) -> Array:
    """
    Compute the cumulative product of the elements of an array along a given axis.

    Parameters
    ----------
    x : Array
        The array to compute the cumulative product of
    axis : int, optional
        The axis along which to compute the cumulative product. If x is 1D, this argument is optional,
        otherwise it is required.
    dtype : Dtype, optional
        The dtype of the returned array. If None, the dtype of the input array is used.
    include_initial : bool, optional
        Whether to include the initial value as the first element of the output.

    Returns
    -------
    Array
        A new array holding the result of the cumulative prod along the given axis.

    Raises
    ------
    ValueError
        Raised if x is multi-dim and no axis was supplied.
    IndexError
        Raised if axis is invalid for the given array.

    Examples
    --------
    >>> import arkouda as ak
    >>> import arkouda.array_api as xp
    >>> a = xp.asarray((1 + ak.arange(6,dtype=ak.float64).reshape(2,3)))
    >>> xp.cumulative_prod(a,axis=0,include_initial=True)
    Arkouda Array ((3, 3), float64)[[1.0 1.0 1.0] [1.0 2.0 3.0] [4.0 10.0 18.0]]
    >>> xp.cumulative_prod(a,axis=1,include_initial=False)
    Arkouda Array ((2, 3), float64)[[1.0 2.0 6.0] [4.0 20.0 120.0]]
    """
    from arkouda.client import generic_msg
    from arkouda.numpy.pdarrayclass import _axis_validation

    x_ = x._array

    if axis is None:
        if x.ndim != 1:
            raise ValueError("multi-dim args to cumulative_prod must supply an axis.")
        else:
            axis = 0

    valid, axis_ = _axis_validation(axis, x.ndim)

    if not valid:
        raise ValueError(f"axis {axis} is not valid for the given array.")

    if dtype is None:
        if x_.dtype == "bool":
            x_ = akcast(x_, int)
    else:
        x_ = akcast(x_, dtype)

    resp = generic_msg(
        cmd=f"cumProd<{x_.dtype},{x.ndim}>",
        args={
            "x": x_,
            "axis": axis_,
            "includeInitial": include_initial,
        },
    )

    return Array._new(create_pdarray(resp))
