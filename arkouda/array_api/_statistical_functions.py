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
from ._array_object import Array
from ._manipulation_functions import broadcast_to, squeeze

from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from ._typing import Dtype

from arkouda.numeric import cast as akcast
from arkouda.client import generic_msg
from arkouda.pdarrayclass import parse_single_value


def max(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
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
        },
    )

    if axis is None:
        return Array._new(parse_single_value(resp))
    else:
        arr = Array._new(resp)

        if keepdims:
            # TODO: eventually, when broadcasting / handling of singleton dimensions
            # is implemented more efficiently, this broadcast call won't be necessary

            # see: https://data-apis.org/array-api/latest/API_specification/broadcasting.html
            return broadcast_to(arr, shape=x.shape)
        else:
            return squeeze(arr, axis)


def mean(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
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
        },
    )

    if axis is None:
        return Array._new(parse_single_value(resp))
    else:
        arr = Array._new(resp)

        if keepdims:
            # TODO: eventually remove 'broadcast_to'. See note in max() function
            return broadcast_to(arr, shape=x.shape)
        else:
            return squeeze(arr, axis)


def min(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in min")

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
        },
    )

    if axis is None:
        return Array._new(parse_single_value(resp))
    else:
        arr = Array._new(resp)

        if keepdims:
            # TODO: eventually remove 'broadcast_to'. See note in max() function
            return broadcast_to(arr, shape=x.shape)
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
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in prod")

    axis_list = []
    if axis is not None:
        axis_list = list(axis) if isinstance(axis, tuple) else [axis]

    # cast to the appropriate dtype if necessary
    cast_to = prod_sum_dtype(x.dtype) if dtype is None else dtype
    if cast_to != x.dtype:
        x_op = akcast(x._array, cast_to)
    else:
        x_op = x._array

    resp = generic_msg(
        cmd=f"reduce{x.ndim}D",
        args={
            "x": x_op,
            "op": "max",
            "nAxes": len(axis_list),
            "axis": axis_list,
        },
    )

    if axis is None:
        return Array._new(parse_single_value(resp))
    else:
        arr = Array._new(resp)

        if keepdims:
            # TODO: eventually remove 'broadcast_to'. See note in max() function
            return broadcast_to(arr, shape=x.shape)
        else:
            return squeeze(arr, axis)


def std(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Array:
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
        },
    )

    if axis is None:
        return Array._new(parse_single_value(resp))
    else:
        arr = Array._new(resp)

        if keepdims:
            # TODO: eventually remove 'broadcast_to'. See note in max() function
            return broadcast_to(arr, shape=x.shape)
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
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sum")

    axis_list = []
    if axis is not None:
        axis_list = list(axis) if isinstance(axis, tuple) else [axis]

    # cast to the appropriate dtype if necessary
    cast_to = prod_sum_dtype(x.dtype) if dtype is None else dtype
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
        },
    )

    if axis is None:
        return Array._new(parse_single_value(resp))
    else:
        arr = Array._new(resp)

        if keepdims:
            # TODO: eventually remove 'broadcast_to'. See note in max() function
            return broadcast_to(arr, shape=x.shape)
        else:
            return squeeze(arr, axis)


def var(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Array:
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
        },
    )

    if axis is None:
        return Array._new(parse_single_value(resp))
    else:
        arr = Array._new(resp)

        if keepdims:
            # TODO: eventually remove 'broadcast_to'. See note in max() function
            return broadcast_to(arr, shape=x.shape)
        else:
            return squeeze(arr, axis)


def prod_sum_dtype(dtype: Dtype) -> Dtype:
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
