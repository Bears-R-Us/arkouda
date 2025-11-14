from __future__ import annotations

from typing import Optional, Tuple, Union

import arkouda as ak

from arkouda.numpy.pdarrayclass import create_pdarray
from arkouda.numpy.pdarraycreation import scalar_array

from .array_object import Array
from .manipulation_functions import reshape
from .statistical_functions import sum


__all__ = [
    "all",
    "any",
    "clip",
    "diff",
    "pad",
    "trapezoid",
    "trapz",
]


def all(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Check whether all elements of an array evaluate to True along a given axis.

    Parameters
    ----------
    x : Array
        The array to check for all True values
    axis : int or Tuple[int], optional
        The axis or axes along which to check for all True values. If None, check all elements.
    keepdims : bool, optional
        Whether to keep the singleton dimensions along `axis` in the result.
    """
    return Array._new(scalar_array(ak.all(x._array)))


def any(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Check whether any elements of an array evaluate to True along a given axis.

    Parameters
    ----------
    x : Array
        The array to check for any True values
    axis : int or Tuple[int], optional
        The axis or axes along which to check for any True values. If None, check all elements.
    keepdims : bool, optional
        Whether to keep the singleton dimensions along `axis` in the result.
    """
    return Array._new(scalar_array(ak.any(x._array)))


def clip(a: Array, a_min, a_max, /) -> Array:
    """
    Clip (limit) the values in an array to a given range.

    Parameters
    ----------
    a : Array
        The array to clip
    a_min : scalar
        The minimum value
    a_max : scalar
        The maximum value
    """
    from arkouda.client import generic_msg

    if a.dtype == ak.bigint or a.dtype == ak.bool_:
        raise RuntimeError(f"Error executing command: clip does not support dtype {a.dtype}")

    return Array._new(
        create_pdarray(
            generic_msg(
                cmd=f"clip<{a.dtype},{a.ndim}>",
                args={
                    "x": a._array,
                    "min": a_min,
                    "max": a_max,
                },
            ),
        )
    )


def diff(a: Array, /, n: int = 1, axis: int = -1, prepend=None, append=None) -> Array:
    """
    Calculate the n-th discrete difference along the given axis.

    Parameters
    ----------
    a : Array
        The array to calculate the difference
    n : int, optional
        The order of the finite difference. Default is 1.
    axis : int, optional
        The axis along which to calculate the difference. Default is the last axis.
    prepend : Array, optional
        Array to prepend to `a` along `axis` before calculating the difference.
    append : Array, optional
        Array to append to `a` along `axis` before calculating the difference.

    Returns
    -------
    Array
        The n-th differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as the type of the difference
        between any two elements of `a`. This is the same as the type of
        `a` in most cases. A notable exception is `datetime64`, which
        results in a `timedelta64` output array.

    Notes
    -----
    Type is preserved for boolean arrays, so the result will contain
    `False` when consecutive elements are the same and `True` when they
    differ.

    For unsigned integer arrays, the results will also be unsigned. This
    should not be surprising, as the result is consistent with
    calculating the difference directly.

    If this is not desirable, then the array should be cast to a larger
    integer type first:

    Examples
    --------
    >>> import arkouda as ak
    >>> import arkouda.array_api as xp
    >>> x = xp.asarray(ak.array([1, 2, 4, 7, 0]))
    >>> xp.diff(x)
    Arkouda Array ((4,), int64)[1 2 3 -7]
    >>> xp.diff(x, n=2)
    Arkouda Array ((3,), int64)[1 1 -10]

    >>> x = xp.asarray(ak.array([[1, 3, 6, 10], [0, 5, 6, 8]]))
    >>> xp.diff(x)
    Arkouda Array ((2, 3), int64)[[2 3 4] [5 1 2]]
    >>> xp.diff(x, axis=0)
    Arkouda Array ((1, 4), int64)[[-1 2 0 -2]]

    """
    from arkouda.numpy.pdarrayclass import diff

    if a.dtype == ak.bigint:
        raise RuntimeError(f"Error executing command: diff does not support dtype {a.dtype}")

    return Array._new(diff(a._array, n, axis, prepend, append))


def trapz(y: Array, x: Optional[Array] = None, dx: Optional[float] = 1.0, axis: int = -1) -> Array:
    r"""
    Integrate along the given axis using the composite trapezoidal rule.

    If `x` is provided, the integration happens in sequence along its
    elements - they are not sorted.

    Integrate `y` (`x`) along each 1d slice on the given axis, compute
    :math:`\int y(x) dx`.
    When `x` is specified, this integrates along the parametric curve,
    computing :math:`\int_t y(t) dt =
    \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`.

    See https://numpy.org/doc/1.26/reference/generated/numpy.trapz.html#numpy.trapz

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    -------
    Array
        Definite integral of `y` = n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,
        then the result is a float. If `n` is greater than 1, then the result
        is an `n`-1 dimensional array.

    Notes
    -----
    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
    will be taken from `y` array, by default x-axis distances between
    points will be 1.0, alternatively they can be provided with `x` array
    or with `dx` scalar.  Return value will be equal to combined area under
    the red lines.


    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule

    .. [2] Illustration image:
           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

    Examples
    --------
    >>> from arkouda import array_api as xp
    >>> y = xp.asarray(ak.array([1, 2, 3]))

    Use the trapezoidal rule on evenly spaced points:
    >>> xp.trapz(y)
    Arkouda Array ((), float64)4.0

    The spacing between sample points can be selected by either the
    ``x`` or ``dx`` arguments:

    >>> x = xp.asarray(ak.array([4, 6, 8]))
    >>> xp.trapz(y, x)
    Arkouda Array ((), float64)8.0
    >>> xp.trapz(y, dx=2.0)
    Arkouda Array ((), float64)8.0

    Using a decreasing ``x`` corresponds to integrating in reverse:

    >>> x = xp.asarray(ak.array([8, 6, 4]))
    >>> xp.trapz(y, x)
    Arkouda Array ((), float64)-8.0

    More generally ``x`` is used to integrate along a parametric curve. We can
    estimate the integral :math:`\int_0^1 x^2 = 1/3` using:

    >>> x = xp.linspace(0, 1, num=50)
    >>> y = x**2
    >>> xp.trapz(y, x)
    Arkouda Array ((), float64)0.333402748854643...

    Or estimate the area of a circle, noting we repeat the sample which closes
    the curve:

    >>> theta = xp.linspace(0, 2 * xp.pi, num=1000, endpoint=True)
    >>> xp.trapz(xp.cos(theta), x=xp.sin(theta))
    Arkouda Array ((), float64)3.14157194137584...

    ``np.trapz`` can be applied along a specified axis to do multiple
    computations in one call:

    >>> a = xp.asarray(ak.arange(6).reshape(2, 3))
    >>> a
    Arkouda Array ((2, 3), int64)[[0 1 2] [3 4 5]]
    >>> xp.trapz(a, axis=0)
    Arkouda Array ((3,), float64)[1.5 2.5 3.5]
    >>> xp.trapz(a, axis=1)
    Arkouda Array ((2,), float64)[2.0 8.0]

    """
    # Implementation is the same as Numpy's implementation of trapezoid
    # Modified slightly to fit Arkouda
    # https://github.com/numpy/numpy/blob/d35cd07ea997f033b2d89d349734c61f5de54b0d/numpy/lib/function_base.py#L4857-L4984

    from arkouda.numpy.util import _integer_axis_validation

    if y.dtype == ak.bigint:
        raise RuntimeError(f"Error executing command: trapz does not support dtype {y.dtype}")

    nd = y.ndim
    valid, axis_ = _integer_axis_validation(axis, nd)
    if not valid:
        raise IndexError(f"{axis} is not a valid axis for array of rank {nd}")

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis_] = slice(1, None)
    slice2[axis_] = slice(None, -1)

    if x is None:
        if dx is None:
            raise ValueError("dx cannot be None when x is None for trapz")
        ret = sum(dx * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis=axis_)
    else:
        if x.dtype == ak.bigint:
            raise RuntimeError(f"Error executing command: trapz does not support dtype {x.dtype}")
        d = diff(x, axis=axis_)
        if x.ndim == 1:
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = reshape(d, tuple(shape))

        ret = sum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis=axis)

    return ret


def trapezoid(y: Array, x: Optional[Array] = None, dx: Optional[float] = 1.0, axis: int = -1) -> Array:
    return trapz(y, x, dx, axis)


def pad(
    array: Array,
    pad_width,  # Union[int, Tuple[int, int], Tuple[Tuple[int, int], ...]]
    mode="constant",
    **kwargs,
) -> Array:
    """
    Pad an array.

    Parameters
    ----------
    array : Array
        The array to pad
    pad_width : int or Tuple[int, int] or Tuple[Tuple[int, int], ...]
        Number of values padded to the edges of each axis. If a single int, the same value is used for
        all axes. If a tuple of two ints, those values are used for all axes. If a tuple of tuples, each
        inner tuple specifies the number of values padded to the beginning and end of each axis.
    mode : str, optional
        Padding mode. Only 'constant' is currently supported. Use the `constant_values` keyword argument
        to specify the padding value or values (in the same format as `pad_width`).
    """
    from arkouda.client import generic_msg

    if mode != "constant":
        raise NotImplementedError(f"pad mode '{mode}' is not supported")

    if array.dtype == ak.bigint:
        raise RuntimeError("Error executing command: pad does not support dtype bigint")

    if "constant_values" not in kwargs:
        cvals = 0
    else:
        cvals = kwargs["constant_values"]

    if isinstance(pad_width, int):
        pad_widths_b = [pad_width] * array.ndim
        pad_widths_a = [pad_width] * array.ndim
    elif isinstance(pad_width, tuple):
        if isinstance(pad_width[0], int):
            pad_widths_b = [pad_width[0]] * array.ndim
            pad_widths_a = [pad_width[1]] * array.ndim
        elif isinstance(pad_width[0], tuple):
            pad_widths_b = [pw[0] for pw in pad_width]
            pad_widths_a = [pw[1] for pw in pad_width]

    if isinstance(cvals, int):
        pad_vals_b = [cvals] * array.ndim
        pad_vals_a = [cvals] * array.ndim
    elif isinstance(cvals, tuple):
        if isinstance(cvals[0], int):
            pad_vals_b = [cvals[0]] * array.ndim
            pad_vals_a = [cvals[1]] * array.ndim
        else:
            pad_vals_b = [cv[0] for cv in cvals]
            pad_vals_a = [cv[1] for cv in cvals]

    return Array._new(
        create_pdarray(
            generic_msg(
                cmd=f"pad<{array.dtype},{array.ndim}>",
                args={
                    "name": array._array,
                    "padWidthBefore": tuple(pad_widths_b),
                    "padWidthAfter": tuple(pad_widths_a),
                    "padValsBefore": pad_vals_b,
                    "padValsAfter": pad_vals_a,
                },
            ),
        )
    )
