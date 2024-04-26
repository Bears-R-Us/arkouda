from __future__ import annotations

from ._dtypes import (
    _boolean_dtypes,
    _floating_dtypes,
    _real_floating_dtypes,
    # _complex_floating_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _real_numeric_dtypes,
    _numeric_dtypes,
    _result_type,
)
from ._array_object import Array
import arkouda as ak


def abs(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`ak.abs`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in abs")
    return Array._new(ak.abs(x._array))


# Note: the function name is different here
def acos(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.arccos`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acos")
    return Array._new(ak.arccos(x._array))


# Note: the function name is different here
def acosh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`ak.arccosh`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acosh")
    return Array._new(ak.arccosh(x._array))


def add(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.add <numpy.add>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in add")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array + x2._array)


# Note: the function name is different here
def asin(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`ak.arcsin`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asin")
    return Array._new(ak.arcsin(x._array))


# Note: the function name is different here
def asinh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`ak.arcsinh`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asinh")
    return Array._new(ak.arcsinh(x._array))


# Note: the function name is different here
def atan(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.arctan`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atan")
    return Array._new(ak.arctan(x._array))


# Note: the function name is different here
def atan2(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`ak.arctan2`.

    See its docstring for more information.
    """
    if x1.dtype not in _real_floating_dtypes or x2.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in atan2")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(ak.arctan2(x1._array, x2._array))


# Note: the function name is different here
def atanh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`ak.arctanh`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atanh")
    return Array._new(ak.arctanh(x._array))


def bitwise_and(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.bitwise_and`.

    See its docstring for more information.
    """
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_and")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array & x2._array)


# Note: the function name is different here
def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.left_shift <numpy.left_shift>`.

    See its docstring for more information.
    """
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in bitwise_left_shift")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    # Note: bitwise_left_shift is only defined for x2 nonnegative.
    if ak.any(x2._array < 0):
        raise ValueError("bitwise_left_shift(x1, x2) is only defined for x2 >= 0")
    return Array._new(x1._array << x2._array)


# Note: the function name is different here
def bitwise_invert(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.invert <numpy.invert>`.

    See its docstring for more information.
    """
    raise ValueError("bitwise invert not implemented")


def bitwise_or(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.bitwise_or <numpy.bitwise_or>`.

    See its docstring for more information.
    """
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_or")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array | x2._array)


# Note: the function name is different here
def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.right_shift <numpy.right_shift>`.

    See its docstring for more information.
    """
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in bitwise_right_shift")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    # Note: bitwise_right_shift is only defined for x2 nonnegative.
    if ak.any(x2._array < 0):
        raise ValueError("bitwise_right_shift(x1, x2) is only defined for x2 >= 0")
    return Array._new(x1._array >> x2._array)


def bitwise_xor(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.bitwise_xor <numpy.bitwise_xor>`.

    See its docstring for more information.
    """
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_xor")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array ^ x2._array)


def ceil(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.ceil <numpy.ceil>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in ceil")
    return Array._new(ak.ceil(x._array))


def conj(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.conj <numpy.conj>`.

    See its docstring for more information.
    """
    raise ValueError("conj not implemented - Arkouda does not support complex types")


def cos(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.cos <numpy.cos>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cos")
    return Array._new(ak.cos(x._array))


def cosh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.cosh <numpy.cosh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cosh")
    return Array._new(ak.cosh(x._array))


def divide(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.divide <numpy.divide>`.

    See its docstring for more information.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in divide")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array / x2._array)


def equal(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.equal <numpy.equal>`.

    See its docstring for more information.
    """
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array == x2._array)


def exp(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.exp <numpy.exp>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in exp")
    return Array._new(ak.exp(x._array))


def expm1(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.expm1 <numpy.expm1>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in exp")
    return Array._new(ak.expm1(x._array))


def floor(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.floor <numpy.floor>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in floor")
    return Array._new(ak.floor(x._array))


def floor_divide(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.floor_divide <numpy.floor_divide>`.

    See its docstring for more information.
    """
    raise ValueError("exp not implemented")


def greater(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.greater <numpy.greater>`.

    See its docstring for more information.
    """
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in greater")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array > x2._array)


def greater_equal(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.greater_equal <numpy.greater_equal>`.

    See its docstring for more information.
    """
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in greater_equal")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array >= x2._array)


def imag(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.imag <numpy.imag>`.

    See its docstring for more information.
    """
    raise ValueError("imag not implemented")


def isfinite(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.isfinite <numpy.isfinite>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in isfinite")
    return Array._new(ak.isfinite(x._array))


def isinf(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.isinf <numpy.isinf>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in isinf")
    return Array._new(ak.isinf(x._array))


def isnan(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.isnan <numpy.isnan>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in isnan")
    return Array._new(ak.isnan(x._array))


def less(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.less <numpy.less>`.

    See its docstring for more information.
    """
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in less")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array < x2._array)


def less_equal(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.less_equal <numpy.less_equal>`.

    See its docstring for more information.
    """
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in less_equal")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array <= x2._array)


def log(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.log <numpy.log>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log")
    return Array._new(ak.log(x._array))


def log1p(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.log1p <numpy.log1p>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log")
    return Array._new(ak.log1p(x._array))


def log2(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.log2 <numpy.log2>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log")
    return Array._new(ak.log2(x._array))


def log10(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.log10 <numpy.log10>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log")
    return Array._new(ak.log10(x._array))


def logaddexp(x1: Array, x2: Array) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.logaddexp <numpy.logaddexp>`.

    See its docstring for more information.
    """
    raise ValueError("logaddexp not implemented")


def logical_and(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.logical_and <numpy.logical_and>`.

    See its docstring for more information.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_and")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array & x2._array)


def logical_not(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.logical_not <numpy.logical_not>`.

    See its docstring for more information.
    """
    repMsg = ak.generic_msg(
        cmd=f"efunc{x._array.ndim}D",
        args={
            "func": "not",
            "array": x._array,
        },
    )
    return Array._new(ak.create_pdarray(repMsg))


def logical_or(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.logical_or <numpy.logical_or>`.

    See its docstring for more information.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_or")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array | x2._array)


def logical_xor(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.logical_xor <numpy.logical_xor>`.

    See its docstring for more information.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_xor")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array ^ x2._array)


def multiply(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.multiply <numpy.multiply>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in multiply")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array * x2._array)


def negative(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.negative <numpy.negative>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in negative")
    return Array._new(-x._array)


def not_equal(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.not_equal <numpy.not_equal>`.

    See its docstring for more information.
    """
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array != x2._array)


def positive(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.positive <numpy.positive>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in positive")
    return Array._new(ak.abs(x._array))


# Note: the function name is different here
def pow(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.power <numpy.power>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in pow")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(ak.power(x1._array, x2._array))


def real(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.real <numpy.real>`.

    See its docstring for more information.
    """
    raise ValueError("real not implemented")


def remainder(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.remainder <numpy.remainder>`.

    See its docstring for more information.
    """
    return Array._new(ak.mod(x1._array, x2._array))


def round(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.round <numpy.round>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in round")
    return Array._new(ak.round(x._array))


def sign(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.sign <numpy.sign>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sign")
    return Array._new(ak.sign(x._array))


def sin(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.sin <numpy.sin>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sin")
    return Array._new(ak.sin(x._array))


def sinh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.sinh <numpy.sinh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sinh")
    return Array._new(ak.sinh(x._array))


def square(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.square <numpy.square>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sign")
    return Array._new(ak.sqrt(x._array))


def sqrt(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.sqrt <numpy.sqrt>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sqrt")
    return Array._new(ak.sqrt(x._array))


def subtract(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.subtract <numpy.subtract>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in subtract")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array - x2._array)


def tan(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.tan <numpy.tan>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tan")
    return Array._new(ak.tan(x._array))


def tanh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.tanh <numpy.tanh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tanh")
    return Array._new(ak.tanh(x._array))


def trunc(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.trunc <numpy.trunc>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in trunc")
    return Array._new(ak.trunc(x._array))
