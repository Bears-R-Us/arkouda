from __future__ import annotations

import arkouda as ak

from ._dtypes import (
    _floating_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _numeric_dtypes,
    _real_floating_dtypes,
    _real_numeric_dtypes,
    _result_type,
)
from ._dtypes import _boolean_dtypes  # _complex_floating_dtypes,
from .array_object import Array


__all__ = [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "conj",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "real",
    "remainder",
    "round",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]


def abs(x: Array, /) -> Array:
    """
    Compute the element-wise absolute value of an array.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in abs")
    return Array._new(ak.abs(x._array))


# Note: the function name is different here
def acos(x: Array, /) -> Array:
    """
    Compute the element-wise arccosine of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acos")
    return Array._new(ak.arccos(x._array))


# Note: the function name is different here
def acosh(x: Array, /) -> Array:
    """
    Compute the element-wise hyperbolic arccosine of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acosh")
    return Array._new(ak.arccosh(x._array))


def add(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise sum of two arrays.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in add")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array + x2._array)


def asin(x: Array, /) -> Array:
    """
    Compute the element-wise arcsine of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asin")
    return Array._new(ak.arcsin(x._array))


def asinh(x: Array, /) -> Array:
    """
    Compute the element-wise hyperbolic arcsine of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asinh")
    return Array._new(ak.arcsinh(x._array))


def atan(x: Array, /) -> Array:
    """
    Compute the element-wise arctangent of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atan")
    return Array._new(ak.arctan(x._array))


def atan2(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise arctangent of x1/x2.
    """
    if x1.dtype not in _real_floating_dtypes or x2.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in atan2")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(ak.arctan2(x1._array, x2._array))


def atanh(x: Array, /) -> Array:
    """
    Compute the element-wise hyperbolic arctangent of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atanh")
    return Array._new(ak.arctanh(x._array))


def bitwise_and(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise bitwise AND of two arrays.
    """
    if x1.dtype not in _integer_or_boolean_dtypes or x2.dtype not in _integer_or_boolean_dtypes:
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_and")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array & x2._array)


def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise bitwise left shift of x1 by x2.
    """
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in bitwise_left_shift")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    # Note: bitwise_left_shift is only defined for x2 nonnegative.
    if ak.any(x2._array < 0):
        raise ValueError("bitwise_left_shift(x1, x2) is only defined for x2 >= 0")
    return Array._new(x1._array << x2._array)


def bitwise_invert(x: Array, /) -> Array:
    """
    Compute the element-wise bitwise NOT of an array.
    """
    raise ValueError("bitwise invert not implemented")


def bitwise_or(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise bitwise OR of two arrays.
    """
    if x1.dtype not in _integer_or_boolean_dtypes or x2.dtype not in _integer_or_boolean_dtypes:
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_or")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array | x2._array)


def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise bitwise right shift of x1 by x2.
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
    Compute the element-wise bitwise XOR of two arrays.
    """
    if x1.dtype not in _integer_or_boolean_dtypes or x2.dtype not in _integer_or_boolean_dtypes:
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_xor")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array ^ x2._array)


def ceil(x: Array, /) -> Array:
    """
    Compute the element-wise ceiling of a floating point array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in ceil")
    return Array._new(ak.ceil(x._array))


def conj(x: Array, /) -> Array:
    """
    Compute the element-wise complex conjugate of an array.

    WARNING: Not yet implemented.
    """
    raise ValueError("conj not implemented - Arkouda does not support complex types")


def cos(x: Array, /) -> Array:
    """
    Compute the element-wise cosine of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cos")
    return Array._new(ak.cos(x._array))


def cosh(x: Array, /) -> Array:
    """
    Compute the element-wise hyperbolic cosine of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cosh")
    return Array._new(ak.cosh(x._array))


def divide(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise division of x1 by x2.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in divide")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array / x2._array)


def equal(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise equality of two arrays.
    """
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array == x2._array)


def exp(x: Array, /) -> Array:
    """
    Compute the element-wise exponential of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in exp")
    return Array._new(ak.exp(x._array))


def expm1(x: Array, /) -> Array:
    """
    Compute the element-wise exponential of x-1.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in exp")
    return Array._new(ak.expm1(x._array))


def floor(x: Array, /) -> Array:
    """
    Compute the element-wise floor of a floating point array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in floor")
    return Array._new(ak.floor(x._array))


def floor_divide(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise floor division of x1 by x2.
    """
    raise ValueError("exp not implemented")


def greater(x1: Array, x2: Array, /) -> Array:
    """
    Apply `>` element-wise to two arrays.
    """
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in greater")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array > x2._array)


def greater_equal(x1: Array, x2: Array, /) -> Array:
    """
    Apply `>=` element-wise to two arrays.
    """
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in greater_equal")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array >= x2._array)


def imag(x: Array, /) -> Array:
    """
    Get the element-wise imaginary part of a Complex array.

    WARNING: Not yet implemented.
    """
    raise ValueError("imag not implemented")


def isfinite(x: Array, /) -> Array:
    """
    Determine if an array's elements are finite.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in isfinite")
    return Array._new(ak.isfinite(x._array))


def isinf(x: Array, /) -> Array:
    """
    Determine if an array's elements are infinite.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in isinf")
    return Array._new(ak.isinf(x._array))


def isnan(x: Array, /) -> Array:
    """
    Determine if an array's elements are NaN.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in isnan")
    return Array._new(ak.isnan(x._array))


def less(x1: Array, x2: Array, /) -> Array:
    """
    Apply `<` element-wise to two arrays.
    """
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in less")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array < x2._array)


def less_equal(x1: Array, x2: Array, /) -> Array:
    """
    Apply `<=` element-wise to two arrays.
    """
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in less_equal")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array <= x2._array)


def log(x: Array, /) -> Array:
    """
    Compute the element-wise natural logarithm of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log")
    return Array._new(ak.log(x._array))


def log1p(x: Array, /) -> Array:
    """
    Compute the element-wise natural logarithm of x+1.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log")
    return Array._new(ak.log1p(x._array))


def log2(x: Array, /) -> Array:
    """
    Compute the element-wise base-2 logarithm of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log")
    return Array._new(ak.log2(x._array))


def log10(x: Array, /) -> Array:
    """
    Compute the element-wise base-10 logarithm of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log")
    return Array._new(ak.log10(x._array))


def logaddexp(x1: Array, x2: Array) -> Array:
    """
    Compute the element-wise logarithm of the sum of exponentials of two arrays
    (i.e., log(exp(x1) + exp(x2))).

    WARNING: Not yet implemented.
    """
    raise ValueError("logaddexp not implemented")


def logical_and(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise logical AND of two boolean arrays.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_and")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array & x2._array)


def logical_not(x: Array, /) -> Array:
    """
    Compute the element-wise logical NOT of a boolean array.
    """
    return ~x


def logical_or(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise logical OR of two boolean arrays.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_or")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array | x2._array)


def logical_xor(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise logical XOR of two boolean arrays.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_xor")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array ^ x2._array)


def multiply(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise product of two arrays.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in multiply")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array * x2._array)


def negative(x: Array, /) -> Array:
    """
    Compute the element-wise negation of an array.
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
    Get the element-wise real part of a Complex array.

    WARNING: Not yet implemented.
    """
    raise ValueError("real not implemented")


def remainder(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise remainder of x1 divided by x2.
    """
    return Array._new(ak.mod(x1._array, x2._array))


def round(x: Array, /) -> Array:
    """
    Compute the element-wise rounding of an array.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in round")
    return Array._new(ak.round(x._array))


def sign(x: Array, /) -> Array:
    """
    Compute the element-wise sign of an array.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sign")
    return Array._new(ak.sign(x._array))


def sin(x: Array, /) -> Array:
    """
    Compute the element-wise sine of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sin")
    return Array._new(ak.sin(x._array))


def sinh(x: Array, /) -> Array:
    """
    Compute the element-wise hyperbolic sine of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sinh")
    return Array._new(ak.sinh(x._array))


def square(x: Array, /) -> Array:
    """
    Compute the element-wise square of an array.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sign")
    return Array._new(ak.sqrt(x._array))


def sqrt(x: Array, /) -> Array:
    """
    Compute the element-wise square root of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sqrt")
    return Array._new(ak.sqrt(x._array))


def subtract(x1: Array, x2: Array, /) -> Array:
    """
    Compute the element-wise difference of two arrays.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in subtract")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(x1._array - x2._array)


def tan(x: Array, /) -> Array:
    """
    Compute the element-wise tangent of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tan")
    return Array._new(ak.tan(x._array))


def tanh(x: Array, /) -> Array:
    """
    Compute the element-wise hyperbolic tangent of an array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tanh")
    return Array._new(ak.tanh(x._array))


def trunc(x: Array, /) -> Array:
    """
    Compute the element-wise truncation of a floating-point array.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in trunc")
    return Array._new(ak.trunc(x._array))
