import json
from enum import Enum
from typing import ForwardRef, List, Optional, Sequence, Tuple, Union
from typing import cast as type_cast
from typing import no_type_check

import numpy as np  # type: ignore
from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.dtypes import (
    BigInt,
    DTypes,
    _as_dtype,
    bigint,
    int_scalars,
    isSupportedNumber,
    numeric_scalars,
    resolve_scalar_dtype,
)
from arkouda.groupbyclass import GroupBy
from arkouda.pdarrayclass import all as ak_all
from arkouda.pdarrayclass import any as ak_any
from arkouda.pdarrayclass import argmax, create_pdarray, pdarray
from arkouda.pdarraycreation import array, linspace, scalar_array
from arkouda.strings import Strings

Categorical = ForwardRef("Categorical")
SegArray = ForwardRef("SegArray")

__all__ = [
    "cast",
    "abs",
    "ceil",
    "clip",
    "floor",
    "trunc",
    "round",
    "sign",
    "isfinite",
    "isinf",
    "isnan",
    "log",
    "log2",
    "log10",
    "log1p",
    "exp",
    "expm1",
    "square",
    "cumsum",
    "cumprod",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "rad2deg",
    "deg2rad",
    "hash",
    "where",
    "histogram",
    "histogram2d",
    "histogramdd",
    "value_counts",
    "ErrorMode",
]


class ErrorMode(Enum):
    strict = "strict"
    ignore = "ignore"
    return_validity = "return_validity"


@typechecked
def cast(
    pda: Union[pdarray, Strings, Categorical],  # type: ignore
    dt: Union[np.dtype, type, str, BigInt],
    errors: ErrorMode = ErrorMode.strict,
) -> Union[Union[pdarray, Strings, Categorical], Tuple[pdarray, pdarray]]:  # type: ignore
    """
    Cast an array to another dtype.

    Parameters
    ----------
    pda : pdarray or Strings
        The array of values to cast
    dt : np.dtype, type, or str
        The target dtype to cast values to
    errors : {strict, ignore, return_validity}
        Controls how errors are handled when casting strings to a numeric type
        (ignored for casts from numeric types).
            - strict: raise RuntimeError if *any* string cannot be converted
            - ignore: never raise an error. Uninterpretable strings get
                converted to NaN (float64), -2**63 (int64), zero (uint64 and
                uint8), or False (bool)
            - return_validity: in addition to returning the same output as
              "ignore", also return a bool array indicating where the cast
              was successful.

    Returns
    -------
    pdarray or Strings
        Array of values cast to desired dtype
    [validity : pdarray(bool)]
        If errors="return_validity" and input is Strings, a second array is
        returned with True where the cast succeeded and False where it failed.

    Notes
    -----
    The cast is performed according to Chapel's casting rules and is NOT safe
    from overflows or underflows. The user must ensure that the target dtype
    has the precision and capacity to hold the desired result.

    Examples
    --------
    >>> ak.cast(ak.linspace(1.0,5.0,5), dt=ak.int64)
    array([1, 2, 3, 4, 5])

    >>> ak.cast(ak.arange(0,5), dt=ak.float64).dtype
    dtype('float64')

    >>> ak.cast(ak.arange(0,5), dt=ak.bool)
    array([False, True, True, True, True])

    >>> ak.cast(ak.linspace(0,4,5), dt=ak.bool)
    array([False, True, True, True, True])
    """
    from arkouda.categorical import Categorical  # type: ignore

    if isinstance(pda, pdarray):
        name = pda.name
    elif isinstance(pda, Strings):
        name = pda.entry.name
        if dt is Categorical or dt == "Categorical":
            return Categorical(pda)  # type: ignore
    elif isinstance(pda, Categorical):  # type: ignore
        if dt is Strings or dt in ["Strings", "str"]:
            return pda.categories[pda.codes]
        else:
            raise ValueError("Categoricals can only be casted to Strings")
    # typechecked decorator guarantees no other case

    dt = _as_dtype(dt)
    cmd = f"cast{pda.ndim}D"
    repMsg = generic_msg(
        cmd=cmd,
        args={
            "name": name,
            "objType": pda.objType,
            "targetDtype": dt.name,
            "opt": errors.name,
        },
    )
    if dt.name.startswith("str"):
        return Strings.from_parts(*(type_cast(str, repMsg).split("+")))
    else:
        if errors == ErrorMode.return_validity:
            a, b = type_cast(str, repMsg).split("+")
            return create_pdarray(type_cast(str, a)), create_pdarray(type_cast(str, b))
        else:
            return create_pdarray(type_cast(str, repMsg))


@typechecked
def abs(pda: pdarray) -> pdarray:
    """
    Return the element-wise absolute value of the array.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing absolute values of the input array elements

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> ak.abs(ak.arange(-5,-1))
    array([5, 4, 3, 2])

    >>> ak.abs(ak.linspace(-5,-1,5))
    array([5, 4, 3, 2, 1])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "abs",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def ceil(pda: pdarray) -> pdarray:
    """
    Return the element-wise ceiling of the array.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing ceiling values of the input array elements

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> ak.ceil(ak.linspace(1.1,5.5,5))
    array([2, 3, 4, 5, 6])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "ceil",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def floor(pda: pdarray) -> pdarray:
    """
    Return the element-wise floor of the array.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing floor values of the input array elements

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> ak.floor(ak.linspace(1.1,5.5,5))
    array([1, 2, 3, 4, 5])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "floor",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def round(pda: pdarray) -> pdarray:
    """
    Return the element-wise rounding of the array.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing input array elements rounded to the nearest integer

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> ak.round(ak.array([1.1, 2.5, 3.14159]))
    array([1, 3, 3])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "round",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def trunc(pda: pdarray) -> pdarray:
    """
    Return the element-wise truncation of the array.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing input array elements truncated to the nearest integer

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> ak.trunc(ak.array([1.1, 2.5, 3.14159]))
    array([1, 2, 3])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "trunc",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def sign(pda: pdarray) -> pdarray:
    """
    Return the element-wise sign of the array.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing sign values of the input array elements

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> ak.sign(ak.array([-10, -5, 0, 5, 10]))
    array([-1, -1, 0, 1, 1])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "sign",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def isfinite(pda: pdarray) -> pdarray:
    """
    Return the element-wise isfinite check applied to the array.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing boolean values indicating whether the
        input array elements are finite

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    RuntimeError
        if the underlying pdarray is not float-based

    Examples
    --------
    >>> ak.isfinite(ak.array[1.0, 2.0, ak.inf])
    array([True, True, False])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "isfinite",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def isinf(pda: pdarray) -> pdarray:
    """
    Return the element-wise isinf check applied to the array.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing boolean values indicating whether the
        input array elements are infinite

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    RuntimeError
        if the underlying pdarray is not float-based

    Examples
    --------
    >>> ak.isinf(ak.array[1.0, 2.0, ak.inf])
    array([False, False, True])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "isinf",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def isnan(pda: pdarray) -> pdarray:
    """
    Return the element-wise isnan check applied to the array.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing boolean values indicating whether the
        input array elements are NaN

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    RuntimeError
        if the underlying pdarray is not float-based

    Examples
    --------
    >>> ak.isnan(ak.array[1.0, 2.0, 1.0 / 0.0])
    array([False, False, True])
    """
    from arkouda.util import is_numeric, is_float

    if is_numeric(pda) and not is_float(pda):
        from arkouda.pdarraycreation import full

        return full(pda.size, False, dtype=bool)
    elif not is_numeric(pda):
        raise TypeError("isnan only supports pdarray of numeric type.")

    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "isnan",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def log(pda: pdarray) -> pdarray:
    """
    Return the element-wise natural log of the array.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing natural log values of the input
        array elements

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Notes
    -----
    Logarithms with other bases can be computed as follows:

    Examples
    --------
    >>> A = ak.array([1, 10, 100])
    # Natural log
    >>> ak.log(A)
    array([0, 2.3025850929940459, 4.6051701859880918])
    # Log base 10
    >>> ak.log(A) / np.log(10)
    array([0, 1, 2])
    # Log base 2
    >>> ak.log(A) / np.log(2)
    array([0, 3.3219280948873626, 6.6438561897747253])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "log",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def log10(x: pdarray) -> pdarray:
    """
    Return the element-wise base 10 log of the array.

    Parameters
    __________
    x : pdarray
        array to compute on

    Returns
    _______
    pdarray contain values of the base 10 log
    """
    repMsg = generic_msg(
        cmd=f"efunc{x.ndim}D",
        args={
            "func": "log10",
            "array": x,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def log2(x: pdarray) -> pdarray:
    """
    Return the element-wise base 2 log of the array.

    Parameters
    __________
    x : pdarray
        array to compute on

    Returns
    _______
    pdarray contain values of the base 2 log
    """
    repMsg = generic_msg(
        cmd=f"efunc{x.ndim}D",
        args={
            "func": "log2",
            "array": x,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def log1p(x: pdarray) -> pdarray:
    """
    Return the element-wise natural log of one plus the array.

    Parameters
    __________
    x : pdarray
        array to compute on

    Returns
    _______
    pdarray contain values of the natural log of one plus the array
    """
    repMsg = generic_msg(
        cmd=f"efunc{x.ndim}D",
        args={
            "func": "log1p",
            "array": x,
        },
    )
    return create_pdarray(repMsg)


@typechecked
def exp(pda: pdarray) -> pdarray:
    """
    Return the element-wise exponential of the array.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing exponential values of the input
        array elements

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> ak.exp(ak.arange(1,5))
    array([2.7182818284590451, 7.3890560989306504, 20.085536923187668, 54.598150033144236])

    >>> ak.exp(ak.uniform(5,1.0,5.0))
    array([11.84010843172504, 46.454368507659211, 5.5571769623557188,
           33.494295836924771, 13.478894913238722])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "exp",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def expm1(pda: pdarray) -> pdarray:
    """
    Return the element-wise exponential of the array minus one.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing exponential values of the input
        array elements minus one

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> ak.exp1m(ak.arange(1,5))
    array([1.7182818284590451, 6.3890560989306504, 19.085536923187668, 53.598150033144236])

    >>> ak.exp1m(ak.uniform(5,1.0,5.0))
    array([10.84010843172504, 45.454368507659211, 4.5571769623557188,
           32.494295836924771, 12.478894913238722])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "expm1",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def square(pda: pdarray) -> pdarray:
    """
    Return the element-wise square of the array.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing square values of the input
        array elements

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> ak.square(ak.arange(1,5))
    array([1, 4, 9, 16])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "square",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def cumsum(pda: pdarray) -> pdarray:
    """
    Return the cumulative sum over the array.

    The sum is inclusive, such that the ``i`` th element of the
    result is the sum of elements up to and including ``i``.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing cumulative sums for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> ak.cumsum(ak.arange([1,5]))
    array([1, 3, 6])

    >>> ak.cumsum(ak.uniform(5,1.0,5.0))
    array([3.1598310770203937, 5.4110385860243131, 9.1622479306453748,
           12.710615785506533, 13.945880905466208])

    >>> ak.cumsum(ak.randint(0, 1, 5, dtype=ak.bool))
    array([0, 1, 1, 2, 3])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "cumsum",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def cumprod(pda: pdarray) -> pdarray:
    """
    Return the cumulative product over the array.

    The product is inclusive, such that the ``i`` th element of the
    result is the product of elements up to and including ``i``.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing cumulative products for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> ak.cumprod(ak.arange(1,5))
    array([1, 2, 6, 24]))

    >>> ak.cumprod(ak.uniform(5,1.0,5.0))
    array([1.5728783400481925, 7.0472855509390593, 33.78523998586553,
           134.05309592737584, 450.21589865655358])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "cumprod",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def sin(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise sine of the array.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True,
        the sine will be applied to the corresponding value. Elsewhere, it will retain
        its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing sin for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    return _trig_helper(pda, "sin", where)


@typechecked
def cos(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise cosine of the array.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True,
        the cosine will be applied to the corresponding value. Elsewhere, it will retain
        its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing cosine for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    return _trig_helper(pda, "cos", where)


@typechecked
def tan(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise tangent of the array.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True,
        the tangent will be applied to the corresponding value. Elsewhere, it will retain
        its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing tangent for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    return _trig_helper(pda, "tan", where)


@typechecked
def arcsin(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise inverse sine of the array. The result is between -pi/2 and pi/2.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True,
        the inverse sine will be applied to the corresponding value. Elsewhere, it will retain
        its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing inverse sine for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    return _trig_helper(pda, "arcsin", where)


@typechecked
def arccos(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise inverse cosine of the array. The result is between 0 and pi.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True,
        the inverse cosine will be applied to the corresponding value. Elsewhere, it will retain
        its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing inverse cosine for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    return _trig_helper(pda, "arccos", where)


@typechecked
def arctan(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise inverse tangent of the array. The result is between -pi/2 and pi/2.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True,
        the inverse tangent will be applied to the corresponding value. Elsewhere, it will retain
        its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing inverse tangent for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    return _trig_helper(pda, "arctan", where)


@typechecked
def arctan2(
    num: Union[pdarray, numeric_scalars],
    denom: Union[pdarray, numeric_scalars],
    where: Union[bool, pdarray] = True,
) -> pdarray:
    """
    Return the element-wise inverse tangent of the array pair. The result chosen is the
    signed angle in radians between the ray ending at the origin and passing through the
    point (1,0), and the ray ending at the origin and passing through the point (denom, num).
    The result is between -pi and pi.

    Parameters
    ----------
    num : Union[numeric_scalars, pdarray]
        Numerator of the arctan2 argument.
    denom : Union[numeric_scalars, pdarray]
        Denominator of the arctan2 argument.
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True,
        the inverse tangent will be applied to the corresponding values. Elsewhere, it will retain
        its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing inverse tangent for each corresponding element pair
        of the original pdarray, using the signed values or the numerator and
        denominator to get proper placement on unit circle.

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    if not all(
        isSupportedNumber(arg) or isinstance(arg, pdarray) for arg in [num, denom]
    ):
        raise TypeError(
            f"Unsupported types {type(num)} and/or {type(denom)}. Supported "
            "types are numeric scalars and pdarrays. At least one argument must be a pdarray."
        )
    if isSupportedNumber(num) and isSupportedNumber(denom):
        raise TypeError(
            f"Unsupported types {type(num)} and/or {type(denom)}. Supported "
            "types are numeric scalars and pdarrays. At least one argument must be a pdarray."
        )
    # TODO: handle shape broadcasting for multidimensional arrays
    if isinstance(num, pdarray) or isinstance(denom, pdarray):
        ndim = num.ndim if isinstance(num, pdarray) else denom.ndim  # type: ignore[union-attr]
        if where is True:
            repMsg = type_cast(
                str,
                generic_msg(
                    cmd=f"efunc2Arg{ndim}D",
                    args={
                        "func": "arctan2",
                        "A": num,
                        "B": denom,
                    },
                ),
            )
            return create_pdarray(repMsg)
        elif where is False:
            return num / denom  # type: ignore
        else:
            if where.dtype != bool:
                raise TypeError(f"where must have dtype bool, got {where.dtype} instead")
            if isinstance(num, pdarray) and isinstance(denom, pdarray):
                # TODO: handle shape broadcasting for multidimensional arrays
                repMsg = type_cast(
                    str,
                    generic_msg(
                        cmd=f"efunc2Arg{ndim}D",
                        args={
                            "func": "arctan2",
                            "A": num[where],
                            "B": denom[where],
                        },
                    ),
                )
            if not isinstance(num, pdarray) or not isinstance(denom, pdarray):
                repMsg = type_cast(
                    str,
                    generic_msg(
                        cmd=f"efunc2Arg{ndim}D",
                        args={
                            "func": "arctan2",
                            "A": num if not isinstance(num, pdarray) else num[where],
                            "B": denom if not isinstance(denom, pdarray) else denom[where],
                        },
                    ),
                )
            new_pda = num / denom
            ret = create_pdarray(repMsg)
            new_pda = cast(new_pda, ret.dtype)
            new_pda[where] = ret
            return new_pda
    else:
        return scalar_array(arctan2(num, denom) if where else num / denom)


@typechecked
def sinh(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise hyperbolic sine of the array.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True,
        the hyperbolic sine will be applied to the corresponding value. Elsewhere, it will retain
        its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing hyperbolic sine for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    return _trig_helper(pda, "sinh", where)


@typechecked
def cosh(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise hyperbolic cosine of the array.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True,
        the hyperbolic cosine will be applied to the corresponding value. Elsewhere, it will retain
        its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing hyperbolic cosine for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    return _trig_helper(pda, "cosh", where)


@typechecked
def tanh(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise hyperbolic tangent of the array.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True,
        the hyperbolic tangent will be applied to the corresponding value. Elsewhere, it will retain
        its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing hyperbolic tangent for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    return _trig_helper(pda, "tanh", where)


@typechecked
def arcsinh(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise inverse hyperbolic sine of the array.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True,
        the inverse hyperbolic sine will be applied to the corresponding value. Elsewhere, it will retain
        its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing inverse hyperbolic sine for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    return _trig_helper(pda, "arcsinh", where)


@typechecked
def arccosh(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise inverse hyperbolic cosine of the array.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True,
        the inverse hyperbolic cosine will be applied to the corresponding value. Elsewhere, it will
        retain its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing inverse hyperbolic cosine for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    return _trig_helper(pda, "arccosh", where)


@typechecked
def arctanh(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise inverse hyperbolic tangent of the array.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True,
        the inverse hyperbolic tangent will be applied to the corresponding value. Elsewhere,
        it will retain its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing inverse hyperbolic tangent for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameters are not a pdarray or numeric scalar.
    """
    return _trig_helper(pda, "arctanh", where)


def _trig_helper(
    pda: pdarray, func: str, where: Union[bool, pdarray] = True
) -> pdarray:
    """
    Returns the result of the input trig function acting element-wise on the array.

    Parameters
    ----------
    pda : pdarray
    func : string
        The designated trig function that is passed in
    where : Boolean or pdarray
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the respective trig function. Elsewhere,
        it will retain its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray with the trig function applied at each element of pda

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    TypeError
        Raised if where condition is not type Boolean
    """
    if where is True:
        repMsg = type_cast(
            str,
            generic_msg(
                cmd=f"efunc{pda.ndim}D",
                args={
                    "func": func,
                    "array": pda,
                },
            ),
        )
        return create_pdarray(repMsg)
    elif where is False:
        return pda
    else:
        if where.dtype != bool:
            raise TypeError(f"where must have dtype bool, got {where.dtype} instead")
        repMsg = type_cast(
            str,
            generic_msg(
                cmd=f"efunc{pda.ndim}D",
                args={
                    "func": func,
                    "array": pda[where],
                },
            ),
        )
        new_pda = pda[:]
        ret = create_pdarray(repMsg)
        new_pda = cast(new_pda, ret.dtype)
        new_pda[where] = ret
        return new_pda


@typechecked
def rad2deg(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Converts angles element-wise from radians to degrees.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True, the
        corresponding value will be converted from radians to degrees. Elsewhere, it will retain its
        original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing an angle converted to degrees, from radians, for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    if where is True:
        return 180 * (pda / np.pi)
    elif where is False:
        return pda
    else:
        new_pda = pda
        ret = 180 * (pda[where] / np.pi)
        new_pda = cast(new_pda, ret.dtype)
        new_pda[where] = ret
        return new_pda


@typechecked
def deg2rad(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Converts angles element-wise from degrees to radians.

    Parameters
    ----------
    pda : pdarray
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True, the
        corresponding value will be converted from degrees to radians. Elsewhere, it will retain its
        original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing an angle converted to radians, from degrees, for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    if where is True:
        return np.pi * pda / 180
    elif where is False:
        return pda
    else:
        new_pda = pda
        ret = np.pi * pda[where] / 180
        new_pda = cast(new_pda, ret.dtype)
        new_pda[where] = ret
        return new_pda


def _hash_helper(a):
    from arkouda import Categorical as Categorical_
    from arkouda import SegArray as SegArray_

    if isinstance(a, SegArray_):
        return json.dumps(
            {
                "segments": a.segments.name,
                "values": a.values.name,
                "valObjType": a.values.objType,
            }
        )
    elif isinstance(a, Categorical_):
        return json.dumps({"categories": a.categories.name, "codes": a.codes.name})
    else:
        return a.name


# this is # type: ignored and doesn't actually do any type checking
# the type hints are there as a reference to show which types are expected
# type validation is done within the function
def hash(
    pda: Union[  # type: ignore
        Union[pdarray, Strings, SegArray, Categorical],
        List[Union[pdarray, Strings, SegArray, Categorical]],
    ],
    full: bool = True,
) -> Union[Tuple[pdarray, pdarray], pdarray]:
    """
    Return an element-wise hash of the array or list of arrays.

    Parameters
    ----------
    pda : Union[pdarray, Strings, Segarray, Categorical],
     List[Union[pdarray, Strings, Segarray, Categorical]]]

    full : bool
        This is only used when a single pdarray is passed into hash
        By default, a 128-bit hash is computed and returned as
        two int64 arrays. If full=False, then a 64-bit hash
        is computed and returned as a single int64 array.

    Returns
    -------
    hashes
        If full=True or a list of pdarrays is passed,
        a 2-tuple of pdarrays containing the high
        and low 64 bits of each hash, respectively.
        If full=False and a single pdarray is passed,
        a single pdarray containing a 64-bit hash

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Notes
    -----
    In the case of a single pdarray being passed, this function
    uses the SIPhash algorithm, which can output either a 64-bit
    or 128-bit hash. However, the 64-bit hash runs a significant
    risk of collisions when applied to more than a few million
    unique values. Unless the number of unique values is known to
    be small, the 128-bit hash is strongly recommended.

    Note that this hash should not be used for security, or for
    any cryptographic application. Not only is SIPhash not
    intended for such uses, but this implementation employs a
    fixed key for the hash, which makes it possible for an
    adversary with control over input to engineer collisions.

    In the case of a list of pdrrays, Strings, Categoricals, or Segarrays
    being passed, a non-linear function must be applied to each
    array since hashes of subsequent arrays cannot be simply XORed
    because equivalent values will cancel each other out, hence we
    do a rotation by the ordinal of the array.
    """
    from arkouda import Categorical as Categorical_
    from arkouda import SegArray as SegArray_

    if isinstance(pda, (pdarray, Strings, SegArray_, Categorical_)):
        return _hash_single(pda, full) if isinstance(pda, pdarray) else pda.hash()
    elif isinstance(pda, List):
        if any(
            wrong_type := [
                not isinstance(a, (pdarray, Strings, SegArray_, Categorical_))
                for a in pda
            ]
        ):
            raise TypeError(
                f"Unsupported type {type(pda[np.argmin(wrong_type)])}. Supported types are pdarray,"
                f" SegArray, Strings, Categoricals, and Lists of these types."
            )
        # replace bigint pdarrays with the uint limbs
        expanded_pda = []
        for a in pda:
            if isinstance(a, pdarray) and a.dtype == bigint:
                expanded_pda.extend(a.bigint_to_uint_arrays())
            else:
                expanded_pda.append(a)
        types_list = [a.objType for a in expanded_pda]
        names_list = [_hash_helper(a) for a in expanded_pda]
        rep_msg = type_cast(
            str,
            generic_msg(
                cmd="hashList",
                args={
                    "nameslist": names_list,
                    "typeslist": types_list,
                    "length": len(expanded_pda),
                    "size": len(expanded_pda[0]),
                },
            ),
        )
        hashes = json.loads(rep_msg)
        return create_pdarray(hashes["upperHash"]), create_pdarray(hashes["lowerHash"])
    else:
        raise TypeError(
            f"Unsupported type {type(pda)}. Supported types are pdarray,"
            f" SegArray, Strings, Categoricals, and Lists of these types."
        )


@typechecked
def _hash_single(pda: pdarray, full: bool = True):
    if pda.dtype == bigint:
        return hash(pda.bigint_to_uint_arrays())
    repMsg = type_cast(
        str,
        generic_msg(
            cmd=f"efunc{pda.ndim}D",
            args={
                "func": "hash128" if full else "hash64",
                "array": pda,
            },
        ),
    )
    if full:
        a, b = repMsg.split("+")
        return create_pdarray(a), create_pdarray(b)
    else:
        return create_pdarray(repMsg)


@no_type_check
def _str_cat_where(
    condition: pdarray,
    A: Union[str, Strings, Categorical],
    B: Union[str, Strings, Categorical],
) -> Union[Strings, Categorical]:
    # added @no_type_check because mypy can't handle Categorical not being declared
    # sooner, but there are circular dependencies preventing that
    from arkouda.categorical import Categorical
    from arkouda.pdarraysetops import concatenate

    if isinstance(A, str) and isinstance(B, (Categorical, Strings)):
        # This allows us to assume if a str is present it is B
        A, B, condition = B, A, ~condition

    # one cat and one str
    if isinstance(A, Categorical) and isinstance(B, str):
        is_in_categories = A.categories == B
        if ak_any(is_in_categories):
            new_categories = A.categories
            b_code = argmax(is_in_categories)
        else:
            new_categories = concatenate([A.categories, array([B])])
            b_code = A.codes.size + 1
        new_codes = where(condition, A.codes, b_code)
        return Categorical.from_codes(
            new_codes, new_categories, NAvalue=A.NAvalue
        ).reset_categories()

    # both cat
    if isinstance(A, Categorical) and isinstance(B, Categorical):
        if A.codes.size != B.codes.size:
            raise TypeError("Categoricals must be same length")
        if A.categories.size != B.categories.size or not ak_all(
            A.categories == B.categories
        ):
            A, B = A.standardize_categories([A, B])
        new_codes = where(condition, A.codes, B.codes)
        return Categorical.from_codes(
            new_codes, A.categories, NAvalue=A.NAvalue
        ).reset_categories()

    # one strings and one str
    if isinstance(A, Strings) and isinstance(B, str):
        new_lens = where(condition, A.get_lengths(), len(B))
        repMsg = generic_msg(
            cmd="segmentedWhere",
            args={
                "seg_str": A,
                "other": B,
                "is_str_literal": True,
                "new_lens": new_lens,
                "condition": condition,
            },
        )
        return Strings.from_return_msg(repMsg)

    # both strings
    if isinstance(A, Strings) and isinstance(B, Strings):
        if A.size != B.size:
            raise TypeError("Strings must be same length")
        new_lens = where(condition, A.get_lengths(), B.get_lengths())
        repMsg = generic_msg(
            cmd="segmentedWhere",
            args={
                "seg_str": A,
                "other": B,
                "is_str_literal": False,
                "new_lens": new_lens,
                "condition": condition,
            },
        )
        return Strings.from_return_msg(repMsg)

    raise TypeError("ak.where is not supported between Strings and Categorical")


@typechecked
def where(
    condition: pdarray,
    A: Union[str, numeric_scalars, pdarray, Strings, Categorical],  # type: ignore
    B: Union[str, numeric_scalars, pdarray, Strings, Categorical],  # type: ignore
) -> Union[pdarray, Strings, Categorical]:  # type: ignore
    """
    Returns an array with elements chosen from A and B based upon a
    conditioning array. As is the case with numpy.where, the return array
    consists of values from the first array (A) where the conditioning array
    elements are True and from the second array (B) where the conditioning
    array elements are False.

    Parameters
    ----------
    condition : pdarray
        Used to choose values from A or B
    A : Union[numeric_scalars, str, pdarray, Strings, Categorical]
        Value(s) used when condition is True
    B : Union[numeric_scalars, str, pdarray, Strings, Categorical]
        Value(s) used when condition is False

    Returns
    -------
    pdarray
        Values chosen from A where the condition is True and B where
        the condition is False

    Raises
    ------
    TypeError
        Raised if the condition object is not a pdarray, if A or B is not
        an int, np.int64, float, np.float64, pdarray, str, Strings, Categorical
        if pdarray dtypes are not supported or do not match, or multiple
        condition clauses (see Notes section) are applied
    ValueError
        Raised if the shapes of the condition, A, and B pdarrays are unequal

    Examples
    --------
    >>> a1 = ak.arange(1,10)
    >>> a2 = ak.ones(9, dtype=np.int64)
    >>> cond = a1 < 5
    >>> ak.where(cond,a1,a2)
    array([1, 2, 3, 4, 1, 1, 1, 1, 1])

    >>> a1 = ak.arange(1,10)
    >>> a2 = ak.ones(9, dtype=np.int64)
    >>> cond = a1 == 5
    >>> ak.where(cond,a1,a2)
    array([1, 1, 1, 1, 5, 1, 1, 1, 1])

    >>> a1 = ak.arange(1,10)
    >>> a2 = 10
    >>> cond = a1 < 5
    >>> ak.where(cond,a1,a2)
    array([1, 2, 3, 4, 10, 10, 10, 10, 10])

    >>> s1 = ak.array([f'str {i}' for i in range(10)])
    >>> s2 = 'str 21'
    >>> cond = (ak.arange(10) % 2 == 0)
    >>> ak.where(cond,s1,s2)
    array(['str 0', 'str 21', 'str 2', 'str 21', 'str 4', 'str 21', 'str 6', 'str 21', 'str 8','str 21'])

    >>> c1 = ak.Categorical(ak.array([f'str {i}' for i in range(10)]))
    >>> c2 = ak.Categorical(ak.array([f'str {i}' for i in range(9, -1, -1)]))
    >>> cond = (ak.arange(10) % 2 == 0)
    >>> ak.where(cond,c1,c2)
    array(['str 0', 'str 8', 'str 2', 'str 6', 'str 4', 'str 4', 'str 6', 'str 2', 'str 8', 'str 0'])

    Notes
    -----
    A and B must have the same dtype and only one conditional clause
    is supported e.g., n < 5, n > 1, which is supported in numpy
    is not currently supported in Arkouda
    """
    if (not isSupportedNumber(A) and not isinstance(A, pdarray)) or (
        not isSupportedNumber(B) and not isinstance(B, pdarray)
    ):
        from arkouda.categorical import Categorical  # type: ignore

        # fmt: off
        if (
            not isinstance(A, (str, Strings, Categorical))  # type: ignore
            or not isinstance(B, (str, Strings, Categorical))  # type: ignore
        ):
            # fmt:on
            raise TypeError(
                "both A and B must be an int, np.int64, float, np.float64, pdarray OR"
                " both A and B must be an str, Strings, Categorical"
            )
        return _str_cat_where(condition, A, B)
    if isinstance(A, pdarray) and isinstance(B, pdarray):
        # TODO: handle shape broadcasting for multidimensional arrays
        repMsg = generic_msg(
            cmd=f"efunc3vv{condition.ndim}D",
            args={
                "func": "where",
                "condition": condition,
                "a": A,
                "b": B,
            },
        )
    # For scalars, try to convert it to the array's dtype
    elif isinstance(A, pdarray) and np.isscalar(B):
        repMsg = generic_msg(
            cmd=f"efunc3vs{condition.ndim}D",
            args={
                "func": "where",
                "condition": condition,
                "a": A,
                "dtype": A.dtype.name,
                "scalar": A.format_other(B),
            },
        )
    elif isinstance(B, pdarray) and np.isscalar(A):
        repMsg = generic_msg(
            cmd=f"efunc3sv{condition.ndim}D",
            args={
                "func": "where",
                "condition": condition,
                "dtype": B.dtype.name,
                "scalar": B.format_other(A),
                "b": B,
            },
        )
    elif np.isscalar(A) and np.isscalar(B):
        # Scalars must share a common dtype (or be cast)
        dtA = resolve_scalar_dtype(A)
        dtB = resolve_scalar_dtype(B)
        # Make sure at least one of the dtypes is supported
        if not (dtA in DTypes or dtB in DTypes):
            raise TypeError(f"Not implemented for scalar types {dtA} and {dtB}")
        # If the dtypes are the same, do not cast
        if dtA == dtB:  # type: ignore
            dt = dtA
        # If the dtypes are different, try casting one direction then the other
        elif dtB in DTypes and np.can_cast(A, dtB):
            A = np.dtype(dtB).type(A)  # type: ignore
            dt = dtB
        elif dtA in DTypes and np.can_cast(B, dtA):
            B = np.dtype(dtA).type(B)  # type: ignore
            dt = dtA
        # Cannot safely cast
        else:
            raise TypeError(
                f"Cannot cast between scalars {str(A)} and {str(B)} to supported dtype"
            )
        repMsg = generic_msg(
            cmd=f"efunc3ss{condition.ndim}D",
            args={
                "func": "where",
                "condition": condition,
                "dtype": dt,
                "a": A,
                "b": B,
            },
        )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def histogram(pda: pdarray, bins: int_scalars = 10) -> Tuple[pdarray, pdarray]:
    """
    Compute a histogram of evenly spaced bins over the range of an array.

    Parameters
    ----------
    pda : pdarray
        The values to histogram

    bins : int_scalars
        The number of equal-size bins to use (default: 10)

    Returns
    -------
    (pdarray, Union[pdarray, int64 or float64])
        Bin edges and The number of values present in each bin

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray or if bins is
        not an int.
    ValueError
        Raised if bins < 1
    NotImplementedError
        Raised if pdarray dtype is bool or uint8

    See Also
    --------
    value_counts, histogram2d

    Notes
    -----
    The bins are evenly spaced in the interval [pda.min(), pda.max()].

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> A = ak.arange(0, 10, 1)
    >>> nbins = 3
    >>> h, b = ak.histogram(A, bins=nbins)
    >>> h
    array([3, 3, 4])
    >>> b
    array([0., 3., 6., 9.])

    # To plot, export the left edges and the histogram to NumPy
    >>> plt.plot(b.to_ndarray()[::-1], h.to_ndarray())
    """
    if bins < 1:
        raise ValueError("bins must be 1 or greater")
    b = linspace(pda.min(), pda.max(), bins + 1)
    repMsg = generic_msg(cmd="histogram", args={"array": pda, "bins": bins})
    return create_pdarray(type_cast(str, repMsg)), b


# Typechecking removed due to circular dependencies with arrayview
# @typechecked
def histogram2d(
    x: pdarray, y: pdarray, bins: Union[int_scalars, Sequence[int_scalars]] = 10
) -> Tuple[pdarray, pdarray, pdarray]:
    """
    Compute the bi-dimensional histogram of two data samples with evenly spaced bins

    Parameters
    ----------
    x : pdarray
        A pdarray containing the x coordinates of the points to be histogrammed.

    y : pdarray
        A pdarray containing the y coordinates of the points to be histogrammed.

    bins : int_scalars or [int, int] = 10
        The number of equal-size bins to use.
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        Defaults to 10

    Returns
    -------
    hist : ArrayView, shape(nx, ny)
        The bi-dimensional histogram of samples x and y.
        Values in x are histogrammed along the first dimension and
        values in y are histogrammed along the second dimension.

    x_edges : pdarray
        The bin edges along the first dimension.

    y_edges : pdarray
        The bin edges along the second dimension.

    Raises
    ------
    TypeError
        Raised if x or y parameters are not pdarrays or if bins is
        not an int or (int, int).
    ValueError
        Raised if bins < 1
    NotImplementedError
        Raised if pdarray dtype is bool or uint8

    See Also
    --------
    histogram

    Notes
    -----
    The x bins are evenly spaced in the interval [x.min(), x.max()]
    and y bins are evenly spaced in the interval [y.min(), y.max()].

    Examples
    --------
    >>> x = ak.arange(0, 10, 1)
    >>> y = ak.arange(9, -1, -1)
    >>> nbins = 3
    >>> h, x_edges, y_edges = ak.histogram2d(x, y, bins=nbins)
    >>> h
    array([[0, 0, 3],
           [0, 2, 1],
           [3, 1, 0]])
    >>> x_edges
    array([0.0 3.0 6.0 9.0])
    >>> x_edges
    array([0.0 3.0 6.0 9.0])
    """
    if not isinstance(bins, Sequence):
        x_bins, y_bins = bins, bins
    else:
        if len(bins) != 2:
            raise ValueError(
                "Sequences of bins must contain two elements (num_x_bins, num_y_bins)"
            )
        x_bins, y_bins = bins
    if x_bins < 1 or y_bins < 1:
        raise ValueError("bins must be 1 or greater")
    x_bin_boundaries = linspace(x.min(), x.max(), x_bins + 1)
    y_bin_boundaries = linspace(y.min(), y.max(), y_bins + 1)
    repMsg = generic_msg(
        cmd="histogram2D", args={"x": x, "y": y, "xBins": x_bins, "yBins": y_bins}
    )
    return (
        create_pdarray(type_cast(str, repMsg)).reshape(x_bins, y_bins),
        x_bin_boundaries,
        y_bin_boundaries,
    )


def histogramdd(
    sample: Sequence[pdarray], bins: Union[int_scalars, Sequence[int_scalars]] = 10
) -> Tuple[pdarray, Sequence[pdarray]]:
    """
    Compute the multidimensional histogram of data in sample with evenly spaced bins.

    Parameters
    ----------
    sample : Sequence[pdarray]
        A sequence of pdarrays containing the coordinates of the points to be histogrammed.

    bins : int_scalars or Sequence[int_scalars] = 10
        The number of equal-size bins to use.
        If int, the number of bins for all dimensions (nx=ny=...=bins).
        If [int, int, ...], the number of bins in each dimension (nx, ny, ... = bins).
        Defaults to 10

    Returns
    -------
    hist : ArrayView, shape(nx, ny, ..., nd)
        The multidimensional histogram of pdarrays in sample.
        Values in first pdarray are histogrammed along the first dimension.
        Values in second pdarray are histogrammed along the second dimension and so on.

    edges : List[pdarray]
        A list of pdarrays containing the bin edges for each dimension.


    Raises
    ------
    ValueError
        Raised if bins < 1
    NotImplementedError
        Raised if pdarray dtype is bool or uint8

    See Also
    --------
    histogram

    Notes
    -----
    The bins for each dimension, m, are evenly spaced in the interval [m.min(), m.max()]

    Examples
    --------
    >>> x = ak.arange(0, 10, 1)
    >>> y = ak.arange(9, -1, -1)
    >>> z = ak.where(x % 2 == 0, x, y)
    >>> h, edges = ak.histogramdd((x, y,z), bins=(2,2,5))
    >>> h
    array([[[0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1]],

           [[1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]]])
    >>> edges
    [array([0.0 4.5 9.0]),
     array([0.0 4.5 9.0]),
     array([0.0 1.6 3.2 4.8 6.4 8.0])]
    """
    if not isinstance(sample, Sequence):
        raise ValueError("Sample must be a sequence of pdarrays")
    if len(set(pda.dtype for pda in sample)) != 1:
        raise ValueError("All pdarrays in sample must have same dtype")

    num_dims = len(sample)
    if not isinstance(bins, Sequence):
        bins = [bins] * num_dims
    else:
        if len(bins) != num_dims:
            raise ValueError(
                "Sequences of bins must contain same number of elements as the sample"
            )
    if any(b < 1 for b in bins):
        raise ValueError("bins must be 1 or greater")

    bins = list(bins) if isinstance(bins, tuple) else bins
    sample = list(sample) if isinstance(sample, tuple) else sample
    bin_boundaries = [linspace(a.min(), a.max(), b + 1) for a, b in zip(sample, bins)]
    bins_pda = array(bins)[::-1]
    dim_prod = (cumprod(bins_pda) // bins_pda)[::-1]
    repMsg = generic_msg(
        cmd="histogramdD",
        args={
            "sample": sample,
            "num_dims": num_dims,
            "bins": bins,
            "dim_prod": dim_prod,
            "num_samples": sample[0].size,
        },
    )
    return create_pdarray(type_cast(str, repMsg)).reshape(bins), bin_boundaries


@typechecked
def value_counts(
    pda: pdarray,
) -> Union[Categorical, Tuple[Union[pdarray, Strings], Optional[pdarray]]]:  # type: ignore
    """
    Count the occurrences of the unique values of an array.

    Parameters
    ----------
    pda : pdarray, int64
        The array of values to count

    Returns
    -------
    unique_values : pdarray, int64 or Strings
        The unique values, sorted in ascending order

    counts : pdarray, int64
        The number of times the corresponding unique value occurs

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    See Also
    --------
    unique, histogram

    Notes
    -----
    This function differs from ``histogram()`` in that it only returns
    counts for values that are present, leaving out empty "bins". This
    function delegates all logic to the unique() method where the
    return_counts parameter is set to True.

    Examples
    --------
    >>> A = ak.array([2, 0, 2, 4, 0, 0])
    >>> ak.value_counts(A)
    (array([0, 2, 4]), array([3, 2, 1]))
    """
    return GroupBy(pda).count()


@typechecked
def clip(
    pda: pdarray,
    lo: Union[numeric_scalars, pdarray],
    hi: Union[numeric_scalars, pdarray],
) -> pdarray:
    """
    Clip (limit) the values in an array to a given range [lo,hi]

    Given an array a, values outside the range are clipped to the
    range edges, such that all elements lie in the range.

    There is no check to enforce that lo < hi.  If lo > hi, the corresponding
    value of the array will be set to hi.

    If lo or hi (or both) are pdarrays, the check is by pairwise elements.
    See examples.

    Parameters
    ----------
    pda : pdarray, int64 or float64
        the array of values to clip
    lo  : scalar or pdarray, int64 or float64
        the lower value of the clipping range
    hi  : scalar or pdarray, int64 or float64
        the higher value of the clipping range
    If lo or hi (or both) are pdarrays, the check is by pairwise elements.
        See examples.

    Returns
    -------
    arkouda.pdarrayclass.pdarray
        A pdarray matching pda, except that element x remains x if lo <= x <= hi,
                                                or becomes lo if x < lo,
                                                or becomes hi if x > hi.

    Examples
    --------
    >>> a = ak.array([1,2,3,4,5,6,7,8,9,10])
    >>> ak.clip(a,3,8)
    array([3,3,3,4,5,6,7,8,8,8])
    >>> ak.clip(a,3,8.0)
    array([3.00000000000000000 3.00000000000000000 3.00000000000000000 4.00000000000000000
           5.00000000000000000 6.00000000000000000 7.00000000000000000 8.00000000000000000
           8.00000000000000000 8.00000000000000000])
    >>> ak.clip(a,None,7)
    array([1,2,3,4,5,6,7,7,7,7])
    >>> ak.clip(a,5,None)
    array([5,5,5,5,5,6,7,8,9,10])
    >>> ak.clip(a,None,None)
    ValueError : either min or max must be supplied
    >>> ak.clip(a,ak.array([2,2,3,3,8,8,5,5,6,6],8))
    array([2,2,3,4,8,8,7,8,8,8])
    >>> ak.clip(a,4,ak.array([10,9,8,7,6,5,5,5,5,5]))
    array([4,4,4,4,5,5,5,5,5,5])

    Notes
    -----
    Either lo or hi may be None, but not both.
    If lo > hi, all x = hi.
    If all inputs are int64, output is int64, but if any input is float64, output is float64.

    Raises
    ------
    ValueError
        Raised if both lo and hi are None
    """

    # Check that a range was actually supplied.

    if lo is None and hi is None:
        raise ValueError("Either min or max must be supplied.")

    # If any of the inputs are float, then make everything float.
    # Some type checking is needed, because scalars and pdarrays get cast differently.

    dataFloat = pda.dtype == float
    minFloat = isinstance(lo, float) or (isinstance(lo, pdarray) and lo.dtype == float)
    maxFloat = isinstance(hi, float) or (isinstance(hi, pdarray) and hi.dtype == float)
    forceFloat = dataFloat or minFloat or maxFloat
    if forceFloat:
        if not dataFloat:
            pda = cast(pda, np.float64)
        if lo is not None and not minFloat:
            lo = cast(lo, np.float64) if isinstance(lo, pdarray) else float(lo)
        if hi is not None and not maxFloat:
            hi = cast(hi, np.float64) if isinstance(hi, pdarray) else float(hi)

    # Now do the clipping.

    pda1 = pda
    if lo is not None:
        pda1 = where(pda < lo, lo, pda)
    if hi is not None:
        pda1 = where(pda1 > hi, hi, pda1)
    return pda1
