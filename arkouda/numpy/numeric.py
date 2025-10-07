from enum import Enum
import json
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, TypeVar, Union
from typing import cast as type_cast
from typing import no_type_check

import numpy as np
from typeguard import typechecked

from arkouda.groupbyclass import GroupBy, groupable
from arkouda.numpy.dtypes import ARKOUDA_SUPPORTED_INTS, _datatype_check, bigint
from arkouda.numpy.dtypes import bool_ as ak_bool
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import float64 as ak_float64
from arkouda.numpy.dtypes import int64 as ak_int64
from arkouda.numpy.dtypes import int_scalars, isSupportedNumber, numeric_scalars, resolve_scalar_dtype
from arkouda.numpy.dtypes import str_
from arkouda.numpy.dtypes import str_ as akstr_
from arkouda.numpy.dtypes import uint64 as ak_uint64
from arkouda.numpy.pdarrayclass import (
    argmax,
    broadcast_if_needed,
    create_pdarray,
    parse_single_value,
    pdarray,
    sum,
)
from arkouda.numpy.pdarrayclass import _reduces_to_single_value
from arkouda.numpy.pdarrayclass import all as ak_all
from arkouda.numpy.pdarrayclass import any as ak_any
from arkouda.numpy.pdarraycreation import array, linspace, scalar_array
from arkouda.numpy.sorting import sort
from arkouda.numpy.strings import Strings


NUMERIC_TYPES = [ak_int64, ak_float64, ak_bool, ak_uint64]
ALLOWED_PERQUANT_METHODS = [
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "midpoint",
    "higher",
]  # not supporting 'nearest' at present


if TYPE_CHECKING:
    from arkouda.client import generic_msg, get_array_ranks
    from arkouda.numpy.segarray import SegArray
    from arkouda.pandas.categorical import Categorical
else:
    Categorical = TypeVar("Categorical")
    SegArray = TypeVar("SegArray")
    generic_msg = TypeVar("generic_msg")
    get_array_ranks = TypeVar("get_array_ranks")

__all__ = [
    "cast",
    "abs",
    "fabs",
    "ceil",
    "clip",
    "count_nonzero",
    "eye",
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
    "matmul",
    "nextafter",
    "triu",
    "tril",
    "transpose",
    "vecdot",
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
    "array_equal",
    "putmask",
    "where",
    "histogram",
    "histogram2d",
    "histogramdd",
    "median",
    "value_counts",
    "ErrorMode",
    "quantile",
    "percentile",
    "take",
]


class ErrorMode(Enum):
    strict = "strict"
    ignore = "ignore"
    return_validity = "return_validity"


# TODO: standardize error checking in python interface

# merge_where comes in handy in arctan2 and some other functions.


def _merge_where(new_pda, where, ret):
    new_pda = cast(new_pda, ret.dtype)
    new_pda[where] = ret
    return new_pda


@typechecked
def cast(
    pda: Union[pdarray, Strings, Categorical],  # type: ignore
    dt: Union[np.dtype, type, str, bigint],
    errors: ErrorMode = ErrorMode.strict,
) -> Union[Union[pdarray, Strings, Categorical], Tuple[pdarray, pdarray]]:  # type: ignore
    """
    Cast an array to another dtype.

    Parameters
    ----------
    pda : pdarray, Strings, or Categorical
        The array of values to cast
    dt : np.dtype, type, str, or bigint
        The target dtype to cast values to
    errors : {strict, ignore, return_validity}, default=ErrorMode.strict
        Controls how errors are handled when casting strings to a numeric type
        (ignored for casts from numeric types).
            - strict: raise RuntimeError if *any* string cannot be converted
            - ignore: never raise an error. Uninterpretable strings get
                converted to NaN (float64), -2**63 (int64), zero (uint64 and
                uint8), or False (bool)
            - return_validity: in addition to returning the same output as
              "ignore", also return a bool array indicating where the cast
              was successful.
        Default set to strict.

    Returns
    -------
    Union[Union[pdarray, Strings, Categorical], Tuple[pdarray, pdarray]]
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
    >>> import arkouda as ak
    >>> ak.cast(ak.linspace(1.0,5.0,5), dt=ak.int64)
    array([1 2 3 4 5])

    >>> ak.cast(ak.arange(0,5), dt=ak.float64).dtype
    dtype('float64')

    >>> ak.cast(ak.arange(0,5), dt=ak.bool_)
    array([False True True True True])

    >>> ak.cast(ak.linspace(0,4,5), dt=ak.bool_)
    array([False True True True True])
    """
    from arkouda.client import generic_msg
    from arkouda.pandas.categorical import Categorical  # type: ignore

    if isinstance(pda, pdarray):
        if dt is Strings or akdtype(dt) == str_:
            if pda.ndim > 1:
                raise ValueError("Cannot cast a multi-dimensional pdarray to Strings")
            repMsg = generic_msg(
                cmd=f"castToStrings<{pda.dtype}>",
                args={"name": pda},
            )
            return Strings.from_parts(*(type_cast(str, repMsg).split("+")))
        else:
            dt = akdtype(dt)
            return create_pdarray(
                generic_msg(
                    cmd=f"cast<{pda.dtype},{dt},{pda.ndim}>",
                    args={"name": pda},
                )
            )
    elif isinstance(pda, Strings):
        if dt is Categorical or dt == "Categorical":
            return Categorical(pda)  # type: ignore
        elif dt is Strings or akdtype(dt) == str_:
            return Strings(type_cast(pdarray, array([], dtype="int64")), 0) if pda.size == 0 else pda[:]
        else:
            dt = akdtype(dt)
            repMsg = generic_msg(
                cmd=f"castStringsTo<{dt}>",
                args={
                    "name": pda.entry.name,
                    "opt": errors.name,
                },
            )
            if errors == ErrorMode.return_validity:
                a, b = type_cast(str, repMsg).split("+")
                return create_pdarray(type_cast(str, a)), create_pdarray(type_cast(str, b))
            else:
                return create_pdarray(type_cast(str, repMsg))
    elif isinstance(pda, Categorical):  # type: ignore
        if dt is Strings or dt in ["Strings", "str"] or dt == str_:
            return pda.categories[pda.codes]
        else:
            raise ValueError("Categoricals can only be casted to Strings")
    else:
        raise TypeError("pda must be a pdarray, Strings, or Categorical object")


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
    >>> import arkouda as ak
    >>> ak.abs(ak.arange(-5,-1))
    array([5 4 3 2])

    >>> ak.abs(ak.linspace(-5,-1,5))
    array([5.00000000... 4.00000000... 3.00000000...
    2.00000000... 1.00000000...])
    """
    from arkouda.client import generic_msg

    repMsg = generic_msg(
        cmd=f"abs<{pda.dtype},{pda.ndim}>",
        args={
            "pda": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def fabs(pda: pdarray) -> pdarray:
    """
    Compute the absolute values element-wise, casting to a float beforehand.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing absolute values of the input array elements, casted to float type

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.fabs(ak.arange(-5,-1))
    array([5.00000000000000000 4.00000000000000000 3.00000000000000000 2.00000000000000000])

    >>> ak.fabs(ak.linspace(-5,-1,5))
    array([5.00000000... 4.00000000... 3.00000000...
    2.00000000... 1.00000000...])
    """
    pda_ = cast(pda, ak_float64)

    return abs(pda_)


@typechecked
def ceil(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise ceiling of the array.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    >>> import arkouda as ak
    >>> ak.ceil(ak.linspace(1.1,5.5,5))
    array([2.00000000... 3.00000000... 4.00000000... 5.00000000... 6.00000000...])

    """
    _datatype_check(pda.dtype, [float], "ceil")
    return _general_helper(pda, "ceil", where)


@typechecked
def floor(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise floor of the array.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    >>> import arkouda as ak
    >>> ak.floor(ak.linspace(1.1,5.5,5))
    array([1.00000000... 2.00000000... 3.00000000...
    4.00000000... 5.00000000...])
    """
    _datatype_check(pda.dtype, [float], "floor")
    return _general_helper(pda, "floor", where)


@typechecked
def round(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise rounding of the array.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    >>> import arkouda as ak
    >>> ak.round(ak.array([1.1, 2.5, 3.14159]))
    array([1.00000000... 3.00000000... 3.00000000...])
    """
    _datatype_check(pda.dtype, [float], "round")
    return _general_helper(pda, "round", where)


@typechecked
def trunc(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise truncation of the array.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    >>> import arkouda as ak
    >>> ak.trunc(ak.array([1.1, 2.5, 3.14159]))
    array([1.00000000... 2.00000000... 3.00000000...])
    """
    _datatype_check(pda.dtype, [float], "trunc")
    return _general_helper(pda, "trunc", where)


#   Noted during Sept 2024 rewrite of EfuncMsg.chpl -- although it's "sign" here, inside the
#   chapel code, it's "sgn"


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
    >>> import arkouda as ak
    >>> ak.sign(ak.array([-10, -5, 0, 5, 10]))
    array([-1 -1 0 1 1])
    """
    from arkouda.client import generic_msg

    _datatype_check(pda.dtype, [int, float], "sign")
    repMsg = generic_msg(
        cmd=f"sgn<{pda.dtype},{pda.ndim}>",
        args={
            "pda": pda,
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
    >>> import arkouda as ak
    >>> ak.isfinite(ak.array([1.0, 2.0, ak.inf]))
    array([True True False])
    """
    from arkouda.client import generic_msg

    repMsg = generic_msg(
        cmd=f"isfinite<{pda.ndim}>",
        args={
            "pda": pda,
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
        input array elements are infinite (positive or negative)

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    RuntimeError
        if the underlying pdarray is not float-based

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.isinf(ak.array([1.0, 2.0, ak.inf]))
    array([False False True])
    """
    from arkouda.client import generic_msg

    repMsg = generic_msg(
        cmd=f"isinf<{pda.ndim}>",
        args={
            "pda": pda,
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
    >>> import arkouda as ak
    >>> ak.isnan(ak.array([1.0, 2.0, np.log(-1)]))
    array([False False True])
    """
    from arkouda.client import generic_msg
    from arkouda.numpy.util import is_float, is_numeric

    if is_numeric(pda) and not is_float(pda):
        from arkouda.numpy.pdarraycreation import full

        return full(pda.size, False, dtype=bool)
    elif not is_numeric(pda):
        raise TypeError("isnan only supports pdarray of numeric type.")

    repMsg = generic_msg(
        cmd=f"isnan<{pda.ndim}>",
        args={
            "pda": pda,
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
    >>> import arkouda as ak
    >>> A = ak.array([1, 10, 100])

    Natural log
    >>> ak.log(A)
    array([0.00000000... 2.30258509... 4.60517018...])

    Log base 10
    >>> ak.log(A) / np.log(10)
    array([0.00000000... 1.00000000... 2.00000000...])

    Log base 2
    >>> ak.log(A) / np.log(2)
    array([0.00000000... 3.32192809... 6.64385618...])
    """
    from arkouda.client import generic_msg

    repMsg = generic_msg(
        cmd=f"log<{pda.dtype},{pda.ndim}>",
        args={
            "pda": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def log10(pda: pdarray) -> pdarray:
    """
    Return the element-wise base 10 log of the array.

    Parameters
    ----------
    pda : pdarray
          array to compute on

    Returns
    -------
    pdarray
         pdarray containing base 10 log values of the input array elements

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.arange(1,5)
    >>> ak.log10(a)
    array([0.00000000... 0.30102999... 0.47712125... 0.60205999...])
    """
    from arkouda.client import generic_msg

    repMsg = generic_msg(
        cmd=f"log10<{pda.dtype},{pda.ndim}>",
        args={
            "pda": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def log2(pda: pdarray) -> pdarray:
    """
    Return the element-wise base 2 log of the array.

    Parameters
    ----------
    pda : pdarray
          array to compute on

    Returns
    -------
    pdarray
         pdarray containing base 2 log values of the input array elements

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.arange(1,5)
    >>> ak.log2(a)
    array([0.00000000... 1.00000000... 1.58496250... 2.00000000...])
    """
    from arkouda.client import generic_msg

    repMsg = generic_msg(
        cmd=f"log2<{pda.dtype},{pda.ndim}>",
        args={
            "pda": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def log1p(pda: pdarray) -> pdarray:
    """
    Return the element-wise natural log of one plus the array.

    Parameters
    ----------
    pda : pdarray
          array to compute on

    Returns
    -------
    pdarray
         pdarray containing natural log values of the input array elements,
         adding one before taking the log

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.log1p(ak.arange(1,5))
    array([0.69314718... 1.09861228... 1.38629436... 1.60943791...])
    """
    from arkouda.client import generic_msg

    repMsg = generic_msg(
        cmd=f"log1p<{pda.dtype},{pda.ndim}>",
        args={
            "pda": pda,
        },
    )
    return create_pdarray(repMsg)


@typechecked
def nextafter(
    x1: Union[pdarray, numeric_scalars, bigint], x2: Union[pdarray, numeric_scalars, bigint]
) -> Union[pdarray, float]:
    """
    Return the next floating-point value after `x1` towards `x2`, element-wise.
    Accuracy only guaranteed for 64 bit values.

    Parameters
    ----------
    x1 : pdarray, numeric_scalars, or bigint
        Values to find the next representable value of.
    x2 : pdarray, numeric_scalars, or bigint
        The direction where to look for the next representable value of `x1`.
        If `x1.shape != x2.shape`, they must be broadcastable to a common shape
        (which becomes the shape of the output).

    Returns
    -------
    pdarray or float
        The next representable values of `x1` in the direction of `x2`.
        This is a scalar if both `x1` and `x2` are scalars.

    Examples
    --------
    >>> import arkouda as ak
    >>> eps = np.finfo(np.float64).eps
    >>> ak.nextafter(1, 2) == 1 + eps
     np.True_
    >>> a = ak.array([1, 2])
    >>> b = ak.array([2, 1])
    >>> ak.nextafter(a, b) == ak.array([eps + 1, 2 - eps])
    array([True True])
    """
    from arkouda.client import generic_msg

    return_scalar = True
    x1_: pdarray
    x2_: pdarray
    if isinstance(x1, pdarray):
        return_scalar = False
        if x1.dtype != ak_float64:
            x1_ = cast(x1, ak_float64)
        else:
            x1_ = x1
    else:
        x1_ = type_cast(pdarray, array([x1], ak_float64))
    if isinstance(x2, pdarray):
        return_scalar = False
        if x2.dtype != ak_float64:
            x2_ = cast(x2, ak_float64)
        else:
            x2_ = x2
    else:
        x2_ = type_cast(pdarray, array([x2], ak_float64))

    x1_, x2_, _, _ = broadcast_if_needed(x1_, x2_)

    repMsg = generic_msg(
        cmd=f"nextafter<{x1_.ndim}>",
        args={
            "x1": x1_,
            "x2": x2_,
        },
    )
    return_array = create_pdarray(repMsg)
    if return_scalar:
        return return_array[0]
    return return_array


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
    >>> import arkouda as ak
    >>> ak.exp(ak.arange(1,5))
    array([2.71828182... 7.38905609... 20.0855369... 54.5981500...])

    >>> ak.exp(ak.uniform(4, 1.0, 5.0, seed=1))
    array([63.3448620... 3.80794671... 54.7254287... 36.2344168...])

    """
    from arkouda.client import generic_msg

    repMsg = generic_msg(
        cmd=f"exp<{pda.dtype},{pda.ndim}>",
        args={
            "pda": pda,
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
        A pdarray containing e raised to each of the inputs,
        then subtracting one.

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.expm1(ak.arange(1,5))
    array([1.71828182... 6.38905609... 19.0855369... 53.5981500...])

    >>> ak.expm1(ak.uniform(5,1.0,5.0, seed=1))
    array([62.3448620... 2.80794671... 53.7254287...
        35.2344168... 41.1929399...])
    """
    from arkouda.client import generic_msg

    repMsg = generic_msg(
        cmd=f"expm1<{pda.dtype},{pda.ndim}>",
        args={
            "pda": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def square(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise square of the array.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    >>> import arkouda as ak
    >>> ak.square(ak.arange(1,5))
    array([1 4 9 16])
    """
    _datatype_check(pda.dtype, NUMERIC_TYPES, "floor")
    return _general_helper(pda, "square", where)


@typechecked
def cumsum(pda: pdarray, axis: Optional[Union[int, None]] = None) -> pdarray:
    """
    Return the cumulative sum over the array.

    The sum is inclusive, such that the ``i`` th element of the
    result is the sum of elements up to and including ``i``.

    Parameters
    ----------
    pda : pdarray
    axis : int, optional
        the axis along which to compute the sum

    Returns
    -------
    pdarray
        A pdarray containing cumulative sums for each element of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    ValueError
        Raised if an invalid axis is given

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.cumsum(ak.arange(1,5))
    array([1 3 6 10])

    >>> ak.cumsum(ak.uniform(5,1.0,5.0, seed=1))
    array([4.14859379... 5.48568392... 9.48801240... 13.0780218... 16.8202747...])

    >>> ak.cumsum(ak.randint(0, 2, 5, dtype=ak.bool_, seed=1))
    array([1 1 2 3 4])
    """
    from arkouda.client import generic_msg
    from arkouda.numpy import cast as akcast
    from arkouda.numpy.util import _integer_axis_validation

    _datatype_check(pda.dtype, [int, float, ak_uint64, ak_bool], "cumsum")

    pda_ = pda
    if pda.dtype == "bool":
        pda_ = akcast(pda, int)  # bools are handled as ints, a la numpy

    if pda.ndim == 1:
        axis_ = 0
    elif axis is None:
        axis_ = 0
        pda_ = pda_.flatten()
    else:
        valid, axis_ = _integer_axis_validation(axis, pda_.ndim)
        if not valid:
            raise IndexError(f"{axis} is not valid for the given array.")

    repMsg = generic_msg(
        cmd=f"cumSum<{pda_.dtype},{pda_.ndim}>",
        args={
            "x": pda_,
            "axis": axis_,
            "includeInitial": False,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def cumprod(pda: pdarray, axis: Optional[Union[int, None]] = None) -> pdarray:
    """
    Return the cumulative product over the array.

    The product is inclusive, such that the ``i`` th element of the
    result is the product of elements up to and including ``i``.

    Parameters
    ----------
    pda : pdarray
    axis : int, optional
        the axis along which to compute the product

    Returns
    -------
    pdarray
        A pdarray containing cumulative products for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    ValueError
        Raised if an invalid axis is given

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.cumprod(ak.arange(1,5))
    array([1 2 6 24])

    >>> ak.cumprod(ak.uniform(5,1.0,5.0, seed=1))
    array([4.14859379... 5.54704379... 22.20109135... 79.7021268... 298.2655159...])

    >>> ak.cumprod(ak.randint(0, 2, 5, dtype=ak.bool_, seed=1))
    array([1 0 0 0 0])
    """
    from arkouda.client import generic_msg
    from arkouda.numpy import cast as akcast
    from arkouda.numpy.util import _integer_axis_validation

    _datatype_check(pda.dtype, [int, float, ak_uint64, ak_bool], "cumprod")

    pda_ = pda
    if pda.dtype == "bool":
        pda_ = akcast(pda, int)  # bools are handled as ints, a la numpy

    if pda.ndim == 1:
        axis_ = 0
    elif axis is None:
        axis_ = 0
        pda_ = pda_.flatten()
    else:
        valid, axis_ = _integer_axis_validation(axis, pda_.ndim)
        if not valid:
            raise IndexError(f"{axis} is not valid for the given array.")

    repMsg = generic_msg(
        cmd=f"cumProd<{pda_.dtype},{pda_.ndim}>",
        args={
            "x": pda_,
            "axis": axis_,
            "includeInitial": False,
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
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(-1.5,0.75,4)
    >>> ak.sin(a)
    array([-0.99749498... -0.68163876... 0.00000000... 0.68163876...])
    """
    return _general_helper(pda, "sin", where)


@typechecked
def cos(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise cosine of the array.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(-1.5,0.75,4)
    >>> ak.cos(a)
    array([0.07073720... 0.73168886... 1.00000000... 0.73168886...])
    """
    return _general_helper(pda, "cos", where)


@typechecked
def tan(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise tangent of the array.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(-1.5,0.75,4)
    >>> ak.tan(a)
    array([-14.1014199... -0.93159645... 0.00000000... 0.93159645...])
    """
    return _general_helper(pda, "tan", where)


@typechecked
def arcsin(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise inverse sine of the array. The result is between -pi/2 and pi/2.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(-0.7,0.5,4)
    >>> ak.arcsin(a)
    array([-0.77539749... -0.30469265... 0.10016742... 0.52359877...])
    """
    return _general_helper(pda, "arcsin", where)


@typechecked
def arccos(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise inverse cosine of the array. The result is between 0 and pi.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(-0.7,0.5,4)
    >>> ak.arccos(a)
    array([2.34619382... 1.87548898... 1.47062890... 1.04719755...])
    """
    return _general_helper(pda, "arccos", where)


@typechecked
def arctan(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise inverse tangent of the array. The result is between -pi/2 and pi/2.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(-10.7,10.5,4)
    >>> ak.arctan(a)
    array([-1.47760906... -1.30221689... 1.28737507... 1.47584462...])
    """
    return _general_helper(pda, "arctan", where)


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
    num : pdarray or numeric_scalars
        Numerator of the arctan2 argument.
    denom : pdarray or numeric_scalars
        Denominator of the arctan2 argument.
    where : bool or pdarray, default=True
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
        | Raised if any parameter fails the typechecking
        | Raised if any element of pdarrays num and denom is not a supported type
        | Raised if both num and denom are scalars
        | Raised if where is neither boolean nor a pdarray of boolean

    Examples
    --------
    >>> import arkouda as ak
    >>> x = ak.array([1,-1,-1,1])
    >>> y = ak.array([1,1,-1,-1])
    >>> ak.arctan2(y,x)
    array([0.78539816... 2.35619449... -2.35619449... -0.78539816...])
    """
    from arkouda.client import generic_msg

    if not all(isSupportedNumber(arg) or isinstance(arg, pdarray) for arg in [num, denom]):
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

    if where is True:
        pass
    elif where is False:
        return num / denom  # type: ignore
    elif where.dtype != bool:
        raise TypeError(f"where must have dtype bool, got {where.dtype} instead")

    if isinstance(num, pdarray) or isinstance(denom, pdarray):
        ndim = num.ndim if isinstance(num, pdarray) else denom.ndim  # type: ignore[union-attr]

        #   The code below will create the command string for arctan2vv, arctan2vs or arctan2sv, based
        #   on a and b.

        if isinstance(num, pdarray) and isinstance(denom, pdarray):
            cmdstring = f"arctan2vv<{num.dtype},{ndim},{denom.dtype}>"
            if where is True:
                argdict = {
                    "a": num,
                    "b": denom,
                }
            elif where is False:
                return num / denom  # type: ignore
            else:
                argdict = {
                    "a": num[where],
                    "b": denom[where],
                }
        elif not isinstance(denom, pdarray):
            ts = resolve_scalar_dtype(denom)
            if ts in ["float64", "int64", "uint64", "bool"]:
                cmdstring = "arctan2vs_" + ts + f"<{num.dtype},{ndim}>"  # type: ignore[union-attr]
            else:
                raise TypeError(f"{ts} is not an allowed denom type for arctan2")
            argdict = {"a": num if where is True else num[where], "b": denom}  # type: ignore
        elif not isinstance(num, pdarray):
            ts = resolve_scalar_dtype(num)
            if ts in ["float64", "int64", "uint64", "bool"]:
                cmdstring = "arctan2sv_" + ts + f"<{denom.dtype},{ndim}>"
            else:
                raise TypeError(f"{ts} is not an allowed num type for arctan2")
            argdict = {"a": num, "b": denom if where is True else denom[where]}  # type: ignore

        repMsg = type_cast(
            str,
            generic_msg(cmd=cmdstring, args=argdict),
        )
        ret = create_pdarray(repMsg)
        if where is True:
            return ret
        else:
            new_pda = num / denom  # type : ignore
            return _merge_where(new_pda, where, ret)

    else:
        return scalar_array(arctan2(num, denom) if where else num / denom)


@typechecked
def sinh(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise hyperbolic sine of the array.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(-0.9,0.7,4)
    >>> ak.sinh(a)
    array([-1.02651672... -0.37493812... 0.16743934... 0.75858370...])
    """
    return _general_helper(pda, "sinh", where)


@typechecked
def cosh(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise hyperbolic cosine of the array.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(-0.9,0.7,4)
    >>> ak.cosh(a)
    array([1.43308638... 1.06797874... 1.01392106... 1.25516900...])
    """
    return _general_helper(pda, "cosh", where)


@typechecked
def tanh(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise hyperbolic tangent of the array.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(-0.9,0.7,4)
    >>> ak.tanh(a)
    array([-0.71629787... -0.35107264... 0.16514041... 0.60436777...])
    """
    return _general_helper(pda, "tanh", where)


@typechecked
def arcsinh(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise inverse hyperbolic sine of the array.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(-500,500,4)
    >>> ak.arcsinh(a)
    array([-6.90775627... -5.80915199... 5.80915199... 6.90775627...])
    """
    return _general_helper(pda, "arcsinh", where)


@typechecked
def arccosh(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise inverse hyperbolic cosine of the array.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(1,500,4)
    >>> ak.arccosh(a)
    array([0.00000000... 5.81312608... 6.50328742... 6.90775427...])
    """
    return _general_helper(pda, "arccosh", where)


@typechecked
def arctanh(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Return the element-wise inverse hyperbolic tangent of the array.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(-.999,.999,4)
    >>> ak.arctanh(a)
    array([-3.80020116... -0.34619863... 0.34619863... 3.80020116...])
    """
    return _general_helper(pda, "arctanh", where)


def _general_helper(pda: pdarray, func: str, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Returns the result of the input function acting element-wise on the array.
    This is used for functions that allow a "where" parameter in their arguments.

    Parameters
    ----------
    pda : pdarray
    func : str
        The designated function that is passed in
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the respective function. Elsewhere,
        it will retain its original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray with the given function applied at each element of pda

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray or if is not real or int or uint, or if where is not Boolean
    """
    from arkouda.client import generic_msg

    _datatype_check(pda.dtype, [ak_float64, ak_int64, ak_uint64], func)
    if where is True:
        repMsg = type_cast(
            str,
            generic_msg(
                cmd=f"{func}<{pda.dtype},{pda.ndim}>",
                args={
                    "x": pda,
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
                cmd=f"{func}<{pda.dtype},{pda.ndim}>",
                args={
                    "x": pda[where],
                },
            ),
        )
        return _merge_where(pda[:], where, create_pdarray(repMsg))


@typechecked
def rad2deg(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Converts angles element-wise from radians to degrees.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(0,6.28,4)
    >>> ak.rad2deg(a)
    array([0.00000000... 119.939165... 239.878330... 359.817495...])
    """
    if where is True:
        return 180 * (pda / np.pi)
    elif where is False:
        return pda
    else:
        return _merge_where(pda[:], where, 180 * (pda[where] / np.pi))


@typechecked
def deg2rad(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
    """
    Converts angles element-wise from degrees to radians.

    Parameters
    ----------
    pda : pdarray
    where : bool or pdarray, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.linspace(0,359,4)
    >>> ak.deg2rad(a)
    array([0.00000000... 2.08857733... 4.17715467... 6.26573201...])
    """
    if where is True:
        return np.pi * pda / 180
    elif where is False:
        return pda
    else:
        return _merge_where(pda[:], where, (np.pi * pda[where] / 180))


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
    pda : pdarray, Strings, SegArray, or Categorical \
    or List of pdarray, Strings, SegArray, or Categorical

    full : bool, default=True
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

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.randint(0,65536,3,seed=8675309)
    >>> ak.hash(a,full=False)
    array([6132219720275344925 189443193828113335 14797568559700425150])
    >>> ak.hash(a)
    (array([12228890592923494910 17773622519799422780 16661993598191972647]),
        array([2936052102410048944 15730675498625067356 4746877828134486787]))

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
    from arkouda.client import generic_msg

    if isinstance(pda, (pdarray, Strings, SegArray_, Categorical_)):
        return _hash_single(pda, full) if isinstance(pda, pdarray) else pda.hash()
    elif isinstance(pda, List):
        if any(
            wrong_type := [not isinstance(a, (pdarray, Strings, SegArray_, Categorical_)) for a in pda]
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
    from arkouda.client import generic_msg

    if pda.dtype == bigint:
        return hash(pda.bigint_to_uint_arrays())
    _datatype_check(pda.dtype, [float, int, ak_uint64], "hash")
    hname = "hash128" if full else "hash64"
    repMsg = type_cast(
        str,
        generic_msg(
            cmd=f"{hname}<{pda.dtype},{pda.ndim}>",
            args={
                "x": pda,
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
    from arkouda.client import generic_msg
    from arkouda.numpy.pdarraysetops import concatenate
    from arkouda.pandas.categorical import Categorical

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
            b_code = A.categories.size
        new_codes = where(condition, A.codes, b_code)
        return Categorical.from_codes(new_codes, new_categories, NAvalue=A.NAvalue).reset_categories()

    # both cat
    if isinstance(A, Categorical) and isinstance(B, Categorical):
        if A.codes.size != B.codes.size:
            raise TypeError("Categoricals must be same length")
        if A.categories.size != B.categories.size or not ak_all(A.categories == B.categories):
            A, B = A.standardize_categories([A, B])
        new_codes = where(condition, A.codes, B.codes)
        return Categorical.from_codes(new_codes, A.categories, NAvalue=A.NAvalue).reset_categories()

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
    Return an array with elements chosen from A and B based upon a
    conditioning array. As is the case with numpy.where, the return array
    consists of values from the first array (A) where the conditioning array
    elements are True and from the second array (B) where the conditioning
    array elements are False.

    Parameters
    ----------
    condition : pdarray
        Used to choose values from A or B
    A : str, numeric_scalars, pdarray, Strings, or Categorical
        Value(s) used when condition is True
    B : str, numeric_scalars, pdarray, Strings, or Categorical
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
        an int, np.int64, float, np.float64, bool, pdarray, str, Strings, Categorical
        if pdarray dtypes are not supported or do not match, or multiple
        condition clauses (see Notes section) are applied
    ValueError
        Raised if the shapes of the condition, A, and B pdarrays are unequal

    Examples
    --------
    >>> import arkouda as ak
    >>> a1 = ak.arange(1,10)
    >>> a2 = ak.ones(9, dtype=np.int64)
    >>> cond = a1 < 5
    >>> ak.where(cond,a1,a2)
    array([1 2 3 4 1 1 1 1 1])

    >>> a1 = ak.arange(1,10)
    >>> a2 = ak.ones(9, dtype=np.int64)
    >>> cond = a1 == 5
    >>> ak.where(cond,a1,a2)
    array([1 1 1 1 5 1 1 1 1])

    >>> a1 = ak.arange(1,10)
    >>> a2 = 10
    >>> cond = a1 < 5
    >>> ak.where(cond,a1,a2)
    array([1 2 3 4 10 10 10 10 10])

    >>> s1 = ak.array([f'str {i}' for i in range(10)])
    >>> s2 = 'str 21'
    >>> cond = (ak.arange(10) % 2 == 0)
    >>> ak.where(cond,s1,s2)
    array(['str 0', 'str 21', 'str 2', 'str 21', 'str 4',
    'str 21', 'str 6', 'str 21', 'str 8', 'str 21'])

    >>> c1 = ak.Categorical(ak.array([f'str {i}' for i in range(10)]))
    >>> c2 = ak.Categorical(ak.array([f'str {i}' for i in range(9, -1, -1)]))
    >>> cond = (ak.arange(10) % 2 == 0)
    >>> ak.where(cond,c1,c2)
    array(['str 0', 'str 8', 'str 2', 'str 6', 'str 4',
    'str 4', 'str 6', 'str 2', 'str 8', 'str 0'])

    Notes
    -----
    A and B must have the same dtype and only one conditional clause
    is supported e.g., n < 5, n > 1, which is supported in numpy
    is not currently supported in Arkouda
    """
    from arkouda.client import generic_msg

    if (not isSupportedNumber(A) and not isinstance(A, pdarray)) or (
        not isSupportedNumber(B) and not isinstance(B, pdarray)
    ):
        from arkouda.pandas.categorical import Categorical  # type: ignore

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

    #   The code below creates a command string for wherevv, wherevs, wheresv or wheress,
    #   based on A and B.

    if isinstance(A, pdarray) and isinstance(B, pdarray):
        cmdstring = f"wherevv<{condition.ndim},{A.dtype},{B.dtype}>"

    elif isinstance(A, pdarray) and np.isscalar(B):
        if resolve_scalar_dtype(B) in ["float64", "int64", "uint64", "bool"]:
            ltr = resolve_scalar_dtype(B)
            cmdstring = "wherevs_" + ltr + f"<{condition.ndim},{A.dtype}>"
        else:  # *should* be impossible because of the IsSupportedNumber check
            raise TypeError(f"where does not accept scalar type {resolve_scalar_dtype(B)}")

    elif isinstance(B, pdarray) and np.isscalar(A):
        if resolve_scalar_dtype(A) in ["float64", "int64", "uint64", "bool"]:
            ltr = resolve_scalar_dtype(A)
            cmdstring = "wheresv_" + ltr + f"<{condition.ndim},{B.dtype}>"
        else:  # *should* be impossible because of the IsSupportedNumber check
            raise TypeError(f"where does not accept scalar type {resolve_scalar_dtype(A)}")

    else:  # both are scalars
        if resolve_scalar_dtype(A) in ["float64", "int64", "uint64", "bool"]:
            ta = resolve_scalar_dtype(A)
            if resolve_scalar_dtype(B) in ["float64", "int64", "uint64", "bool"]:
                tb = resolve_scalar_dtype(B)
            else:
                raise TypeError(f"where does not accept scalar type {resolve_scalar_dtype(B)}")
        else:
            raise TypeError(f"where does not accept scalar type {resolve_scalar_dtype(A)}")
        cmdstring = "wheress_" + ta + "_" + tb + f"<{condition.ndim}>"

    repMsg = generic_msg(
        cmd=cmdstring,
        args={
            "condition": condition,
            "a": A,
            "b": B,
        },
    )

    return create_pdarray(type_cast(str, repMsg))


# histogram helper
def _pyrange(count):
    """Simply makes a range(count). For use in histogram* functions
    that, like in numpy, have a 'range' parameter.
    """
    return range(count)


# histogram helper, to avoid typechecker errors
def _conv_dim(sampleDim, rangeDim):
    if rangeDim:
        return (rangeDim[0], rangeDim[1])
    else:
        return (sampleDim.min(), sampleDim.max())


@typechecked
def histogram(
    pda: pdarray,
    bins: int_scalars = 10,
    range: Optional[Tuple[numeric_scalars, numeric_scalars]] = None,
) -> Tuple[pdarray, pdarray]:
    """
    Compute a histogram of evenly spaced bins over the range of an array.

    Parameters
    ----------
    pda : pdarray
        The values to histogram

    bins : int_scalars, default=10
        The number of equal-size bins to use (default: 10)

    range : (minVal, maxVal), optional
        The range of the values to count.
        Values outside of this range are dropped.
        By default, all values are counted.

    Returns
    -------
    (pdarray, Union[pdarray, int64 or float64])
        The number of values present in each bin and the bin edges

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
    If range parameter is provided, the interval is [range[0], range[1]].

    Examples
    --------
    >>> import arkouda as ak
    >>> import matplotlib.pyplot as plt
    >>> A = ak.arange(0, 10, 1)
    >>> nbins = 3
    >>> h, b = ak.histogram(A, bins=nbins)
    >>> h
    array([3 3 4])
    >>> b
    array([0.00000000... 3.00000000... 6.00000000... 9.00000000...])

    To plot, export the left edges and the histogram to NumPy
    >>> b_np = b.to_ndarray()
    >>> import numpy as np
    >>> b_widths = np.diff(b_np)
    >>> plt.bar(b_np[:-1], h.to_ndarray(), width=b_widths, align='edge', edgecolor='black')
    <BarContainer object of 3 artists>
    >>> plt.show() # doctest: +SKIP
    """
    from arkouda.client import generic_msg

    if bins < 1:
        raise ValueError("bins must be 1 or greater")

    minVal, maxVal = _conv_dim(pda, range)

    b = linspace(minVal, maxVal, bins + 1)
    repMsg = generic_msg(
        cmd="histogram", args={"array": pda, "bins": bins, "minVal": minVal, "maxVal": maxVal}
    )
    return create_pdarray(type_cast(str, repMsg)), b


# Typechecking removed due to circular dependencies with arrayview
# @typechecked
def histogram2d(
    x: pdarray,
    y: pdarray,
    bins: Union[int_scalars, Sequence[int_scalars]] = 10,
    range: Optional[
        Tuple[Tuple[numeric_scalars, numeric_scalars], Tuple[numeric_scalars, numeric_scalars]]
    ] = None,
) -> Tuple[pdarray, pdarray, pdarray]:
    """
    Compute the bi-dimensional histogram of two data samples with evenly spaced bins.

    Parameters
    ----------
    x : pdarray
        A pdarray containing the x coordinates of the points to be histogrammed.

    y : pdarray
        A pdarray containing the y coordinates of the points to be histogrammed.

    bins : int_scalars or [int, int], default=10
        The number of equal-size bins to use.
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        Defaults to 10

    range : ((xMin, xMax), (yMin, yMax)), optional
        The ranges of the values in x and y to count.
        Values outside of these ranges are dropped.
        By default, all values are counted.

    Returns
    -------
    Tuple[pdarray, pdarray, pdarray]
        hist : pdarray
            shape(nx, ny)
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
    If range parameter is provided, the intervals are given
    by range[0] for x and range[1] for y..

    Examples
    --------
    >>> import arkouda as ak
    >>> x = ak.arange(0, 10, 1)
    >>> y = ak.arange(9, -1, -1)
    >>> nbins = 3
    >>> h, x_edges, y_edges = ak.histogram2d(x, y, bins=nbins)
    >>> h
    array([array([0.00000000... 0.00000000... 3.00000000...])
           array([0.00000000... 2.00000000... 1.00000000...])
           array([3.00000000... 1.00000000... 0.00000000...])])
    >>> x_edges
    array([0.00000000... 3.00000000... 6.00000000... 9.00000000...])
    >>> y_edges
    array([0.00000000... 3.00000000... 6.00000000... 9.00000000...])
    """
    from arkouda.client import generic_msg

    if not isinstance(bins, Sequence):
        x_bins, y_bins = bins, bins
    else:
        if len(bins) != 2:
            raise ValueError("Sequences of bins must contain two elements (num_x_bins, num_y_bins)")
        x_bins, y_bins = bins
    x_bins, y_bins = int(x_bins), int(y_bins)
    if x_bins < 1 or y_bins < 1:
        raise ValueError("bins must be 1 or greater")

    xMin, xMax = _conv_dim(x, range[0] if range else None)
    yMin, yMax = _conv_dim(y, range[1] if range else None)

    x_bin_boundaries = linspace(xMin, xMax, x_bins + 1)
    y_bin_boundaries = linspace(yMin, yMax, y_bins + 1)
    repMsg = generic_msg(
        cmd="histogram2D",
        args={
            "x": x,
            "y": y,
            "xBins": x_bins,
            "yBins": y_bins,
            "xMin": xMin,
            "xMax": xMax,
            "yMin": yMin,
            "yMax": yMax,
        },
    )
    return (
        create_pdarray(type_cast(str, repMsg)).reshape(x_bins, y_bins),
        x_bin_boundaries,
        y_bin_boundaries,
    )


def histogramdd(
    sample: Sequence[pdarray],
    bins: Union[int_scalars, Sequence[int_scalars]] = 10,
    range: Optional[Sequence[Optional[Tuple[numeric_scalars, numeric_scalars]]]] = None,
) -> Tuple[pdarray, Sequence[pdarray]]:
    """
    Compute the multidimensional histogram of data in sample with evenly spaced bins.

    Parameters
    ----------
    sample : Sequence of pdarray
        A sequence of pdarrays containing the coordinates of the points to be histogrammed.

    bins : int_scalars or Sequence of int_scalars, default=10
        The number of equal-size bins to use.
        If int, the number of bins for all dimensions (nx=ny=...=bins).
        If [int, int, ...], the number of bins in each dimension (nx, ny, ... = bins).
        Defaults to 10

    range : Sequence[optional (minVal, maxVal)], optional
        The ranges of the values to count for each array in sample.
        Values outside of these ranges are dropped.
        By default, all values are counted.

    Returns
    -------
    Tuple[pdarray, Sequence[pdarray]]
        hist : pdarray
            shape(nx, ny, ..., nd)
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
    or in the inverval determined by range[dimension], if provided.

    Examples
    --------
    >>> import arkouda as ak
    >>> x = ak.arange(0, 10, 1)
    >>> y = ak.arange(9, -1, -1)
    >>> z = ak.where(x % 2 == 0, x, y)
    >>> h, edges = ak.histogramdd((x, y,z), bins=(2,2,3))
    >>> h
    array([array([array([0.00000000... 0.00000000... 0.00000000...])
        array([2.00000000... 1.00000000... 2.00000000...])])
        array([array([2.00000000... 1.00000000... 2.00000000...])
        array([0.00000000... 0.00000000... 0.00000000...])])])
    >>> edges
    [array([0.00000000... 4.5 9.00000000...]),
        array([0.00000000... 4.5 9.00000000...]),
        array([0.00000000... 2.66666666... 5.33333333... 8.00000000...])]
    """
    from arkouda.client import generic_msg

    if not isinstance(sample, Sequence):
        raise ValueError("Sample must be a sequence of pdarrays")
    if len(set(pda.dtype for pda in sample)) != 1:
        raise ValueError("All pdarrays in sample must have same dtype")

    num_dims = len(sample)
    if not isinstance(bins, Sequence):
        bins = [bins] * num_dims
    else:
        if len(bins) != num_dims:
            raise ValueError("Sequences of bins must contain same number of elements as the sample")
    if any(b < 1 for b in bins):
        raise ValueError("bins must be 1 or greater")

    if not range:
        range = [None for pda in sample]
    elif len(range) != num_dims:
        raise ValueError("The range sequence contains a different number of elements than the sample")

    range_list = [_conv_dim(sample[i], range[i]) for i in _pyrange(num_dims)]

    bins = list(bins) if isinstance(bins, tuple) else bins
    sample = list(sample) if isinstance(sample, tuple) else sample
    bin_boundaries = [linspace(r[0], r[1], b + 1) for r, b in zip(range_list, bins)]
    d_curr, d_next = 1, 1
    dim_prod = [(d_curr := d_next, d_next := d_curr * int(v))[0] for v in bins[::-1]][::-1]  # noqa: F841
    repMsg = generic_msg(
        cmd="histogramdD",
        args={
            "sample": sample,
            "num_dims": num_dims,
            "bins": bins,
            "rangeMin": [r[0] for r in range_list],
            "rangeMax": [r[1] for r in range_list],
            "dim_prod": dim_prod,
            "num_samples": sample[0].size,
        },
    )
    return create_pdarray(type_cast(str, repMsg)).reshape(bins), bin_boundaries


@typechecked
def value_counts(
    pda: pdarray,
) -> tuple[groupable, pdarray]:
    """
    Count the occurrences of the unique values of an array.

    Parameters
    ----------
    pda : pdarray
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
    >>> import arkouda as ak
    >>> A = ak.array([2, 0, 2, 4, 0, 0])
    >>> ak.value_counts(A)
    (array([0 2 4]), array([3 2 1]))
    """
    return GroupBy(pda.flatten()).size() if pda.ndim > 1 else GroupBy(pda).size()


@typechecked
def clip(
    pda: pdarray,
    lo: Union[numeric_scalars, pdarray],
    hi: Union[numeric_scalars, pdarray],
) -> pdarray:
    """
    Clip (limit) the values in an array to a given range [lo,hi].

    Given an array a, values outside the range are clipped to the
    range edges, such that all elements lie in the range.

    There is no check to enforce that lo < hi.  If lo > hi, the corresponding
    value of the array will be set to hi.

    If lo or hi (or both) are pdarrays, the check is by pairwise elements.
    See examples.

    Parameters
    ----------
    pda : pdarray
        the array of values to clip
    lo : numeric_scalars or pdarray
        the lower value of the clipping range
    hi : numeric_scalars or pdarray
        the higher value of the clipping range
        If lo or hi (or both) are pdarrays, the check is by pairwise elements.
        See examples.

    Returns
    -------
    pdarray
        A pdarray matching pda, except that element x remains x if lo <= x <= hi,
                                                or becomes lo if x < lo,
                                                or becomes hi if x > hi.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([1,2,3,4,5,6,7,8,9,10])
    >>> ak.clip(a,3,8)
    array([3 3 3 4 5 6 7 8 8 8])
    >>> ak.clip(a,3,8.0)
    array([3.00000000... 3.00000000... 3.00000000... 4.00000000...
           5.00000000... 6.00000000... 7.00000000... 8.00000000...
           8.00000000... 8.00000000...])
    >>> ak.clip(a,None,7)
    array([1 2 3 4 5 6 7 7 7 7])
    >>> ak.clip(a,5,None)
    array([5 5 5 5 5 6 7 8 9 10])
    >>> ak.clip(a,None,None) # doctest: +SKIP
    ValueError: Either min or max must be supplied.
    >>> ak.clip(a,ak.array([2,2,3,3,8,8,5,5,6,6]),8)
    array([2 2 3 4 8 8 7 8 8 8])
    >>> ak.clip(a,4,ak.array([10,9,8,7,6,5,5,5,5,5]))
    array([4 4 4 4 5 5 5 5 5 5])

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


def median(pda: pdarray) -> np.float64:
    """
    Compute the median of a given array.  1d case only, for now.

    Parameters
    ----------
    pda: pdarray
        The input data, in pdarray form, numeric type or boolean

    Returns
    -------
    np.float64
        | The median of the entire pdarray
        | The array is sorted, and then if the number of elements is odd,
            the return value is the middle element.  If even, then the
            mean of the two middle elements.

    Examples
    --------
    >>> import arkouda as ak
    >>> pda = ak.array([0,4,7,8,1,3,5,2,-1])
    >>> ak.median(pda)
    np.float64(3.0)
    >>> pda = ak.array([0,1,3,3,1,2,3,4,2,3])
    >>> ak.median(pda)
    np.float64(2.5)

    """
    #  Now do the computation

    if pda.dtype == bool:
        pda_srtd = sort(cast(pda, dt=np.int64))
    else:
        pda_srtd = sort(pda)
    if len(pda_srtd) % 2 == 1:
        return pda_srtd[len(pda_srtd) // 2].astype(np.float64)
    else:
        return ((pda_srtd[len(pda_srtd) // 2] + pda_srtd[len(pda_srtd) // 2 - 1]) / 2.0).astype(
            np.float64
        )


def count_nonzero(pda: pdarray) -> int_scalars:
    """
    Compute the nonzero count of a given array. 1D case only, for now.

    Parameters
    ----------
    pda: pdarray
        The input data, in pdarray form, numeric, bool, or str

    Returns
    -------
    int_scalars
        The nonzero count of the entire pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray with numeric, bool, or str datatype
    ValueError
        Raised if sum applied to the pdarray doesn't come back with a scalar

    Examples
    --------
    >>> import arkouda as ak
    >>> pda = ak.array([0,4,7,8,1,3,5,2,-1])
    >>> ak.count_nonzero(pda)
    np.int64(8)
    >>> pda = ak.array([False,True,False,True,False])
    >>> ak.count_nonzero(pda)
    np.int64(2)
    >>> pda = ak.array(["hello","","there"])
    >>> ak.count_nonzero(pda)
    np.int64(2)

    """
    from arkouda.numpy.dtypes import can_cast
    from arkouda.numpy.util import is_numeric

    #  Handle different data types.

    if is_numeric(pda):
        value = sum((pda != 0).astype(np.int64))
        if can_cast(value, np.int64):
            return np.int64(value)
        else:
            raise ValueError("summing the pdarray did not generate a scalar")
    elif pda.dtype == bool:
        value = sum((pda).astype(np.int64))
        if can_cast(value, np.int64):
            return np.int64(value)
        else:
            raise ValueError("summing the pdarray did not generate a scalar")
    elif pda.dtype == str:
        value = sum((pda != "").astype(np.int64))
        if can_cast(value, np.int64):
            return np.int64(value)
        else:
            raise ValueError("summing the pdarray did not generate a scalar")
    raise TypeError("pda must be numeric, bool, or str pdarray")


def array_equal(pda_a: pdarray, pda_b: pdarray, equal_nan: bool = False) -> bool:
    """
    Compares two pdarrays for equality.
    If neither array has any nan elements, then if all elements are pairwise equal,
    it returns True.
    If equal_Nan is False, then any nan element in either array gives a False return.
    If equal_Nan is True, then pairwise-corresponding nans are considered equal.

    Parameters
    ----------
    pda_a : pdarray
    pda_b : pdarray
    equal_nan : bool, default=False
        Determines how to handle nans

    Returns
    -------
    boolean
      With string data:
         False if one array is type ak.str_ & the other isn't, True if both are ak.str_ & they match.

      With numeric data:
         True if neither array has any nan elements, and all elements pairwise equal.

         True if equal_Nan True, all non-nans pairwise equal & nans in pda_a correspond to nans in pda_b

         False if equal_Nan False, & either array has any nan element.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.randint(0,10,10,dtype=ak.float64)
    >>> b = a
    >>> ak.array_equal(a,b)
    True
    >>> b[9] = np.nan
    >>> ak.array_equal(a,b)
    False
    >>> a[9] = np.nan
    >>> ak.array_equal(a,b)
    False
    >>> ak.array_equal(a,b,True)
    True
    """
    if (pda_a.shape != pda_b.shape) or ((pda_a.dtype == akstr_) ^ (pda_b.dtype == akstr_)):
        return False
    elif equal_nan:
        return bool(ak_all(where(isnan(pda_a), isnan(pda_b), pda_a == pda_b)))
    else:
        return bool(ak_all(pda_a == pda_b))


def putmask(
    A: pdarray, mask: pdarray, Values: pdarray
) -> None:  # doesn't return anything, as A is overwritten in place
    """
    Overwrite elements of A with elements from B based upon a mask array.
    Similar to numpy.putmask, where mask = False, A retains its original value,
    but where mask = True, A is overwritten with the corresponding entry from Values.

    This is similar to ak.where, except that (1) no new pdarray is created, and
    (2) Values does not have to be the same size as A and mask.

    Parameters
    ----------
    A : pdarray
        Value(s) used when mask is False (see Notes for allowed dtypes)
    mask : pdarray
        Used to choose values from A or B, must be same size as A, and of type ak.bool_
    Values : pdarray
        Value(s) used when mask is False (see Notes for allowed dtypes)

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array(np.arange(10))
    >>> ak.putmask (a,a>2,a**2)
    >>> a
    array([0 1 2 9 16 25 36 49 64 81])

    >>> a = ak.array(np.arange(10))
    >>> values = ak.array([3,2])
    >>> ak.putmask (a,a>2,values)
    >>> a
    array([0 1 2 2 3 2 3 2 3 2])

    Raises
    ------
    RuntimeError
        Raised if mask is not same size as A, or if A.dtype and Values.dtype are not
        an allowed pair (see Notes for details).

    Notes
    -----
    | A and mask must be the same size.  Values can be any size.
    | Allowed dtypes for A and Values conform to types accepted by numpy putmask.
    | If A is ak.float64, Values can be ak.float64, ak.int64, ak.uint64, ak.bool_.
    | If A is ak.int64, Values can be ak.int64 or ak.bool_.
    | If A is ak.uint64, Values can be ak.uint64, or ak.bool_.
    | If A is ak.bool_, Values must be ak.bool_.

    Only one conditional clause is supported e.g., n < 5, n > 1.

    multi-dim pdarrays are now implemented.
    """
    from arkouda.client import generic_msg

    ALLOWED_PUTMASK_PAIRS = [
        (ak_float64, ak_float64),
        (ak_float64, ak_int64),
        (ak_float64, ak_uint64),
        (ak_float64, ak_bool),
        (ak_int64, ak_int64),
        (ak_int64, ak_bool),
        (ak_uint64, ak_uint64),
        (ak_uint64, ak_bool),
        (ak_bool, ak_bool),
    ]

    if (A.dtype, Values.dtype) not in ALLOWED_PUTMASK_PAIRS:
        raise RuntimeError(f"Types {A.dtype} and {Values.dtype} are not compatible in putmask.")
    if mask.size != A.size:
        raise RuntimeError("mask and A must be same size in putmask")
    generic_msg(
        cmd=f"putmask<{mask.ndim},{A.dtype},{Values.dtype},{Values.ndim}>",
        args={
            "mask": mask,
            "a": A,
            "v": Values,
        },
    )
    return


def eye(N: int_scalars, M: int_scalars, k: int_scalars = 0, dt: type = ak_float64) -> pdarray:
    """
    Return a pdarray with zeros everywhere except along a diagonal, which is all ones.
    The matrix need not be square.

    Parameters
    ----------
    N : int_scalars
    M : int_scalars
    k : int_scalars, default=0
        | if k = 0, zeros start at element [0,0] and proceed along diagonal
        | if k > 0, zeros start at element [0,k] and proceed along diagonal
        | if k < 0, zeros start at element [k,0] and proceed along diagonal
    dt : type, default=ak_float64
        The data type of the elements in the matrix being returned. Default set to ak_float64

    Returns
    -------
    pdarray
        an array of zeros with ones along the specified diagonal

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.eye(N=4,M=4,k=0,dt=ak.int64)
    array([array([1 0 0 0]) array([0 1 0 0]) array([0 0 1 0]) array([0 0 0 1])])
    >>> ak.eye(N=3,M=3,k=1,dt=ak.float64)
    array([array([0.00000000... 1.00000000... 0.00000000...])
    array([0.00000000... 0.00000000... 1.00000000...])
    array([0.00000000... 0.00000000... 0.00000000...])])
    >>> ak.eye(N=4,M=4,k=-1,dt=ak.bool_)
    array([array([False False False False]) array([True False False False])
    array([False True False False]) array([False False True False])])

    Notes
    -----
    if N = M and k = 0, the result is an identity matrix
    Server returns an error if rank of pda < 2

    """
    from arkouda.client import generic_msg

    cmd = f"eye<{akdtype(dt).name}>"
    args = {
        "N": N,
        "M": M,
        "k": k,
    }
    return create_pdarray(
        generic_msg(
            cmd=cmd,
            args=args,
        )
    )


def triu(pda: pdarray, diag: int_scalars = 0) -> pdarray:
    """
    Return a copy of the pda with the lower triangle zeroed out.

    Parameters
    ----------
    pda : pdarray
    diag : int_scalars, default=0
        | if diag = 0, zeros start just below the main diagonal
        | if diag = 1, zeros start at the main diagonal
        | if diag = 2, zeros start just above the main diagonal
        | etc. Default set to 0.

    Returns
    -------
    pdarray
        a copy of pda with zeros in the lower triangle

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
    >>> ak.triu(a,diag=0)
    array([array([1 2 3 4 5]) array([0 3 4 5 6]) array([0 0 5 6 7])
    array([0 0 0 7 8]) array([0 0 0 0 9])])
    >>> ak.triu(a,diag=1)
    array([array([0 2 3 4 5]) array([0 0 4 5 6]) array([0 0 0 6 7])
    array([0 0 0 0 8]) array([0 0 0 0 0])])
    >>> ak.triu(a,diag=2)
    array([array([0 0 3 4 5]) array([0 0 0 5 6]) array([0 0 0 0 7])
    array([0 0 0 0 0]) array([0 0 0 0 0])])
    >>> ak.triu(a,diag=3)
    array([array([0 0 0 4 5]) array([0 0 0 0 6]) array([0 0 0 0 0])
    array([0 0 0 0 0]) array([0 0 0 0 0])])
    >>> ak.triu(a,diag=4)
    array([array([0 0 0 0 5]) array([0 0 0 0 0]) array([0 0 0 0 0])
    array([0 0 0 0 0]) array([0 0 0 0 0])])

    Notes
    -----
    Server returns an error if rank of pda < 2

    """
    from arkouda.client import generic_msg

    cmd = f"triu<{pda.dtype},{pda.ndim}>"
    args = {
        "array": pda,
        "diag": diag,
    }
    return create_pdarray(
        generic_msg(
            cmd=cmd,
            args=args,
        )
    )


def tril(pda: pdarray, diag: int_scalars = 0) -> pdarray:
    """
    Return a copy of the pda with the upper triangle zeroed out.

    Parameters
    ----------
    pda : pdarray
    diag : int_scalars, optional
        | if diag = 0, zeros start just above the main diagonal
        | if diag = 1, zeros start at the main diagonal
        | if diag = 2, zeros start just below the main diagonal
        | etc. Default set to 0.

    Returns
    -------
    pdarray
        a copy of pda with zeros in the upper triangle

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
    >>> ak.tril(a,diag=4)
    array([array([1 2 3 4 5]) array([2 3 4 5 6]) array([3 4 5 6 7])
    array([4 5 6 7 8]) array([5 6 7 8 9])])
    >>> ak.tril(a,diag=3)
    array([array([1 2 3 4 0]) array([2 3 4 5 6]) array([3 4 5 6 7])
    array([4 5 6 7 8]) array([5 6 7 8 9])])
    >>> ak.tril(a,diag=2)
    array([array([1 2 3 0 0]) array([2 3 4 5 0]) array([3 4 5 6 7])
    array([4 5 6 7 8]) array([5 6 7 8 9])])
    >>> ak.tril(a,diag=1)
    array([array([1 2 0 0 0]) array([2 3 4 0 0]) array([3 4 5 6 0])
    array([4 5 6 7 8]) array([5 6 7 8 9])])
    >>> ak.tril(a,diag=0)
    array([array([1 0 0 0 0]) array([2 3 0 0 0]) array([3 4 5 0 0])
    array([4 5 6 7 0]) array([5 6 7 8 9])])

    Notes
    -----
    Server returns an error if rank of pda < 2

    """
    from arkouda.client import generic_msg

    cmd = f"tril<{pda.dtype},{pda.ndim}>"
    args = {
        "array": pda,
        "diag": diag,
    }
    return create_pdarray(
        generic_msg(
            cmd=cmd,
            args=args,
        )
    )


@typechecked
def transpose(pda: pdarray, axes: Optional[Tuple[int_scalars, ...]] = None) -> pdarray:
    """
    Compute the transpose of a matrix.

    Parameters
    ----------
    pda : pdarray
    axes: Tuple[int,...] Optional, defaults to None
        If specified, must be a tuple which contains a permutation of the axes of pda.

    Returns
    -------
    pdarray
        the transpose of the input matrix
        For a 1-D array, this is the original array.
        For a 2-D array, this is the standard matrix transpose.
        For an n-D array, if axes are given, their order indicates how the axes are permuted.
        If axes is None, the axes are reversed.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([[1,2,3,4,5],[1,2,3,4,5]])
    >>> ak.transpose(a)
    array([array([1 1]) array([2 2]) array([3 3]) array([4 4]) array([5 5])])
    >>> z = ak.array(np.arange(27).reshape(3,3,3))
    >>> ak.transpose(z,axes=(1,0,2))
    array([array([array([0 1 2]) array([9 10 11]) array([18 19 20])]) array([array([3 4 5])
      array([12 13 14]) array([21 22 23])]) array([array([6 7 8]) array([15 16 17]) array([24 25 26])])])

    Raises
    ------
    ValueError
        Raised if axes is not a legitimate permutation of the axes of pda
    TypeError
        Raised if pda is not a pdarray, or if axes is neither a tuple nor None
    """
    from arkouda.client import generic_msg

    if axes is not None:  # if axes was supplied, check that it's valid
        r = tuple(np.arange(pda.ndim))
        if not (np.sort(np.array(axes)) == r).all():
            raise ValueError(f"{axes} is not a valid set of axes for pdarray of rank {pda.ndim}")
    else:  # if axes is None, create a tuple of the axes in reverse order
        axes = tuple(reversed(range(pda.ndim)))

    return create_pdarray(
        generic_msg(
            cmd=f"permuteDims<{pda.dtype},{pda.ndim}>",
            args={
                "name": pda,
                "axes": axes,
            },
        )
    )


def matmul(pdaLeft: pdarray, pdaRight: pdarray) -> pdarray:
    """
    Compute the product of two matrices.

    Parameters
    ----------
    pdaLeft : pdarray
    pdaRight : pdarray

    Returns
    -------
    pdarray
        the matrix product pdaLeft x pdaRight

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([[1,2,3,4,5],[1,2,3,4,5]])
    >>> b = ak.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
    >>> ak.matmul(a,b)
    array([array([55 55]) array([55 55])])

    >>> x = ak.array([[1,2,3],[1.1,2.1,3.1]])
    >>> y = ak.array([[1,1,1],[0,2,2],[0,0,3]])
    >>> ak.matmul(x,y)
    array([array([1.00000000... 5.00000000... 14.0000000...])
    array([1.10000000... 5.30000000... 14.6000000...])])

    Notes
    -----
    Server returns an error if shapes of pdaLeft and pdaRight
    are incompatible with matrix multiplication.

    """
    from arkouda.client import generic_msg

    if pdaLeft.ndim != pdaRight.ndim:
        raise ValueError("matmul requires matrices of matching rank.")
    cmd = f"matmul<{pdaLeft.dtype},{pdaRight.dtype},{pdaLeft.ndim}>"
    args = {
        "x1": pdaLeft,
        "x2": pdaRight,
    }
    return create_pdarray(
        generic_msg(
            cmd=cmd,
            args=args,
        )
    )


@typechecked
def vecdot(
    x1: pdarray, x2: pdarray, axis: Optional[Union[int, None]] = None
) -> Union[numeric_scalars, pdarray]:
    """
    Computes the numpy-style vecdot product of two matrices.  This differs from the
    vecdot function above.  See https://numpy.org/doc/stable/reference/index.html.

    Parameters
    ----------
    x1 : pdarray
    x2 : pdarray
    axis : int, None, optional, default = None

    Returns
    -------
    pdarray, numeric_scalar
        x1 vecdot x2

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([[1,2,3,4,5],[1,2,3,4,5]])
    >>> b = ak.array([[2,2,2,2,2],[2,2,2,2,2]])
    >>> ak.vecdot(a,b)
    array([30 30])
    >>> ak.vecdot(b,a)
    array([30 30])

    Raises
    ------
    ValueError
        Raised if x1 and x2 can not be broadcast to a compatible shape
        or if the last dimensions of x1 and x2 don't match.

    Notes
    -----
    This matches the behavior of numpy vecdot, but as commented above, it is not the
    behavior of the deprecated vecdot, which calls the chapel-side vecdot function.
    This function only uses broadcast_dims, broadcast_to_shape, ak.sum, and the
    binops pdarray multiplication function.  The last dimension of x1 and x2 must
    match, and it must be possible to broadcast them to a compatible shape.
    The deprecated vecdot can be computed via ak.vecdot(a,b,axis=0) on pdarrays
    of matching shape.

    """
    from arkouda.numpy.pdarrayclass import broadcast_to_shape
    from arkouda.numpy.util import broadcast_dims

    #  axis handling in vecdot is unique, and doesn't use one of the standard
    #  validation functions.

    if x1.shape[-1] != x2.shape[-1]:
        raise ValueError("Last dimensions of inputs must match for vecdot.")

    if x1.shape == x2.shape:
        if axis is None:
            axis = -1
        elif -x1.ndim <= axis < x1.ndim:
            pass
        else:
            raise ValueError(f"axis {axis} is out of bounds of given inputs.")
    else:
        if axis is not None:
            raise ValueError("axis param can only be supplied if input shapes match.")
        else:
            axis = -1

    ns = broadcast_dims(x1.shape, x2.shape)
    return sum((broadcast_to_shape(x1, ns) * broadcast_to_shape(x2, ns)), axis=axis)


def quantile(
    a: pdarray,
    q: Optional[Union[numeric_scalars, Tuple[numeric_scalars], np.ndarray, pdarray]] = 0.5,
    axis: Optional[Union[int_scalars, Tuple[int_scalars, ...], None]] = None,
    method: Optional[str] = "linear",
    keepdims: bool = False,
) -> Union[numeric_scalars, pdarray]:  # type : ignore
    """
    Compute the q-th quantile of the data along the specified axis.

    Parameters
    ----------
    a : pdarray
        data whose quantile will be computed
    q : pdarray, Tuple, or np.ndarray
        a scalar, tuple, or np.ndarray of q values for the computation.  All values
        must be in the range 0 <= q <= 1
    axis : None, int scalar, or tuple of int scalars
        the axis or axes along which the quantiles are computed.  The default is None,
        which computes the quantile along a flattened version of the array.
    method : string
        one of "inverted_cdf," "averaged_inverted_cdf", "closest_observation",
        "interpolated_inverted_cdf", "hazen", "weibull", "linear", 'median_unbiased",
        "normal_unbiased", "lower"," higher", "midpoint"
    keepdims : bool
        True if the degenerate axes are to be retained after slicing, False if not


    Returns
    -------
    pdarray or scalar
        If q is a scalar and axis is None, the result is a scalar.
        If q is a scalar and axis is supplied, the result is a pdarray of rank len(axis)
        less than the rank of a.
        If q is an array and axis is None, the result is a pdarray of shape q.shape
        If q is an array and axis is None, the result is a pdarray of rank q.ndim +
        pda.ndim - len(axis).  However, there is an intermediate result which is of rank
        q.ndim + pda.ndim.  If this is not in the compiled ranks, an error will be thrown
        even if the final result would be in the compiled ranks.

    Notes
    -----
    np.quantile also supports the method "nearest," however its behavior does not match
    the numpy documentation, so it's not supported here.
    np.quantile also allows for weighted inputs, but only for the method "inverted_cdf."
    That also is not supported here.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([[1,2,3,4,5],[1,2,3,4,5]])
    >>> q = 0.7
    >>> ak.quantile(a,q,axis=None,method="linear")
    np.float64(4.0)
    >>> ak.quantile(a,q,axis=1,method="lower")
    array([3.00000000... 3.00000000...])
    >>> q = np.array([0.4,0.6])
    >>> ak.quantile(a,q,axis=None,method="weibull")
    array([2.40000000... 3.59999999...])
    >>> a = ak.array([[1,2],[5,3]])
    >>> ak.quantile(a,q,axis=0,method="hazen")
    array([array([2.20000000... 2.29999999...])
        array([3.79999999... 2.69999999...])])

    Raises
    ------
    ValueError
        Raised if scalar q or any value of array q is outside the range [0,1]
        Raised if the method is not one of the 12 supported methods.
        Raised if the result would have a rank not in the compiled ranks.

    """
    from arkouda.client import generic_msg, get_array_ranks

    keepdims = False if keepdims is None else keepdims

    from .manipulation_functions import squeeze

    axis_ = (
        []
        if axis is None
        else (
            [
                axis,
            ]
            if isinstance(axis, ARKOUDA_SUPPORTED_INTS)
            else list(axis)
        )
    )

    q_ = 0.5 if q is None else q if np.isscalar(q) else array(q)  # type: ignore

    if np.isscalar(q_):
        if q_ < 0.0 or q_ > 1.0:  # type: ignore
            raise ValueError("Values of q in quantile must be in range [0,1].")
    else:
        if (q_ < 0.0).any() or (q_ > 1.0).any():  # type: ignore
            raise ValueError("Values of q in quantile must be in range [0,1].")

    if method not in ALLOWED_PERQUANT_METHODS:
        raise ValueError(f"Method {method} is not supported in quantile.")

    # scalar q, no axis slicing

    if np.isscalar(q_) and _reduces_to_single_value(axis_, a.ndim):
        return parse_single_value(
            generic_msg(
                cmd=f"quantile_scalar_no_axis<{a.dtype},{a.ndim}>",
                args={
                    "a": a,
                    "q": q_,
                    "method": method,
                },
            )
        )

    # array q, no axis slicing

    elif not np.isscalar(q_) and _reduces_to_single_value(axis_, a.ndim):
        return create_pdarray(
            generic_msg(
                cmd=f"quantile_array_no_axis<{a.dtype},{a.ndim},{q.ndim}>",  # type: ignore
                args={
                    "a": a,
                    "q": q_,
                    "method": method,
                },
            )
        )

    # scalar q, with axis slicing

    elif np.isscalar(q_) and not _reduces_to_single_value(axis_, a.ndim):
        result = create_pdarray(
            generic_msg(
                cmd=f"quantile_scalar_with_axis<{a.dtype},{a.ndim}>",
                args={
                    "a": a,
                    "q": q_,
                    "axis": axis_,
                    "method": method,
                },
            )
        )
        if keepdims:
            return result
        else:  # squeeze out the degenerate axis or axes
            return squeeze(result, axis)

    # array q, with axis slicing (it's the only option left)

    else:
        if q_.ndim + a.ndim not in get_array_ranks():  # type: ignore
            raise ValueError(
                f"Computation has rank {q_.ndim + a.ndim}, not in compiled ranks"  # type: ignore
            )
        result = create_pdarray(
            generic_msg(
                cmd=f"quantile_array_with_axis<{a.dtype},{a.ndim},{q.ndim}>",  # type: ignore
                args={
                    "a": a,
                    "q": q_,
                    "axis": axis_,
                    "method": method,
                },
            )
        )
        # squeeze out the degenerate axis or axes, which is/are now offset by the rank of q
        if keepdims:
            return result
        else:  # squeeze out the degenerate axis or axes
            squeezeable = tuple([m + q_.ndim for m in axis_])  # type: ignore
            return squeeze(result, squeezeable)


def percentile(
    a: pdarray,
    q: Optional[Union[numeric_scalars, Tuple[numeric_scalars], np.ndarray]] = 0.5,
    axis: Optional[Union[int_scalars, Tuple[int_scalars, ...], None]] = None,
    method: Optional[str] = "linear",
    keepdims: bool = False,
) -> Union[numeric_scalars, pdarray]:  # type : ignore
    """
    Compute the q-th percentile of the data along the specified axis.

    Parameters
    ----------
    a : pdarray
        data whose percentile will be computed
    q : pdarray, Tuple, or np.ndarray
        a scalar, tuple, or np.ndarray of q values for the computation.  All values
        must be in the range 0 <= q <= 100
    axis : None, int scalar, or tuple of int scalars
        the axis or axes along which the percentiles are computed.  The default is None,
        which computes the percenntile along a flattened version of the array.
    method : string
        one of "inverted_cdf," "averaged_inverted_cdf", "closest_observation",
        "interpolated_inverted_cdf", "hazen", "weibull", "linear", 'median_unbiased",
        "normal_unbiased", "lower"," higher", "midpoint"
    keepdims : bool
        True if the degenerate axes are to be retained after slicing, False if not


    Returns
    -------
    pdarray or scalar
        If q is a scalar and axis is None, the result is a scalar.
        If q is a scalar and axis is supplied, the result is a pdarray of rank len(axis)
        less than the rank of a.
        If q is an array and axis is None, the result is a pdarray of shape q.shape
        If q is an array and axis is None, the result is a pdarray of rank q.ndim +
        pda.ndim - len(axis).  However, there is an intermediate result which is of rank
        q.ndim + pda.ndim.  If this is not in the compiled ranks, an error will be thrown
        even if the final result would be in the compiled ranks.

    Notes
    -----
    np.percentile also supports the method "nearest," however its behavior does not match
    the numpy documentation, so it's not supported here.
    np.percentile also allows for weighted inputs, but only for the method "inverted_cdf."
    That also is not supported here.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([[1,2,3,4,5],[1,2,3,4,5]])
    >>> q = 70
    >>> ak.percentile(a,q,axis=None,method="linear")
    np.float64(4.0)
    >>> ak.percentile(a,q,axis=1,method="lower")
    array([3.00000000... 3.00000000...])
    >>> q = np.array([40,60])
    >>> ak.percentile(a,q,axis=None,method="weibull")
    array([2.40000000... 3.59999999...])
    >>> a = ak.array([[1,2],[5,3]])
    >>> ak.percentile(a,q,axis=0,method="hazen")
    array([array([2.20000000... 2.29999999...])
        array([3.79999999... 2.69999999...])])

    Raises
    ------
    ValueError
        Raised if scalar q or any value of array q is outside the range [0,100]
        Raised if the method is not one of the 12 supported methods.
        Raised if the result would have a rank not in the compiled ranks.

    """
    q_ = 50.0 if q is None else q if np.isscalar(q) else array(q)  # type: ignore

    if np.isscalar(q_):
        if q_ < 0.0 or q_ > 100.0:  # type: ignore
            raise ValueError("Values of q in percentile must be in range [0,100].")
    else:
        if (q_ < 0.0).any() or (q_ > 100.0).any():  # type: ignore
            raise ValueError("Values of q in percentile must be in range [0,100].")

    return quantile(a, q_ / 100.0, axis, method, keepdims)  # type: ignore


def take(
    a: Union[pdarray, Strings], indices: Union[numeric_scalars, pdarray], axis: Optional[int] = None
) -> pdarray:
    """
    Take elements from an array along an axis.

    When axis is not None, this function does the same thing as “fancy” indexing (indexing arrays
    using arrays); however, it can be easier to use if you need elements along a given axis.
    A call such as ``np.take(arr, indices, axis=3)`` is equivalent to ``arr[:,:,:,indices,...]``.

    Parameters
    ----------
    a : pdarray
        The array from which to take elements
    indices : numeric_scalars or pdarray
        The indices of the values to extract. Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened input array is used.

    Returns
    -------
    pdarray
        The returned array has the same type as `a`.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([4, 3, 5, 7, 6, 8])
    >>> indices = [0, 1, 4]
    >>> ak.take(a, indices)
    array([4 3 6])

    """
    from arkouda.client import generic_msg
    from arkouda.numpy.util import _integer_axis_validation

    if isinstance(a, Strings):
        from arkouda.numpy.pdarraycreation import arange

        idx = arange(a.size)
        return a[take(idx, indices=indices, axis=axis)]

    if axis is None:
        axis_ = 0
        if a.ndim != 1:
            a = a.flatten()
    else:
        valid, axis_ = _integer_axis_validation(axis, a.ndim)
        if not valid:
            raise IndexError(f"{axis} is not a valid axis for rank {a.ndim}")

    if isinstance(indices, pdarray) and indices.ndim != 1:
        raise ValueError("indices must be 1D")

    if not isinstance(indices, pdarray) and isinstance(indices, list):
        indices_ = array(indices)
    elif not isinstance(indices, pdarray):
        indices_ = array([indices])
    else:
        indices_ = indices

    result = create_pdarray(
        generic_msg(
            cmd=f"takeAlongAxis<{a.dtype},{indices_.dtype},{a.ndim}>",
            args={
                "x": a,
                "indices": indices_,
                "axis": axis_,
            },
        )
    )

    return result
