from __future__ import annotations

import json

from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    get_args,
    no_type_check,
    overload,
)
from typing import Union as _Union
from typing import cast as type_cast

import numpy as np

from typeguard import typechecked

from arkouda.groupbyclass import GroupBy, groupable
from arkouda.numpy.dtypes import (
    ARKOUDA_SUPPORTED_INTS,
    _datatype_check,
    bigint,
    bool_scalars,
    can_cast,
    int_scalars,
    is_supported_bool,
    is_supported_number,
    numeric_and_bool_scalars,
    numeric_scalars,
    resolve_scalar_dtype,
    str_,
    str_scalars,
)
from arkouda.numpy.dtypes import bool_ as ak_bool
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import float64 as ak_float64
from arkouda.numpy.dtypes import int64 as ak_int64
from arkouda.numpy.dtypes import uint64 as ak_uint64
from arkouda.numpy.pdarrayclass import (
    _reduces_to_single_value,
    argmax,
    broadcast_if_needed,
    create_pdarray,
    parse_single_value,
    pdarray,
    sum,
)
from arkouda.numpy.pdarrayclass import all as ak_all
from arkouda.numpy.pdarrayclass import any as ak_any
from arkouda.numpy.pdarraycreation import array, full, linspace
from arkouda.numpy.sorting import sort
from arkouda.numpy.strings import Strings

from ._typing import ArkoudaNumericTypes, BuiltinNumericTypes, NumericDTypeTypes, StringDTypeTypes


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
    from arkouda.numpy.segarray import SegArray
    from arkouda.pandas.categorical import Categorical
else:
    Categorical = TypeVar("Categorical")
    SegArray = TypeVar("SegArray")

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
    "minimum",
    "maximum",
]


class ErrorMode(Enum):
    strict = "strict"
    ignore = "ignore"
    return_validity = "return_validity"


# TODO: standardize error checking in python interface

# merge_where will be phased out as more functions are converted to the ufunc interface.


def _merge_where(new_pda, where, ret):
    new_pda = cast(new_pda, ret.dtype)
    new_pda[where] = ret
    return new_pda


def handle_bools_as_float(x):
    if np.isscalar(x):
        if type(x) in (bool, np.bool_, ak_bool):
            return float(x)
        else:
            return x
    if isinstance(x, pdarray):
        if x.dtype == "bool":
            return x.astype(ak_float64)
        else:
            return x


def handle_bools_as_int(x):
    if np.isscalar(x):
        if type(x) in (bool, np.bool_, ak_bool):
            return int(x)
        else:
            return x
    if isinstance(x, pdarray):
        if x.dtype == "bool":
            return x.astype(ak_int64)
        else:
            return x


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def cast(
    pda: pdarray,
    dt: StringDTypeTypes,
    errors: Literal[ErrorMode.strict, ErrorMode.ignore] = ErrorMode.strict,
) -> Strings: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def cast(
    pda: pdarray,
    dt: NumericDTypeTypes,
    errors: Literal[ErrorMode.strict, ErrorMode.ignore] = ErrorMode.strict,
) -> pdarray: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def cast(
    pda: Strings,
    dt: _Union[ArkoudaNumericTypes, BuiltinNumericTypes, np.dtype[Any], bigint],
    errors: Literal[ErrorMode.return_validity],
) -> Tuple[pdarray, pdarray]: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def cast(
    pda: Strings,
    dt: _Union[ArkoudaNumericTypes, BuiltinNumericTypes, np.dtype[Any], bigint],
    errors: Literal[ErrorMode.strict, ErrorMode.ignore] = ErrorMode.strict,
) -> pdarray: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def cast(
    pda: Strings,
    dt: StringDTypeTypes,
    errors: Literal[ErrorMode.strict, ErrorMode.ignore] = ErrorMode.strict,
) -> Strings: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def cast(
    pda: Strings,
    dt: type["Categorical"],
    errors: Literal[ErrorMode.strict, ErrorMode.ignore] = ErrorMode.strict,
) -> "Categorical": ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def cast(
    pda: "Categorical",
    dt: StringDTypeTypes,
    errors: Literal[ErrorMode.strict, ErrorMode.ignore] = ErrorMode.strict,
) -> Strings: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def cast(
    pda: _Union[pdarray, numeric_scalars],
    dt: _Union[ArkoudaNumericTypes, BuiltinNumericTypes, np.dtype[Any], bigint, None],
    errors: Literal[ErrorMode.strict, ErrorMode.ignore] = ErrorMode.strict,
) -> pdarray: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def cast(
    pda: _Union[pdarray, Strings, "Categorical", numeric_scalars],
    dt: str,
    errors: Literal[ErrorMode.strict, ErrorMode.ignore] = ErrorMode.strict,
) -> _Union[pdarray, Strings, "Categorical"]: ...


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
    from arkouda.core.client import generic_msg
    from arkouda.pandas.categorical import Categorical  # type: ignore

    if isinstance(pda, pdarray):
        if dt is Strings or akdtype(dt) == str_:
            if pda.ndim > 1:
                raise ValueError("Cannot cast a multi-dimensional pdarray to Strings")
            rep_msg = generic_msg(
                cmd=f"castToStrings<{pda.dtype}>",
                args={"name": pda},
            )
            return Strings.from_parts(*(type_cast(str, rep_msg).split("+")))
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
            rep_msg = generic_msg(
                cmd=f"castStringsTo<{dt}>",
                args={
                    "name": pda.entry.name,
                    "opt": errors.name,
                },
            )
            if errors == ErrorMode.return_validity:
                a, b = rep_msg.split("+")
                return create_pdarray(type_cast(str, a)), create_pdarray(type_cast(str, b))
            else:
                return create_pdarray(rep_msg)
    elif isinstance(pda, Categorical):  # type: ignore
        if dt is Strings or dt in ["Strings", "str"] or dt == str_:
            return pda.categories[pda.codes]
        else:
            raise ValueError("Categoricals can only be casted to Strings")
    else:
        raise TypeError("pda must be a pdarray, Strings, or Categorical object")


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
    from arkouda.core.client import generic_msg

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

    rep_msg = generic_msg(
        cmd=f"nextafter<{x1_.ndim}>",
        args={
            "x1": x1_,
            "x2": x2_,
        },
    )
    return_array = create_pdarray(rep_msg)
    if return_scalar:
        return return_array[0]
    return return_array


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
    from arkouda.core.client import generic_msg
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

    rep_msg = generic_msg(
        cmd=f"cumSum<{pda_.dtype},{pda_.ndim}>",
        args={
            "x": pda_,
            "axis": axis_,
            "includeInitial": False,
        },
    )
    return create_pdarray(rep_msg)


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
    from arkouda.core.client import generic_msg
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

    rep_msg = generic_msg(
        cmd=f"cumProd<{pda_.dtype},{pda_.ndim}>",
        args={
            "x": pda_,
            "axis": axis_,
            "includeInitial": False,
        },
    )
    return create_pdarray(rep_msg)


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


HashableItems = Union[pdarray, Strings, SegArray, Categorical]
HashableList = List[HashableItems]


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def hash(pda: HashableItems, full: Literal[True] = True) -> Tuple[pdarray, pdarray]: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def hash(pda: HashableItems, full: Literal[False]) -> pdarray: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def hash(pda: HashableList, full: Literal[True] = True) -> Tuple[pdarray, pdarray]: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def hash(pda: HashableList, full: Literal[False]) -> pdarray: ...


@typechecked
def hash(
    pda: Union[
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
    from arkouda.core.client import generic_msg

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
        rep_msg = generic_msg(
            cmd="hashList",
            args={
                "nameslist": names_list,
                "typeslist": types_list,
                "length": len(expanded_pda),
                "size": len(expanded_pda[0]),
            },
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
    from arkouda.core.client import generic_msg

    if pda.dtype == bigint:
        return hash(pda.bigint_to_uint_arrays())
    _datatype_check(pda.dtype, [float, int, ak_uint64], "hash")
    hname = "hash128" if full else "hash64"
    rep_msg = generic_msg(
        cmd=f"{hname}<{pda.dtype},{pda.ndim}>",
        args={
            "x": pda,
        },
    )
    if full:
        a, b = rep_msg.split("+")
        return create_pdarray(a), create_pdarray(b)
    else:
        return create_pdarray(rep_msg)


@no_type_check
def _str_cat_where(
    condition: pdarray,
    a: Union[str, Strings, Categorical],
    b: Union[str, Strings, Categorical],
) -> Union[Strings, Categorical]:
    # added @no_type_check because mypy can't handle Categorical not being declared
    # sooner, but there are circular dependencies preventing that
    from arkouda.core.client import generic_msg
    from arkouda.numpy.pdarraysetops import concatenate
    from arkouda.pandas.categorical import Categorical

    if isinstance(a, str) and isinstance(b, (Categorical, Strings)):
        # This allows us to assume if a str is present it is B
        a, b, condition = b, a, ~condition

    # one cat and one str
    if isinstance(a, Categorical) and isinstance(b, str):
        is_in_categories = a.categories == b
        if ak_any(is_in_categories):
            new_categories = a.categories
            b_code = argmax(is_in_categories)
        else:
            new_categories = concatenate([a.categories, array([b])])
            b_code = a.categories.size
        new_codes = where(condition, a.codes, b_code)
        return Categorical.from_codes(new_codes, new_categories, na_value=a.na_value).reset_categories()

    # both cat
    if isinstance(a, Categorical) and isinstance(b, Categorical):
        if a.codes.size != b.codes.size:
            raise TypeError("Categoricals must be same length")
        if a.categories.size != b.categories.size or not ak_all(a.categories == b.categories):
            a, b = a.standardize_categories([a, b])
        new_codes = where(condition, a.codes, b.codes)
        return Categorical.from_codes(new_codes, a.categories, na_value=a.na_value).reset_categories()

    # one strings and one str
    if isinstance(a, Strings) and isinstance(b, str):
        new_lens = where(condition, a.get_lengths(), len(b))
        rep_msg = generic_msg(
            cmd="segmentedWhere",
            args={
                "seg_str": a,
                "other": b,
                "is_str_literal": True,
                "new_lens": new_lens,
                "condition": condition,
            },
        )
        return Strings.from_return_msg(rep_msg)

    # both strings
    if isinstance(a, Strings) and isinstance(b, Strings):
        if a.size != b.size:
            raise TypeError("Strings must be same length")
        new_lens = where(condition, a.get_lengths(), b.get_lengths())
        rep_msg = generic_msg(
            cmd="segmentedWhere",
            args={
                "seg_str": a,
                "other": b,
                "is_str_literal": False,
                "new_lens": new_lens,
                "condition": condition,
            },
        )
        return Strings.from_return_msg(rep_msg)

    raise TypeError("ak.where is not supported between Strings and Categorical")


# -------------------------
# pdarray condition overloads
# -------------------------


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def where(
    condition: pdarray,
    A: Union[numeric_and_bool_scalars, pdarray],
    B: Union[numeric_and_bool_scalars, pdarray],
) -> pdarray: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def where(
    condition: pdarray,
    A: Union[str_scalars, Strings],
    B: Union[str_scalars, Strings],
) -> Strings: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def where(condition: pdarray, A: Categorical, B: Categorical) -> Categorical: ...


# -------------------------
# scalar-bool condition overloads
# -------------------------


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def where(
    condition: bool_scalars,
    A: numeric_and_bool_scalars,
    B: numeric_and_bool_scalars,
) -> numeric_and_bool_scalars: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def where(
    condition: bool_scalars,
    A: pdarray,
    B: Union[numeric_and_bool_scalars, pdarray],
) -> pdarray: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def where(
    condition: bool_scalars,
    A: Union[numeric_and_bool_scalars, pdarray],
    B: pdarray,
) -> pdarray: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def where(condition: bool_scalars, A: str_scalars, B: str_scalars) -> str_scalars: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def where(
    condition: bool_scalars,
    A: Strings,
    B: Union[str_scalars, Strings],
) -> Strings: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def where(
    condition: bool_scalars,
    A: Union[str_scalars, Strings],
    B: Strings,
) -> Strings: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def where(condition: bool_scalars, A: Categorical, B: Categorical) -> Categorical: ...


# -------------------------
# Broad fallbacks (last)
# -------------------------


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def where(
    condition: pdarray,
    A: Union[str_scalars, numeric_and_bool_scalars, pdarray, Strings, Categorical],
    B: Union[str_scalars, numeric_and_bool_scalars, pdarray, Strings, Categorical],
) -> Union[str_scalars, numeric_and_bool_scalars, pdarray, Strings, Categorical]: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def where(
    condition: bool_scalars,
    A: Union[str_scalars, numeric_and_bool_scalars, pdarray, Strings, Categorical],
    B: Union[str_scalars, numeric_and_bool_scalars, pdarray, Strings, Categorical],
) -> Union[str_scalars, numeric_and_bool_scalars, pdarray, Strings, Categorical]: ...


@typechecked
def where(
    condition: Union[bool_scalars, pdarray],
    x: Union[str_scalars, numeric_and_bool_scalars, pdarray, Strings, "Categorical"],
    y: Union[str_scalars, numeric_and_bool_scalars, pdarray, Strings, "Categorical"],
) -> Union[str_scalars, numeric_and_bool_scalars, pdarray, Strings, "Categorical"]:
    """
    Return values chosen from ``x`` and ``y`` depending on ``condition``.

    Broadcasting rules (NumPy-style):
    - If ``condition`` is a boolean ``pdarray``, the output shape is the broadcasted
      shape of ``condition`` and any array-like operands among ``x`` and ``y``.
    - If ``condition`` is a scalar bool:
        * if any operand is array-like, broadcast the condition to the broadcasted
          shape of the array-like operands and return an array-like result
        * if both operands are scalars, return a scalar

    Notes
    -----
    * For numeric/bool ``pdarray`` inputs, Arkouda requires matching dtypes between
      array operands (server-side constraint).
    * Broadcasting is performed for ``pdarray`` operands via ``broadcast_to``.
    * ``Strings`` and ``Categorical`` are treated as 1D; we validate broadcast
      compatibility but do not currently broadcast these objects.
    """
    from arkouda.core.client import generic_msg
    from arkouda.numpy.util import broadcast_shapes, broadcast_to
    from arkouda.pandas.categorical import Categorical

    operand = Union[str_scalars, numeric_and_bool_scalars, pdarray, Strings, Categorical]

    # -------- helpers --------

    def _is_arraylike(obj: operand) -> bool:
        return isinstance(obj, (pdarray, Strings, Categorical))

    def _shape_of(obj: operand) -> tuple[int, ...]:
        if isinstance(obj, pdarray):
            return obj.shape
        # Strings/Categorical: treat as 1D
        return (int(getattr(obj, "size")),)

    def _maybe_broadcast_pd(val: operand, shape: tuple[int, ...]) -> operand:
        if isinstance(val, pdarray) and val.shape != shape:
            return broadcast_to(val, shape)
        return val

    def _is_numeric_like(obj: object) -> bool:
        return is_supported_number(obj) or is_supported_bool(obj) or isinstance(obj, pdarray)

    def _is_string_like(obj: operand) -> bool:
        return isinstance(obj, (str, Strings, Categorical))

    # -------- classify path --------

    x_num = _is_numeric_like(x)
    y_num = _is_numeric_like(y)

    # Mixed numeric/string is an error (matches previous behavior)
    if x_num != y_num:
        raise TypeError(
            "both x and y must be numeric/bool (including pdarray) OR both must be "
            "string-like (str, Strings, Categorical)"
        )

    # -------- compute broadcast shape (if any array-like exists) --------

    shapes: list[tuple[int, ...]] = []
    if isinstance(condition, pdarray):
        shapes.append(condition.shape)
    if _is_arraylike(x):
        shapes.append(_shape_of(x))
    if _is_arraylike(y):
        shapes.append(_shape_of(y))

    out_shape: tuple[int, ...] | None = None
    if shapes:
        try:
            out_shape = broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"where: operands could not be broadcast together: {tuple(shapes)}") from e

    # -------- scalar condition fast-path / broadcast condition --------
    if is_supported_bool(condition):
        chosen = x if bool(condition) else y

        if out_shape is None:
            return chosen

        if np.isscalar(chosen):
            if _is_numeric_like(chosen):
                from arkouda.numpy.pdarraycreation import full

                return full(
                    out_shape,
                    type_cast(Union[str_scalars, numeric_and_bool_scalars], chosen),
                    dtype=None,
                )

        if isinstance(chosen, pdarray):
            return broadcast_to(chosen, out_shape) if chosen.shape != out_shape else chosen

        return chosen

    # from here on, condition must be pdarray (and mypy should know that)
    assert isinstance(condition, pdarray)
    cond = condition  # pdarray-typed alias for mypy
    if out_shape is not None and cond.shape != out_shape:
        cond = broadcast_to(cond, out_shape)

    # -------- string-like path --------

    if not x_num and not y_num:
        if not _is_string_like(x) or not _is_string_like(y):
            raise TypeError("both x and y must be string-like (str, Strings, Categorical) on this path")

        # Strings/Categorical are not broadcasted; require they already match out_shape when present
        if out_shape is not None:
            for obj, name in ((x, "x"), (y, "y")):
                if isinstance(obj, (Strings, Categorical)) and _shape_of(obj) != out_shape:
                    raise ValueError(
                        f"where: {name} has shape {_shape_of(obj)} but broadcast shape is {out_shape}; "
                        "Strings/Categorical broadcasting is not supported"
                    )

        return _str_cat_where(cond, x, y)

    # -------- numeric/bool path --------

    if out_shape is not None:
        x = _maybe_broadcast_pd(x, out_shape)
        y = _maybe_broadcast_pd(y, out_shape)

    # mypy-friendly aliases
    xn = x  # operand
    yn = y  # operand

    # -------- server dispatch --------

    if isinstance(xn, pdarray) and isinstance(yn, pdarray):
        cmdstring = f"wherevv<{cond.ndim},{xn.dtype},{yn.dtype}>"

    elif isinstance(xn, pdarray) and np.isscalar(yn):
        ltr = resolve_scalar_dtype(yn)
        if ltr in ["float64", "int64", "uint64", "bool"]:
            cmdstring = "wherevs_" + ltr + f"<{cond.ndim},{xn.dtype}>"
        else:
            raise TypeError(f"where does not accept scalar type {ltr}")

    elif isinstance(yn, pdarray) and np.isscalar(xn):
        ltr = resolve_scalar_dtype(xn)
        if ltr in ["float64", "int64", "uint64", "bool"]:
            cmdstring = "wheresv_" + ltr + f"<{cond.ndim},{yn.dtype}>"
        else:
            raise TypeError(f"where does not accept scalar type {ltr}")

    else:
        # both scalars here implies cond is pdarray (broadcasted), so result is array-like
        tx = resolve_scalar_dtype(xn)
        ty = resolve_scalar_dtype(yn)
        if tx not in ["float64", "int64", "uint64", "bool"]:
            raise TypeError(f"where does not accept scalar type {tx}")
        if ty not in ["float64", "int64", "uint64", "bool"]:
            raise TypeError(f"where does not accept scalar type {ty}")
        cmdstring = "wheress_" + tx + "_" + ty + f"<{cond.ndim}>"

    rep_msg = generic_msg(
        cmd=cmdstring,
        args={
            "condition": cond,
            "a": xn,  # server arg names
            "b": yn,
        },
    )
    return create_pdarray(type_cast(str, rep_msg))


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

    range : (min_val, max_val), optional
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
    from arkouda.core.client import generic_msg

    if bins < 1:
        raise ValueError("bins must be 1 or greater")

    min_val, max_val = _conv_dim(pda, range)

    b = linspace(min_val, max_val, bins + 1)
    rep_msg = generic_msg(
        cmd="histogram", args={"array": pda, "bins": bins, "minVal": min_val, "maxVal": max_val}
    )
    return create_pdarray(type_cast(str, rep_msg)), b


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

    range : ((x_min, x_max), (y_min, y_max)), optional
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
    from arkouda.core.client import generic_msg

    if not isinstance(bins, Sequence):
        x_bins, y_bins = bins, bins
    else:
        if len(bins) != 2:
            raise ValueError("Sequences of bins must contain two elements (num_x_bins, num_y_bins)")
        x_bins, y_bins = bins
    x_bins, y_bins = int(x_bins), int(y_bins)
    if x_bins < 1 or y_bins < 1:
        raise ValueError("bins must be 1 or greater")

    x_min, x_max = _conv_dim(x, range[0] if range else None)
    y_min, y_max = _conv_dim(y, range[1] if range else None)

    x_bin_boundaries = linspace(x_min, x_max, x_bins + 1)
    y_bin_boundaries = linspace(y_min, y_max, y_bins + 1)
    rep_msg = generic_msg(
        cmd="histogram2D",
        args={
            "x": x,
            "y": y,
            "xBins": x_bins,
            "yBins": y_bins,
            "xMin": x_min,
            "xMax": x_max,
            "yMin": y_min,
            "yMax": y_max,
        },
    )
    return (
        create_pdarray(type_cast(str, rep_msg)).reshape(x_bins, y_bins),
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

    range : Sequence[optional (min_val, max_val)], optional
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
    from arkouda.core.client import generic_msg

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
    rep_msg = generic_msg(
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
    return create_pdarray(rep_msg).reshape(bins), bin_boundaries


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

    data_float = pda.dtype == float
    min_float = isinstance(lo, float) or (isinstance(lo, pdarray) and lo.dtype == float)
    max_float = isinstance(hi, float) or (isinstance(hi, pdarray) and hi.dtype == float)
    force_float = data_float or min_float or max_float
    if force_float:
        if not data_float:
            pda = cast(pda, np.float64)
        if lo is not None and not min_float:
            lo = cast(lo, np.float64) if isinstance(lo, pdarray) else float(lo)
        if hi is not None and not max_float:
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
    #    from arkouda.numpy.dtypes import can_cast
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
    if (pda_a.shape != pda_b.shape) or ((pda_a.dtype == str_) ^ (pda_b.dtype == str_)):
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
    from arkouda.core.client import generic_msg

    allowed_putmask_pairs = [
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

    if (A.dtype, Values.dtype) not in allowed_putmask_pairs:
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
    from arkouda.core.client import generic_msg

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
    from arkouda.core.client import generic_msg

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
    from arkouda.core.client import generic_msg

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
    from arkouda.core.client import generic_msg

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


def _matmul2d(pda_L: pdarray, pda_R: pdarray) -> pdarray:
    from arkouda.core.client import generic_msg

    if pda_L.ndim == 2 and pda_R.ndim == 2:
        if pda_L.shape[-1] != pda_R.shape[0]:
            raise ValueError(
                f"Mismatch in dimensions of arguments for matmul: {pda_L.shape} and {pda_R.shape}"
            )
        else:
            cmd = f"matmul<{pda_L.dtype},{pda_R.dtype},{pda_L.ndim}>"
            args = {
                "x1": pda_L,
                "x2": pda_R,
            }
            return create_pdarray(
                generic_msg(
                    cmd=cmd,
                    args=args,
                )
            )
    else:
        raise ValueError(
            f"Mismatch in dimensions of arguments for matmul: {pda_L.shape} and {pda_R.shape}"
        )


@typechecked
def matmul(pda_L: pdarray, pda_R: pdarray) -> pdarray:
    """
    Compute the product of two matrices.
    If both are 1D, this returns a simple dot product.
    If both are 2D, it returns a conventional matrix multiplication.
    If only one is 1D, the result matches the "dot" function, so we use that.
    If neither is 1D and at least one is > 2D, then broadcasting is involved.
    If pda_L's shape is [(leftshape),m,n] and pda_R's shape is [(rightshape),n,k],
    then the result will have shape [(common shape),m,k] where common shape is a
    shape that both leftshape and rightshape can be broadcast to.

    Parameters
    ----------
    pda_L : pdarray
    pda_R : pdarray

    Returns
    -------
    pdarray
        the matrix product pda_L x pda_R

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

    Raises
    ------
    ValueError
        Raised if shapes are incompatible with matrix multiplication.

    """
    from arkouda.core.client import generic_msg
    from arkouda.numpy.pdarrayclass import dot
    from arkouda.numpy.util import broadcast_shapes, broadcast_to

    # Disallow scalar arguments.  That's not a matmul thing.

    if pda_L.ndim < 1 or pda_R.ndim < 1:
        raise ValueError("Scalar arguments not allowed for matmul.")

    # Handle the 1D and 1D case.

    elif pda_L.ndim == 1 and pda_R.ndim == 1:
        if pda_L.size != pda_R.size:
            raise ValueError(
                f"Mismatch in dimensions of arguments for matmul: {pda_L.shape} and {pda_R.shape}"
            )
        else:
            return dot(pda_L, pda_R)

    # Handle the 2D and 2D case.

    elif pda_L.ndim == 2 and pda_R.ndim == 2:
        return _matmul2d(pda_L, pda_R)

    # Handle both singleton 1D cases (i.e. either left or right is 1D, but not both)

    elif pda_L.ndim == 1:
        if pda_L.size != pda_R.shape[-2]:
            raise ValueError(
                f"Mismatch in dimensions of arguments for matmul: {pda_L.shape} and {pda_R.shape}"
            )
        else:
            return dot(pda_L, pda_R)

    elif pda_R.ndim == 1:
        if pda_R.size != pda_L.shape[-1]:
            raise ValueError(
                f"Mismatch in dimensions of arguments for matmul: {pda_L.shape} and {pda_R.shape}"
            )
        else:
            return dot(pda_L, pda_R)

    # Handle the multi-dim cases.  This involves finding a common shape for broadcast.

    else:
        left_preshape = pda_L.shape[0:-2]  # pull off all but last 2 dims of
        right_preshape = pda_R.shape[0:-2]  # both shapes
        try:
            tmp_preshape = broadcast_shapes(left_preshape, right_preshape)
            tmp_pda_lshape = list(tmp_preshape)
            tmp_pda_lshape.append(pda_L.shape[-2])  # restore the last 2 dims
            tmp_pda_lshape.append(pda_L.shape[-1])  # of the left shape
            new_pda_lshape = tuple(tmp_pda_lshape)
            tmp_pda_rshape = list(tmp_preshape)  # now do the same jiggery-pokery
            tmp_pda_rshape.append(pda_R.shape[-2])  # with the shape of pda_R
            tmp_pda_rshape.append(pda_R.shape[-1])
            new_pda_rshape = tuple(tmp_pda_rshape)
            new_pda_l = broadcast_to(pda_L, new_pda_lshape)
            new_pda_r = broadcast_to(pda_R, new_pda_rshape)  # args are now ready
            cmd = f"multidimmatmul<{pda_L.dtype},{new_pda_l.ndim},{pda_R.dtype},{new_pda_r.ndim}>"
            args = {
                "a": new_pda_l,
                "b": new_pda_r,
            }
            return create_pdarray(
                generic_msg(
                    cmd=cmd,
                    args=args,
                )
            )
        except Exception as e:
            raise ValueError(
                f"Mismatch in dimensions of arguments for matmul: {pda_L.shape} and {pda_R.shape}"
            ) from e


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
    This function only uses broadcast_to, broadcast_shapes, ak.sum, and the
    binops pdarray multiplication function.  The last dimension of x1 and x2 must
    match, and it must be possible to broadcast them to a compatible shape.
    The deprecated vecdot can be computed via ak.vecdot(a,b,axis=0) on pdarrays
    of matching shape.

    """
    from arkouda.numpy.util import broadcast_shapes, broadcast_to

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

    ns = broadcast_shapes(x1.shape, x2.shape)
    return sum((broadcast_to(x1, ns) * broadcast_to(x2, ns)), axis=axis)


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
    from arkouda.core.client import generic_msg, get_array_ranks

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
    a: Union[pdarray, Strings],
    indices: Union[numeric_scalars, pdarray, Iterable[numeric_scalars]],
    axis: Optional[int] = None,
) -> pdarray:
    """
    Take elements from an array along an axis.

    When axis is not None, this function does the same thing as “fancy” indexing (indexing arrays
    using arrays); however, it can be easier to use if you need elements along a given axis.
    A call such as ``np.take(arr, indices, axis=3)`` is equivalent to ``arr[:,:,:,indices,...]``.

    Parameters
    ----------
    a : pdarray or Strings
        The array from which to take elements
    indices : numeric_scalars or pdarray or Iterable[numeric_scalars]
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
    from arkouda.core.client import generic_msg
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

    indices_: pdarray
    if isinstance(indices, Iterable):
        indices_ = type_cast(pdarray, array(indices))
    elif isinstance(indices, get_args(numeric_scalars)):
        indices_ = array([indices])
    elif isinstance(indices, pdarray):
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


@typechecked
def minimum(
    x1: Union[pdarray, numeric_scalars],
    x2: Union[pdarray, numeric_scalars],
) -> Union[pdarray, numeric_scalars]:
    """
    Return the element-wise minimum of ``x1`` and ``x2``.

    Where either value is NaN, return NaN; otherwise return the lesser of the two.
    If ``x1`` and ``x2`` are not the same shape, they are broadcast to a mutual
    shape when possible.

    Parameters
    ----------
    x1 : pdarray or numeric_scalars
        First argument in comparison.
    x2 : pdarray or numeric_scalars
        Second argument in comparison.

    Returns
    -------
    Union[pdarray, numeric_scalars]
        The element-wise minimum of ``x1`` and ``x2``. If both inputs are scalars,
        returns a scalar (via ``numpy.minimum``). Otherwise returns a ``pdarray``
        (after broadcasting as needed).

    Raises
    ------
    ValueError
        Raised if ``x1`` and ``x2`` cannot be broadcast to a mutual shape.
    """
    from arkouda.numpy import isnan, where
    from arkouda.numpy.imports import nan
    from arkouda.numpy.pdarraycreation import full
    from arkouda.numpy.util import broadcast_shapes, broadcast_to

    def _scalar_is_nan(x: numeric_scalars) -> bool:
        return isinstance(x, (float, np.floating)) and np.isnan(x)

    # pdarray / pdarray
    if isinstance(x1, pdarray) and isinstance(x2, pdarray):
        tx1, tx2 = x1, x2

        if tx1.shape != tx2.shape:
            try:
                mutual = broadcast_shapes(tx1.shape, tx2.shape)
                if mutual != tx1.shape:
                    tx1 = broadcast_to(tx1, mutual)
                if mutual != tx2.shape:
                    tx2 = broadcast_to(tx2, mutual)
            except Exception as e:
                raise ValueError(f"Shapes {x1.shape} and {x2.shape} incompatible for minimum.") from e

        return type_cast(
            pdarray,
            where(isnan(tx1), tx1, where(isnan(tx2), tx2, where(tx1 < tx2, tx1, tx2))),
        )

    # pdarray / scalar
    if isinstance(x1, pdarray) and not isinstance(x2, pdarray):
        s2 = type_cast(numeric_scalars, x2)
        if _scalar_is_nan(s2):
            # If scalar is NaN, result is all NaNs
            return full(x1.shape, nan)
        return type_cast(pdarray, where(isnan(x1), x1, where(x1 < s2, x1, s2)))

    # scalar / pdarray
    if not isinstance(x1, pdarray) and isinstance(x2, pdarray):
        s1 = type_cast(numeric_scalars, x1)
        if _scalar_is_nan(s1):
            return full(x2.shape, nan)
        return type_cast(pdarray, where(isnan(x2), x2, where(s1 < x2, s1, x2)))

    # scalar / scalar
    s1 = type_cast(numeric_scalars, x1)
    s2 = type_cast(numeric_scalars, x2)
    return np.minimum(s1, s2)


@typechecked
def maximum(
    x1: Union[pdarray, numeric_scalars],
    x2: Union[pdarray, numeric_scalars],
) -> Union[pdarray, numeric_scalars]:
    from arkouda.numpy import isnan, where
    from arkouda.numpy.imports import nan
    from arkouda.numpy.pdarraycreation import full
    from arkouda.numpy.util import broadcast_shapes, broadcast_to

    def _scalar_is_nan(x: numeric_scalars) -> bool:
        return isinstance(x, (float, np.floating)) and np.isnan(x)

    # pdarray / pdarray
    if isinstance(x1, pdarray) and isinstance(x2, pdarray):
        tx1, tx2 = x1, x2

        if tx1.shape != tx2.shape:
            try:
                mutual = broadcast_shapes(tx1.shape, tx2.shape)
                if mutual != tx1.shape:
                    tx1 = broadcast_to(tx1, mutual)
                if mutual != tx2.shape:
                    tx2 = broadcast_to(tx2, mutual)
            except Exception as e:
                raise ValueError(f"Shapes {x1.shape} and {x2.shape} incompatible for maximum.") from e

        return type_cast(
            pdarray,
            where(isnan(tx1), tx1, where(isnan(tx2), tx2, where(tx1 > tx2, tx1, tx2))),
        )

    # pdarray / scalar
    if isinstance(x1, pdarray) and not isinstance(x2, pdarray):
        s2 = type_cast(numeric_scalars, x2)
        if _scalar_is_nan(s2):
            return full(x1.shape, nan)
        return type_cast(pdarray, where(isnan(x1), x1, where(x1 > s2, x1, s2)))

    # scalar / pdarray
    if not isinstance(x1, pdarray) and isinstance(x2, pdarray):
        s1 = type_cast(numeric_scalars, x1)
        if _scalar_is_nan(s1):
            return full(x2.shape, nan)
        return type_cast(pdarray, where(isnan(x2), x2, where(s1 > x2, s1, x2)))

    # scalar / scalar
    s1 = type_cast(numeric_scalars, x1)
    s2 = type_cast(numeric_scalars, x2)
    return np.maximum(s1, s2)


#   New implementation of arctan2 using ufunc tools.


@typechecked
def arctan2(
    x1: Union[pdarray, numeric_and_bool_scalars],
    x2: Union[pdarray, numeric_and_bool_scalars],
    /,
    out: Optional[pdarray] = None,
    *,
    where: Optional[Union[bool_scalars, pdarray]] = None,
) -> Union[pdarray, numeric_scalars]:
    """
    Return the element-wise inverse tangent of the array pair. The result chosen is the
    signed angle in radians between the ray ending at the origin and passing through the
    point (1,0), and the ray ending at the origin and passing through the point (denom, num).
    The result is between -pi and pi.

    Parameters
    ----------
    x1 : pdarray or numeric_scalars
        Numerator of the arctan2 argument.
    x2 : pdarray or numeric_scalars
        Denominator of the arctan2 argument.
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
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

    Notes
    -----
    Unlike numpy, arkouda requires out if where is used.
    """
    #  arctan2-specific checking

    def _is_supported(arg):
        return is_supported_number(arg) or is_supported_bool(arg)

    #  arctan2 allows bools, but treats them as floats.

    x1 = handle_bools_as_float(x1)
    x2 = handle_bools_as_float(x2)

    #   for arctan2, out must be float.  Any other specification is an error.

    if out is not None and out.dtype != ak_float64:
        raise TypeError(f"Cannot return arctan2 result as type {out.dtype}")

    if not all(_is_supported(arg) or isinstance(arg, pdarray) for arg in [x1, x2]):
        raise TypeError(
            f"Unsupported types {type(x1)} and/or {type(x2)}. Supported "
            "types are numeric scalars and pdarrays."
        )

    # end of arctan2-specific checking.  Now use ufunc style call.

    return _apply_where_out(
        _arctan2_impl,  # the actual function
        x1,
        x2,
        where=where,
        out=out,
        dtype=ak_float64,
        scalar_op=np.arctan2,
    )


# ------------------------------------------------------------
# arctan2 implementation helpers
# ------------------------------------------------------------


@typechecked
def _arctan2_impl(
    x1: Union[pdarray, numeric_and_bool_scalars],
    x2: Union[pdarray, numeric_and_bool_scalars],
    out=None,
    dtype=None,
) -> pdarray:
    from arkouda.core.client import generic_msg

    # Require at least one pdarray argument

    if not (isinstance(x1, pdarray) or isinstance(x2, pdarray)):
        raise TypeError("_arctan2_ helper function called with no pdarray arguments.")

    # Determine ndim from the pdarray argument

    if isinstance(x1, pdarray):
        ndim = x1.ndim
    elif isinstance(x2, pdarray):
        ndim = x2.ndim

    argdict = {"a": x1, "b": x2}

    # Build command string

    if isinstance(x1, pdarray) and isinstance(x2, pdarray):
        cmdstring = f"arctan2vv<{x1.dtype},{ndim},{x2.dtype}>"

    elif isinstance(x1, pdarray) and not isinstance(x2, pdarray):
        ts = resolve_scalar_dtype(x2)
        if ts in ["float64", "int64", "uint64", "bool"]:
            cmdstring = "arctan2vs_" + ts + f"<{x1.dtype},{ndim}>"
        else:
            raise TypeError(f"{ts} is not an allowed x2 type for arctan2")

    elif not isinstance(x1, pdarray) and isinstance(x2, pdarray):
        ts = resolve_scalar_dtype(x1)
        if ts in ["float64", "int64", "uint64", "bool"]:
            cmdstring = "arctan2sv_" + ts + f"<{x2.dtype},{ndim}>"
        else:
            raise TypeError(f"{ts} is not an allowed x1 type for arctan2")

    rep_msg = generic_msg(cmd=cmdstring, args=argdict)
    res = create_pdarray(rep_msg)

    if out is None:
        return res
    out[:] = res
    return out


#  Helper functions for ufuncs (things that implement "out" and "where").

# ============================================================
# scalar normalization
# ============================================================


def _normalize_scalar(x):
    """
    Convert NumPy scalar types (np.bool_, np.int64, etc.)
    into Python scalars so broadcasting behaves sanely.
    """
    if isinstance(x, np.generic):
        return x.item()
    return x


# ============================================================
# "where: normalization and policy enforcement
# ============================================================


def _normalize_where(where):
    """Treat None as True (no mask), normalize NumPy scalars."""
    where = _normalize_scalar(where)

    if isinstance(where, pdarray) and where.dtype != bool:
        raise TypeError(f"where must have dtype bool, got {where.dtype} instead")

    if np.isscalar(where) and type(where) is not bool:
        raise TypeError(f"where must have dtype bool, got {type(where)} instead")

    return True if where is None else where


def _validate_out(out):
    if out is None:
        return None
    if isinstance(out, pdarray):
        return out
    raise TypeError("out must be a pdarray or None")


def _require_out_for_mask(where, out):
    """Arkouda policy: if where is anything other than True/None, out must be provided."""
    if out is None and where is not True:
        raise ValueError("out must be provided when where is not None/True")


# ============================================================
# minimal dispatcher abstraction
# ============================================================


def _dispatch(op, *ops, out=None, dtype=None):
    """
    op: server-side implementation callable.

    Convention:
      - if out is None: return a new pdarray
      - if out is provided: fill it and return it
    """
    if out is None:
        return op(*ops, dtype=dtype)
    else:
        op(*ops, out=out, dtype=dtype)
        return out


# ============================================================
# Helpers
# ============================================================


def _scalar_to_bool(where_scalar):
    """Strict-ish conversion for scalar where values."""
    if isinstance(where_scalar, bool):
        return where_scalar
    # after _normalize_scalar, np.bool_ becomes bool, but keep this safe:
    raise TypeError("where must be None/True, bool, or pdarray[bool]")


# ============================================================
# unified where/out utility
# ============================================================


def _apply_where_out(
    op,  # the arkouda function
    *operands,
    where=None,
    out=None,
    dtype=None,
    scalar_op=None,  # Python/NumPy scalar implementation
):
    from arkouda.numpy.numeric import where as ak_where
    from arkouda.numpy.util import broadcast_shapes as bcast_shapes
    from arkouda.numpy.util import broadcast_to as bcast_to

    # --- normalize scalars ---
    ops = tuple(_normalize_scalar(x) for x in operands)
    where_n = _normalize_where(where)
    out_n = _validate_out(out)
    # out_n = _normalize_scalar(out)

    # --------------------------------------------------------
    # Scalar-only fast path: return scalar
    # Only when out is None and where is "no mask"
    # --------------------------------------------------------
    if out_n is None and where_n is True and all(np.isscalar(x) for x in ops):
        if scalar_op is None:
            raise RuntimeError("scalar_op required for scalar-only inputs")
        return scalar_op(*ops)

    # --------------------------------------------------------
    # Enforce where/out policy (unlike numpy, arkouda requires out if where is used).
    # --------------------------------------------------------
    _require_out_for_mask(where_n, out_n)

    # --------------------------------------------------------
    # Determine target shape
    #
    # - If out is a pdarray, out.shape is THE target shape to which everything is broadcast.
    # - Otherwise (no out pdarray), broadcast operand(s) (and where pdarray) to find shape.
    # --------------------------------------------------------
    out_pd = out_n if isinstance(out_n, pdarray) else None

    if out_pd is not None:
        shape = out_pd.shape
    else:
        shape_args = [x.shape if isinstance(x, pdarray) else () for x in ops]
        if isinstance(where_n, pdarray):
            shape_args.append(where_n)
        shape = bcast_shapes(*shape_args)

    # --------------------------------------------------------
    # Broadcast operand(s) (and where) to target shape
    # --------------------------------------------------------
    b_ops = tuple(bcast_to(x, shape) for x in ops)

    # --------------------------------------------------------
    # FAST PATH: no mask
    # --------------------------------------------------------
    if where_n is True:
        if out_pd is None:
            return _dispatch(op, *b_ops, out=None, dtype=dtype)

        _dispatch(op, *b_ops, out=out_pd, dtype=dtype)
        return out_pd

    # --------------------------------------------------------
    # MASKED PATH (out required, by policy)
    # --------------------------------------------------------
    if out_pd is None:
        # out could be a scalar here; policy requires out be provided,
        # but caller may pass scalar. To match numpy, if inputs are scalar and out is provided,
        # out must be >=1-element pdarray.
        raise ValueError("out must be a pdarray when where is not None/True")

    # Normalize where to pdarray mask of target shape
    if isinstance(where_n, pdarray):
        cond = bcast_to(where_n, shape)
    else:
        cond = bcast_to(_scalar_to_bool(where_n), shape)

    # Compute full temporary result (server-side)
    tmp = _dispatch(op, *b_ops, out=None, dtype=dtype)

    # Merge
    merged = ak_where(cond, tmp, out_pd)

    # Write into out in place and return it
    out_pd[:] = merged
    return out_pd


"""
Generic unary ufunc-style wrappers for Arkouda, with consistent handling of:
  - where/out semantics (requires out when where is not True)
  - broadcasting (x and where)
  - scalar fast-path (including scalar + out broadcast-fill)
  - res_dtype determined from a table (no dtype= parameter)
  - out casting governed by can_cast(res_dtype, out.dtype) else TypeError
  - Chapel command dispatch via (chapel_name, argname, ndim, dtype?) conventions
  - per-function bypass/precompute hooks and input-cast hooks
  - per-function extra argument mapping (round: decimals -> n)

This module assumes the following utilities/types exist in your codebase:
  - pdarray
  - ak_where, bcast_to, bcast_shapes
  - _normalize_where, _normalize_scalar, _validated_bool_scalar
  - resolve_scalar_dtype
  - arkouda dtypes: ak_bool, ak_int64, ak_uint64, ak_float64
  - can_cast from arkouda.numpy.dtypes
  - create_pdarray, generic_msg
  - ak.full
"""

# -----------------------------------------------------------------------------
# Core dtype resolution (res_dtype comes from table + a few explicit overrides)
# -----------------------------------------------------------------------------

_PRESERVE = {
    "abs",
    "ceil",
    "floor",
    "trunc",
    # "round" is preserve except bool->float64 (handled explicitly below)
    "square",  # preserve except bool->int64 (handled via precompute)
}

_ALWAYS_FLOAT = {
    "fabs",
    "log",
    "log2",
    "log10",
    "log1p",
    "exp",
    "expm1",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
}

_ALWAYS_BOOL = {
    "isfinite",
    "isinf",
    "isnan",
}


def _validated_bool_scalar(x: object) -> bool:
    # Accept Python bool
    if isinstance(x, bool):
        return x
    # Accept NumPy bool_ (common when users pass np.bool_(True))
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError("where must be a bool scalar or a pdarray")


def resolve_output_dtype(func_name: str, input_dtype: Any) -> Any:
    """
    Determine res_dtype (dtype of the computed result before casting to out.dtype).

    input_dtype is:
      - a pdarray dtype when input is pdarray
      - resolve_scalar_dtype(x) (string or dtype) when input is scalar

    Rules for determining dtype
      - ceil/floor/trunc preserve input dtype
      - round preserves except bool -> float64
      - square preserves, but bool is handled via precompute to int64
      - sign rejects bool
      - fabs always float64
      - isfinite/isinf/isnan always bool
      - log/exp/trig/hyperbolic always float64
    """
    if func_name in _PRESERVE:
        return input_dtype

    if func_name == "round":
        # Closest match to numpy behavior
        # bool -> float64, otherwise preserve
        if input_dtype == ak_bool or input_dtype == "bool":
            return ak_float64
        return input_dtype

    if func_name in _ALWAYS_FLOAT:
        return ak_float64

    if func_name in _ALWAYS_BOOL:
        return ak_bool

    if func_name == "sign":
        if input_dtype == ak_bool or input_dtype == "bool":
            raise TypeError("sign does not support bool input")
        return input_dtype

    raise ValueError(f"Unknown function: {func_name}")


# -----------------------------------------------------------------------------
# Scalar casting helper (used for scalar-only path and scalar fill for out)
# -----------------------------------------------------------------------------


def cast_scalar_to_dtype(val: Any, dtype: Any) -> Any:
    # Accept dtype objects or strings like "float64"
    if dtype == ak_float64 or dtype == "float64":
        return float(val)
    if dtype == ak_int64 or dtype == "int64":
        return int(val)
    if dtype == ak_uint64 or dtype == "uint64":
        return int(val)  # assumes already validated non-negative
    if dtype == ak_bool or dtype == "bool":
        return bool(val)
    raise TypeError(f"Unsupported dtype {dtype}")


# -----------------------------------------------------------------------------
# Spec + hooks
# -----------------------------------------------------------------------------

ExtraArgsBuilder = Callable[[Mapping[str, Any]], dict[str, Any]]
PrecomputeHook = Callable[[Sequence[pdarray], Any], Optional[pdarray]]
ValidateHook = Callable[[str, Sequence[Any]], None]
InputCastHook = Callable[[pdarray, Any], pdarray]  # (bx, res_dtype) -> bx_casted


@dataclass(frozen=True)
class UfuncSpec:
    name: str  # Python-facing name
    chapel_name: str  # used in cmd=...
    argname: str  # "x" or "pda" in args dict
    output_dtype_resolver: Callable[[str, Any], Any]
    scalar_op: Optional[Callable[..., Any]] = None
    validate: Optional[ValidateHook] = None
    precompute: Optional[PrecomputeHook] = None
    extra_args_builder: Optional[ExtraArgsBuilder] = None
    input_cast: Optional[InputCastHook] = None
    chapel_accepts_dtype: bool = True  # isfinite/isinf/isnan float-only path uses False


# -----------------------------------------------------------------------------
# Generic Chapel dispatch
# -----------------------------------------------------------------------------


def _dispatch_unary_chapel(
    spec: UfuncSpec, bx: pdarray, res_dtype: Any, extra_args: dict[str, Any]
) -> pdarray:
    from arkouda.core.client import generic_msg

    args = {spec.argname: bx}
    args.update(extra_args)

    if spec.chapel_accepts_dtype:
        cmd = f"{spec.chapel_name}<{bx.dtype},{bx.ndim}>"
    else:
        cmd = f"{spec.chapel_name}<{bx.ndim}>"

    rep_msg = generic_msg(cmd=cmd, args=args)
    return create_pdarray(rep_msg).astype(res_dtype)


# -----------------------------------------------------------------------------
# Generic unary ufunc handler
# -----------------------------------------------------------------------------


def ufunc_unary(
    spec: UfuncSpec,
    x: Any,
    /,
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
    **kwargs: Any,
):
    from arkouda.numpy.numeric import where as ak_where
    from arkouda.numpy.util import broadcast_shapes as bcast_shapes
    from arkouda.numpy.util import broadcast_to as bcast_to

    where_n = _normalize_where(where)

    if out is not None and not isinstance(out, pdarray):
        raise TypeError("out must be a pdarray or None")

    # Where policy
    if out is None and where_n is not True:
        raise ValueError("out must be provided when where is not None/True")

    # Optional validation on raw inputs
    if spec.validate is not None:
        spec.validate(spec.name, (x,))

    # -------------------------
    # Scalar-only fast path
    # -------------------------
    if not isinstance(x, pdarray):
        if spec.scalar_op is None:
            raise RuntimeError(f"scalar_op required for scalar-only inputs: {spec.name}")

        in_dt_scalar = resolve_scalar_dtype(x)
        res_dtype = spec.output_dtype_resolver(spec.name, in_dt_scalar)

        if out is not None and not can_cast(res_dtype, out.dtype):
            raise TypeError(f"Cannot cast {res_dtype} to out dtype {out.dtype}")

        # scalar_op may accept kwargs (round uses decimals)
        scalar_val = spec.scalar_op(x, **kwargs)

        if out is None:
            return cast_scalar_to_dtype(scalar_val, res_dtype)

        # scalar + out: broadcast-fill out.shape, then where/out merge
        shape = out.shape
        fill_val = cast_scalar_to_dtype(scalar_val, out.dtype)
        tmp = bcast_to(fill_val, shape)

        if where_n is True:
            out[:] = tmp
            return out

        cond = bcast_to(
            where_n if isinstance(where_n, pdarray) else _validated_bool_scalar(where_n),
            shape,
        )

        assert out is not None  # we know it isn't, but mypy doesn't know that
        out[:] = ak_where(cond, tmp, out)
        return out

    # -------------------------
    # pdarray path
    # -------------------------
    in_dt_array = x.dtype
    res_dtype = spec.output_dtype_resolver(spec.name, in_dt_array)

    if out is not None and not can_cast(res_dtype, out.dtype):
        raise TypeError(f"Cannot cast {res_dtype} to out dtype {out.dtype}")

    # Broadcast shape (x and where)
    shape = (
        out.shape
        if out is not None
        else bcast_shapes(x.shape, *([where_n.shape] if isinstance(where_n, pdarray) else []))
    )
    bx = bcast_to(x, shape)

    # Per-function input cast (e.g., fabs always float64, bool->float64 for many float funcs,
    # round bool->float64)
    if spec.input_cast is not None:
        bx = spec.input_cast(bx, res_dtype)

    # Precompute/bypass hook (after broadcasting)
    tmp = spec.precompute((bx,), res_dtype) if spec.precompute is not None else None

    if tmp is None:
        extra_args = spec.extra_args_builder(kwargs) if spec.extra_args_builder is not None else {}
        if kwargs and spec.extra_args_builder is None:
            raise TypeError(f"{spec.name} got unexpected keyword arguments: {', '.join(kwargs.keys())}")

        tmp = _dispatch_unary_chapel(spec, bx, res_dtype, extra_args)

    # If out is provided, cast computed result to out.dtype before assignment/merge
    if out is not None and tmp.dtype != out.dtype:
        tmp = tmp.astype(out.dtype)

    if where_n is True:
        if out is None:
            return tmp
        out[:] = tmp
        return out

    cond = bcast_to(
        where_n if isinstance(where_n, pdarray) else _validated_bool_scalar(where_n),
        shape,
    )

    assert out is not None  # we know it isn't, but mypy doesn't
    out[:] = ak_where(cond, tmp, out)
    return out


# -----------------------------------------------------------------------------
# Helper hooks used by specs
# -----------------------------------------------------------------------------


def validate_sign(name: str, ops: Sequence[Any]) -> None:
    (x,) = ops
    if isinstance(x, pdarray):
        if x.dtype == ak_bool:
            raise TypeError("sign does not support bool input")
    else:
        if resolve_scalar_dtype(x) == "bool":
            raise TypeError("sign does not support bool input")


def cast_all_to_float64(bx: pdarray, res_dtype: Any) -> pdarray:
    # fabs: always convert input to float64 before processing
    if bx.dtype != ak_float64:
        return bx.astype(ak_float64)
    return bx


def cast_bool_to_float64_only(bx: pdarray, res_dtype: Any) -> pdarray:
    # For float-producing functions that specifically want bool->float64
    if bx.dtype == ak_bool:
        return bx.astype(ak_float64)
    return bx


def cast_bool_to_float64_for_round(bx: pdarray, res_dtype: Any) -> pdarray:
    # round: bool -> float64; otherwise preserve
    if bx.dtype == ak_bool:
        return bx.astype(ak_float64)
    return bx


def abs_precompute(ops: Sequence[pdarray], res_dtype: Any) -> Optional[pdarray]:
    (bx,) = ops
    if bx.dtype in (ak_bool, ak_uint64):
        return bx.copy()
    return None


def ceil_precompute(ops: Sequence[pdarray], res_dtype: Any) -> Optional[pdarray]:
    (bx,) = ops
    if bx.dtype in (ak_int64, ak_uint64, ak_bool):
        return bx.copy()
    return None


def floor_precompute(ops: Sequence[pdarray], res_dtype: Any) -> Optional[pdarray]:
    (bx,) = ops
    if bx.dtype in (ak_int64, ak_uint64, ak_bool):
        return bx.copy()
    return None


def trunc_precompute(ops: Sequence[pdarray], res_dtype: Any) -> Optional[pdarray]:
    (bx,) = ops
    if bx.dtype in (ak_int64, ak_uint64, ak_bool):
        return bx.copy()
    return None


def round_precompute(ops: Sequence[pdarray], res_dtype: Any) -> Optional[pdarray]:
    (bx,) = ops
    # Integers: numpy effectively leaves them unchanged; bypass server.
    if bx.dtype in (ak_int64, ak_uint64):
        return bx.copy()
    # Bool: numpy returns float16; we return float64 (your chosen policy).
    if bx.dtype == ak_bool:
        return bx.astype(ak_float64)
    return None


def square_precompute(ops: Sequence[pdarray], res_dtype: Any) -> Optional[pdarray]:
    (bx,) = ops
    if bx.dtype == ak_bool:
        # Return int64; bypass Chapel
        return bx.astype(ak_int64)
    return None


def sign_precompute(ops: Sequence[pdarray], res_dtype: Any) -> Optional[pdarray]:
    (bx,) = ops
    if bx.dtype == ak_uint64:
        # result = (input != 0)
        return bx != 0
    return None


def isfinite_precompute(ops: Sequence[pdarray], res_dtype: Any) -> Optional[pdarray]:
    (bx,) = ops
    if bx.dtype != ak_float64:
        return full(bx.shape, True, ak_bool)
    return None


def isinf_precompute(ops: Sequence[pdarray], res_dtype: Any) -> Optional[pdarray]:
    (bx,) = ops
    if bx.dtype != ak_float64:
        return full(bx.shape, False, ak_bool)
    return None


def isnan_precompute(ops: Sequence[pdarray], res_dtype: Any) -> Optional[pdarray]:
    (bx,) = ops
    if bx.dtype != ak_float64:
        return full(bx.shape, False, ak_bool)
    return None


def round_args_builder(py_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    decimals = py_kwargs.get("decimals", 0)
    return {"n": decimals}


# -----------------------------------------------------------------------------
# Specs (registry)
# -----------------------------------------------------------------------------

ABS_SPEC = UfuncSpec(
    name="abs",
    chapel_name="abs",
    argname="pda",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.abs,
    precompute=abs_precompute,
)

FABS_SPEC = UfuncSpec(
    name="fabs",
    chapel_name="abs",  # chapel-side abs
    argname="pda",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.fabs,
    input_cast=cast_all_to_float64,
    precompute=abs_precompute,
)

CEIL_SPEC = UfuncSpec(
    name="ceil",
    chapel_name="ceil",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.ceil,
    precompute=ceil_precompute,
)

FLOOR_SPEC = UfuncSpec(
    name="floor",
    chapel_name="floor",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.floor,
    precompute=floor_precompute,
)

ROUND_SPEC = UfuncSpec(
    name="round",
    chapel_name="round",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.round,
    extra_args_builder=round_args_builder,
    input_cast=cast_bool_to_float64_for_round,
    precompute=round_precompute,
)

TRUNC_SPEC = UfuncSpec(
    name="trunc",
    chapel_name="trunc",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.trunc,
    precompute=trunc_precompute,
)

SIGN_SPEC = UfuncSpec(
    name="sign",
    chapel_name="sgn",
    argname="pda",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.sign,
    validate=validate_sign,
    precompute=sign_precompute,
)

ISFINITE_SPEC = UfuncSpec(
    name="isfinite",
    chapel_name="isfinite",
    argname="pda",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.isfinite,
    precompute=isfinite_precompute,
    chapel_accepts_dtype=False,
)

ISINF_SPEC = UfuncSpec(
    name="isinf",
    chapel_name="isinf",
    argname="pda",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.isinf,
    precompute=isinf_precompute,
    chapel_accepts_dtype=False,
)

ISNAN_SPEC = UfuncSpec(
    name="isnan",
    chapel_name="isnan",
    argname="pda",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.isnan,
    precompute=isnan_precompute,
    chapel_accepts_dtype=False,
)

# log family (bool->float64 before processing)
LOG_SPEC = UfuncSpec(
    name="log",
    chapel_name="log",
    argname="pda",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.log,
    input_cast=cast_bool_to_float64_only,
)

LOG2_SPEC = UfuncSpec(
    name="log2",
    chapel_name="log2",
    argname="pda",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.log2,
    input_cast=cast_bool_to_float64_only,
)

LOG10_SPEC = UfuncSpec(
    name="log10",
    chapel_name="log10",
    argname="pda",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.log10,
    input_cast=cast_bool_to_float64_only,
)

LOG1P_SPEC = UfuncSpec(
    name="log1p",
    chapel_name="log1p",
    argname="pda",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.log1p,
    input_cast=cast_bool_to_float64_only,
)

# exp family (bool->float64 before processing)
EXP_SPEC = UfuncSpec(
    name="exp",
    chapel_name="exp",
    argname="pda",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.exp,
    input_cast=cast_bool_to_float64_only,
)

EXPM1_SPEC = UfuncSpec(
    name="expm1",
    chapel_name="expm1",
    argname="pda",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.expm1,
    input_cast=cast_bool_to_float64_only,
)

SQUARE_SPEC = UfuncSpec(
    name="square",
    chapel_name="square",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.square,
    precompute=square_precompute,
)

# trig and hyperbolic (bool->float64 before processing)
SIN_SPEC = UfuncSpec(
    name="sin",
    chapel_name="sin",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.sin,
    input_cast=cast_bool_to_float64_only,
)
COS_SPEC = UfuncSpec(
    name="cos",
    chapel_name="cos",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.cos,
    input_cast=cast_bool_to_float64_only,
)
TAN_SPEC = UfuncSpec(
    name="tan",
    chapel_name="tan",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.tan,
    input_cast=cast_bool_to_float64_only,
)

ARCSIN_SPEC = UfuncSpec(
    name="arcsin",
    chapel_name="arcsin",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.arcsin,
    input_cast=cast_bool_to_float64_only,
)
ARCCOS_SPEC = UfuncSpec(
    name="arccos",
    chapel_name="arccos",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.arccos,
    input_cast=cast_bool_to_float64_only,
)
ARCTAN_SPEC = UfuncSpec(
    name="arctan",
    chapel_name="arctan",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.arctan,
    input_cast=cast_bool_to_float64_only,
)

SINH_SPEC = UfuncSpec(
    name="sinh",
    chapel_name="sinh",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.sinh,
    input_cast=cast_bool_to_float64_only,
)
COSH_SPEC = UfuncSpec(
    name="cosh",
    chapel_name="cosh",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.cosh,
    input_cast=cast_bool_to_float64_only,
)
TANH_SPEC = UfuncSpec(
    name="tanh",
    chapel_name="tanh",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.tanh,
    input_cast=cast_bool_to_float64_only,
)

ARCSINH_SPEC = UfuncSpec(
    name="arcsinh",
    chapel_name="arcsinh",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.arcsinh,
    input_cast=cast_bool_to_float64_only,
)
ARCCOSH_SPEC = UfuncSpec(
    name="arccosh",
    chapel_name="arccosh",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.arccosh,
    input_cast=cast_bool_to_float64_only,
)
ARCTANH_SPEC = UfuncSpec(
    name="arctanh",
    chapel_name="arctanh",
    argname="x",
    output_dtype_resolver=resolve_output_dtype,
    scalar_op=np.arctanh,
    input_cast=cast_bool_to_float64_only,
)


# -----------------------------------------------------------------------------
# Public API wrappers
# -----------------------------------------------------------------------------


@typechecked
def abs(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise absolute value of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is broadcast over the input. At locations where the condition is True,
        abs will be applied to the corresponding values. Elsewhere, out will retain
        its original value. Default set to True.


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
    return ufunc_unary(ABS_SPEC, x, out=out, where=where)


@typechecked
def fabs(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Compute the absolute values element-wise, casting to a float beforehand.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is broadcast over the input. At locations where the condition is True,
        fabs will be applied to the corresponding values. Elsewhere, out will retain
        its original value. Default set to True.

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
    return ufunc_unary(FABS_SPEC, x, out=out, where=where)


@typechecked
def ceil(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise ceiling of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
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

    Notes
    -----
    Unlike numpy, arkouda requires out if where is used.
    """
    return ufunc_unary(CEIL_SPEC, x, out=out, where=where)


@typechecked
def floor(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise floor of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
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

    Notes
    -----
    Unlike numpy, arkouda requires out if where is used.
    """
    return ufunc_unary(FLOOR_SPEC, x, out=out, where=where)


@typechecked
def round(
    x: Union[pdarray, numeric_and_bool_scalars],
    decimals: int = 0,
    out: Optional[pdarray] = None,
):
    """
    Return the element-wise rounding of the array.

    Parameters
    ----------
    x : pdarray or numeric_and_bool_scalars
    decimals: Optional[Union[int, None]], default = None
        for float pdarrays, the number of decimal places of accuracy for the round.
        May be None, positive, negative, or zero.  If None, zero is used.
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.

    Returns
    -------
    pdarray
        A pdarray containing input array elements rounded to the nearest integer

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray or if the dtype of the pdarray
        is other than ak.float64, ak.int64, ak.uint64 or ak.bool.

    Notes
    -----
    This function follows numpy's rule of "round to even" when the fractional part
    of the number equals .5.  For example, 2.5 rounds to 2, but 3.5 rounds to 4.
    Arkouda's use of decimal is not perfect, as shown in the examples below.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.round(ak.array([1.1, 2.5, 3.14159]))
    array([1.00000000000000000 2.00000000000000000 3.00000000000000000])
    >>> ak.round(ak.array([1.5, 2.5, 3.5]))
    array([2.00000000000000000 2.00000000000000000 4.00000000000000000])
    >>> ak.round(ak.array([-143.1, 279.8]),decimals=-1)
    array([-140.00000000000000000 280.00000000000000000])
    >>> ak.round(ak.array([-143.1, 279.8]),decimals=0)
    array([-143.00000000000000000 280.00000000000000000])
    >>> ak.round(ak.array([1.541, 2.732]),decimals=2)
    array([1.54 2.73])
    >>> ak.round(ak.array([1.541, 2.732]),decimals=3)
    array([1.5409999999999999 2.7320000000000002])
    """
    return ufunc_unary(ROUND_SPEC, x, out=out, where=True, decimals=decimals)


@typechecked
def trunc(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise truncation of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
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

    Notes
    -----
    Unlike numpy, arkouda requires out if where is used.
    """
    return ufunc_unary(TRUNC_SPEC, x, out=out, where=where)


@typechecked
def sign(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise sign of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing sign values of the input array elements
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.sign(ak.array([-10, -5, 0, 5, 10]))
    array([-1 -1 0 1 1])
    """
    return ufunc_unary(SIGN_SPEC, x, out=out, where=where)


@typechecked
def isfinite(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise isfinite check applied to the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    return ufunc_unary(ISFINITE_SPEC, x, out=out, where=where)


@typechecked
def isinf(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise isinf check applied to the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    return ufunc_unary(ISINF_SPEC, x, out=out, where=where)


@typechecked
def isnan(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise isnan check applied to the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
        if the underlying pdarray is not one of float, int, uint or bool

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.isnan(ak.array([1.0, 2.0, np.log(-1)]))
    array([False False True])
    """
    return ufunc_unary(ISNAN_SPEC, x, out=out, where=where)


@typechecked
def log(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise natural log of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing natural log values of the input array elements

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
    """
    return ufunc_unary(LOG_SPEC, x, out=out, where=where)


@typechecked
def log2(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise natural log of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing log base 2 values of the input array elements

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

    Log base 2
    >>> ak.log(A) / np.log(2)
    array([0.00000000... 3.32192809... 6.64385618...])
    """
    return ufunc_unary(LOG2_SPEC, x, out=out, where=where)


@typechecked
def log10(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise natural log of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing log base 10 values of the input array elements

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

    Log base 10
    >>> ak.log(A) / np.log(10)
    array([0.00000000... 1.00000000... 2.00000000...])
    """
    return ufunc_unary(LOG10_SPEC, x, out=out, where=where)


@typechecked
def log1p(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise natural log of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing natural log values of 1 plus the input array elements

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
    >>> ak.log1p(A)
    array([0.69314718055994529 2.3978952727983707 4.6151205168412597])

    """
    return ufunc_unary(LOG1P_SPEC, x, out=out, where=where)


@typechecked
def exp(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise exponential of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing exponential values of the input array elements

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
    return ufunc_unary(EXP_SPEC, x, out=out, where=where)


@typechecked
def expm1(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise exponential of the array minus one.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing e raised to each of the inputs, then subtracting one.

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
    return ufunc_unary(EXPM1_SPEC, x, out=out, where=where)


@typechecked
def square(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise square of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing square values of the input array elements

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
    return ufunc_unary(SQUARE_SPEC, x, out=out, where=where)


@typechecked
def sin(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise sine of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    return ufunc_unary(SIN_SPEC, x, out=out, where=where)


@typechecked
def cos(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise cosine of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing cos for each element
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
    return ufunc_unary(COS_SPEC, x, out=out, where=where)


@typechecked
def tan(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise tangent of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

    Returns
    -------
    pdarray
        A pdarray containing tan for each element
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
    return ufunc_unary(TAN_SPEC, x, out=out, where=where)


@typechecked
def arcsin(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise inverse sine of the array. The result is between -pi/2 and pi/2.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    return ufunc_unary(ARCSIN_SPEC, x, out=out, where=where)


@typechecked
def arccos(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise inverse cosine of the array. The result is between 0 and pi.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    return ufunc_unary(ARCCOS_SPEC, x, out=out, where=where)


@typechecked
def arctan(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise inverse tangent of the array. The result is between -pi/2 and pi/2.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    return ufunc_unary(ARCTAN_SPEC, x, out=out, where=where)


@typechecked
def sinh(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise hyperbolic sine of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    return ufunc_unary(SINH_SPEC, x, out=out, where=where)


@typechecked
def cosh(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise hyperbolic cosine of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    return ufunc_unary(COSH_SPEC, x, out=out, where=where)


@typechecked
def tanh(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise hyperbolic tangent of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    return ufunc_unary(TANH_SPEC, x, out=out, where=where)


@typechecked
def arcsinh(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise inverse hyperbolic sine of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    return ufunc_unary(ARCSINH_SPEC, x, out=out, where=where)


@typechecked
def arccosh(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise inverse hyperbolic cosine of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    return ufunc_unary(ARCCOSH_SPEC, x, out=out, where=where)


@typechecked
def arctanh(
    x: Union[pdarray, numeric_and_bool_scalars],
    out: Optional[pdarray] = None,
    *,
    where: Optional[Any] = None,
):
    """
    Return the element-wise inverse hyperbolic tangent of the array.

    Parameters
    ----------
    x : pdarray
    out: None or pdarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to.
    where : bool or pdarray, default=True
        This condition is applied over the input. At locations where the condition is True, the
        corresponding value will be acted on by the function. Elsewhere, it will retain its
        original value. Default set to True.

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
    return ufunc_unary(ARCTANH_SPEC, x, out=out, where=where)
