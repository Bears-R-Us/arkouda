import json
from enum import Enum
from typing import ForwardRef, List, Optional, Tuple, Union
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
from arkouda.pdarraycreation import array
from arkouda.strings import Strings

Categorical = ForwardRef("Categorical")
SegArray = ForwardRef("SegArray")

__all__ = [
    "cast",
    "abs",
    "log",
    "exp",
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
    "value_counts",
    "isnan",
    "ErrorMode",
]


class ErrorMode(Enum):
    strict = "strict"
    ignore = "ignore"
    return_validity = "return_validity"


@typechecked
def cast(
    pda: Union[pdarray, Strings],
    dt: Union[np.dtype, type, str, BigInt],
    errors: ErrorMode = ErrorMode.strict,
) -> Union[Union[pdarray, Strings], Tuple[pdarray, pdarray]]:
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

    if isinstance(pda, pdarray):
        name = pda.name
    elif isinstance(pda, Strings):
        name = pda.entry.name
    # typechecked decorator guarantees no other case

    dt = _as_dtype(dt)
    cmd = "cast"
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
        cmd="efunc",
        args={
            "func": "abs",
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
        cmd="efunc",
        args={
            "func": "log",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def log10(x: pdarray) -> pdarray:
    """
    compute the log of a pdarray and perform a basechange

    Parameters
    __________
    x : pdarray
        array to compute on

    Returns
    _______
    pdarray contain values of the base 10 log
    """
    basechange = float(np.log10(np.exp(1)))
    return basechange * log(x)


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
        cmd="efunc",
        args={
            "func": "exp",
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
        cmd="efunc",
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
        cmd="efunc",
        args={
            "func": "cumprod",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def sin(pda: pdarray) -> pdarray:
    """
    Return the element-wise sine of the array.

    Parameters
    ----------
    pda : pdarray

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
    repMsg = generic_msg(
        cmd="efunc",
        args={
            "func": "sin",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def cos(pda: pdarray) -> pdarray:
    """
    Return the element-wise cosine of the array.

    Parameters
    ----------
    pda : pdarray

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
    repMsg = type_cast(
        str,
        generic_msg(
            cmd="efunc",
            args={
                "func": "cos",
                "array": pda,
            },
        ),
    )
    return create_pdarray(repMsg)


@typechecked
def tan(pda: pdarray) -> pdarray:
    """
    Return the element-wise tangent of the array.

    Parameters
    ----------
    pda : pdarray

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
    repMsg = generic_msg(
        cmd="efunc",
        args={
            "func": "tan",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def arcsin(pda: pdarray) -> pdarray:
    """
    Return the element-wise inverse sine of the array. The result is between -pi/2 and pi/2.

    Parameters
    ----------
    pda : pdarray

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
    repMsg = generic_msg(
        cmd="efunc",
        args={
            "func": "arcsin",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def arccos(pda: pdarray) -> pdarray:
    """
    Return the element-wise inverse cosine of the array. The result is between 0 and pi.

    Parameters
    ----------
    pda : pdarray

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
    repMsg = generic_msg(
        cmd="efunc",
        args={
            "func": "arccos",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def arctan(pda: pdarray) -> pdarray:
    """
    Return the element-wise inverse tangent of the array. The result is between -pi/2 and pi/2.

    Parameters
    ----------
    pda : pdarray

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
    repMsg = generic_msg(
        cmd="efunc",
        args={
            "func": "arctan",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def arctan2(num: Union[pdarray, numeric_scalars], denom: Union[pdarray, numeric_scalars]) -> pdarray:
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
    return create_pdarray(
        type_cast(
            str,
            generic_msg(
                cmd="efunc2",
                args={
                    "func": "arctan2",
                    "A": num,
                    "B": denom,
                },
            ),
        )
    )


@typechecked
def sinh(pda: pdarray) -> pdarray:
    """
    Return the element-wise hyperbolic sine of the array.

    Parameters
    ----------
    pda : pdarray

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
    repMsg = generic_msg(
        cmd="efunc",
        args={
            "func": "sinh",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def cosh(pda: pdarray) -> pdarray:
    """
    Return the element-wise hyperbolic cosine of the array.

    Parameters
    ----------
    pda : pdarray

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
    repMsg = generic_msg(
        cmd="efunc",
        args={
            "func": "cosh",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def tanh(pda: pdarray) -> pdarray:
    """
    Return the element-wise hyperbolic tangent of the array.

    Parameters
    ----------
    pda : pdarray

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
    repMsg = generic_msg(
        cmd="efunc",
        args={
            "func": "tanh",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def arcsinh(pda: pdarray) -> pdarray:
    """
    Return the element-wise inverse hyperbolic sine of the array.

    Parameters
    ----------
    pda : pdarray

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
    repMsg = generic_msg(
        cmd="efunc",
        args={
            "func": "arcsinh",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def arccosh(pda: pdarray) -> pdarray:
    """
    Return the element-wise inverse hyperbolic cosine of the array.

    Parameters
    ----------
    pda : pdarray

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
    repMsg = generic_msg(
        cmd="efunc",
        args={
            "func": "arccosh",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def arctanh(pda: pdarray) -> pdarray:
    """
    Return the element-wise inverse hyperbolic tangent of the array.

    Parameters
    ----------
    pda : pdarray

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
    repMsg = generic_msg(
        cmd="efunc",
        args={
            "func": "arctanh",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))


@typechecked
def rad2deg(pda: pdarray) -> pdarray:
    """
    Converts angles element-wise from radians to degrees.

    Parameters
    ----------
    pda : pdarray

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
    return 180 * (pda / np.pi)


@typechecked
def deg2rad(pda: pdarray) -> pdarray:
    """
    Converts angles element-wise from degrees to radians.

    Parameters
    ----------
    pda : pdarray

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
    return np.pi * pda / 180


def _hash_helper(a):
    from arkouda import Categorical as Categorical_
    from arkouda import SegArray as SegArray_

    if isinstance(a, SegArray_):
        return json.dumps(
            {"segments": a.segments.name, "values": a.values.name, "valObjType": a.values.objType}
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
    if pda.dtype == bigint:
        return hash(pda.bigint_to_uint_arrays())
    repMsg = type_cast(
        str,
        generic_msg(
            cmd="efunc",
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
        repMsg = generic_msg(
            cmd="efunc3vv",
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
            cmd="efunc3vs",
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
            cmd="efunc3sv",
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
            raise TypeError(f"Cannot cast between scalars {str(A)} and {str(B)} to supported dtype")
        repMsg = generic_msg(
            cmd="efunc3ss",
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
def histogram(pda: pdarray, bins: int_scalars = 10) -> Tuple[np.ndarray, pdarray]:
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
    (np.ndarray, Union[pdarray, int64 or float64])
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
    value_counts

    Notes
    -----
    The bins are evenly spaced in the interval [pda.min(), pda.max()].

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> A = ak.arange(0, 10, 1)
    >>> nbins = 3
    >>> b, h = ak.histogram(A, bins=nbins)
    >>> h
    array([3, 3, 4])
    >>> b
    array([0., 3., 6.])

    # To plot, use only the left edges (now returned), and export the histogram to NumPy
    >>> plt.plot(b, h.to_ndarray())
    """
    if bins < 1:
        raise ValueError("bins must be 1 or greater")
    b = np.linspace(pda.min(), pda.max(), bins + 1)[:-1]  # type: ignore
    repMsg = generic_msg(cmd="histogram", args={"array": pda, "bins": bins})
    return b, create_pdarray(type_cast(str, repMsg))


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
def isnan(pda: pdarray) -> pdarray:
    """
    Test a pdarray for Not a number / NaN values
    Currently only supports float-value-based arrays

    Parameters
    ----------
    pda : pdarray to test

    Returns
    -------
    pdarray consisting of True / False values; True where NaN, False otherwise

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    RuntimeError
        if the underlying pdarray is not float-based
    """
    rep_msg = generic_msg(
        cmd="efunc",
        args={
            "func": "isnan",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, rep_msg))
