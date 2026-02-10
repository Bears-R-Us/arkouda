from __future__ import annotations

import builtins
import sys
import warnings

from enum import Enum
from typing import (  # noqa: F401
    Literal,
    TypeAlias,
    TypeGuard,
    Union,
    cast,
)

import numpy as np

from numpy import (
    bool,
    bool_,
    complex64,
    complex128,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    str_,
    uint8,
    uint16,
    uint32,
    uint64,
)
from numpy.dtypes import (
    BoolDType,
    ByteDType,
    BytesDType,
    CLongDoubleDType,
    Complex64DType,
    Complex128DType,
    DateTime64DType,
    Float16DType,
    Float32DType,
    Float64DType,
    Int8DType,
    Int16DType,
    Int32DType,
    Int64DType,
    IntDType,
    LongDoubleDType,
    LongDType,
    LongLongDType,
    ObjectDType,
    ShortDType,
    StrDType,
    TimeDelta64DType,
    UByteDType,
    UInt8DType,
    UInt16DType,
    UInt32DType,
    UInt64DType,
    UIntDType,
    ULongDType,
    ULongLongDType,
    UShortDType,
    VoidDType,
)

from ._bigint import bigint, bigint_


__all__ = [
    "_datatype_check",
    "ARKOUDA_SUPPORTED_DTYPES",
    "ARKOUDA_SUPPORTED_INTS",
    "DType",
    "DTypeObjects",
    "DTypes",
    "NUMBER_FORMAT_STRINGS",
    "NumericDTypes",
    "ScalarDTypes",
    "SeriesDTypes",
    "_is_dtype_in_union",
    "_val_isinstance_of_union",
    "all_scalars",
    "bigint",
    "bigint_",
    "bitType",
    "bool",
    "bool_scalars",
    "can_cast",
    "complex128",
    "complex64",
    "dtype",
    "dtype_for_chapel",
    "float16",
    "float32",
    "float64",
    "float_scalars",
    "get_byteorder",
    "get_server_byteorder",
    "int16",
    "int32",
    "int64",
    "int8",
    "intTypes",
    "int_scalars",
    "is_supported_bool",
    "is_supported_dtype",
    "is_supported_float",
    "is_supported_int",
    "is_supported_number",
    "numeric_and_bool_scalars",
    "numeric_scalars",
    "numpy_scalars",
    "resolve_scalar_dtype",
    "result_type",
    "str_",
    "str_scalars",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "BoolDType",
    "ByteDType",
    "BytesDType",
    "CLongDoubleDType",
    "Complex64DType",
    "Complex128DType",
    "DateTime64DType",
    "Float16DType",
    "Float32DType",
    "Float64DType",
    "Int8DType",
    "Int16DType",
    "Int32DType",
    "Int64DType",
    "IntDType",
    "LongDoubleDType",
    "LongDType",
    "LongLongDType",
    "ObjectDType",
    "ShortDType",
    "StrDType",
    "TimeDelta64DType",
    "UByteDType",
    "UInt8DType",
    "UInt16DType",
    "UInt32DType",
    "UInt64DType",
    "UIntDType",
    "ULongDType",
    "ULongLongDType",
    "UShortDType",
    "VoidDType",
    "isSupportedDType",
    "isSupportedInt",
    "isSupportedFloat",
    "isSupportedBool",
    "isSupportedNumber",
]


NUMBER_FORMAT_STRINGS = {
    "bool": "{}",
    "int64": "{:d}",
    "float64": "{:.17f}",
    "uint8": "{:d}",
    "np.float64": "{f}",
    "uint64": "{:d}",
    "bigint": "{:d}",
}

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_UINT64_MAX = (1 << 64) - 1


def _datatype_check(the_dtype, allowed_list, name):
    if the_dtype not in allowed_list:
        raise TypeError(f"{name} only implements types {allowed_list}")


def dtype(x):
    """
    Normalize a dtype-like input into an Arkouda dtype sentinel or a NumPy dtype.

    This function accepts many dtype-like forms—including Python scalars,
    NumPy scalar types, Arkouda ``bigint`` sentinels, and strings—and resolves
    them to the canonical Arkouda/NumPy dtype object. The resolution rules
    include special handling of the ``bigint`` family and magnitude-aware routing
    for Python integers.

    Arguments
    ---------
    x : Any
        The dtype-like object to normalize. May be a Python scalar, a NumPy
        dtype or scalar, the ``bigint`` sentinel or scalar, a dtype-specifying
        string, or any object accepted by ``numpy.dtype``.

    Raises
    ------
    TypeError
        If ``x`` cannot be interpreted as either an Arkouda dtype or a NumPy
        dtype. This includes cases where ``numpy.dtype(x)`` itself fails.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.numpy import dtype

    # bigint family
    >>> dtype("bigint")
    dtype(bigint)
    >>> dtype(ak.bigint(10_000_000_000_000_000_000))
    dtype(bigint)

    # magnitude-based routing for Python ints
    >>> dtype(10)
    dtype('int64')
    >>> dtype(2**63 - 1)
    dtype('int64')
    >>> dtype(2**63)
    dtype('uint64')
    >>> dtype(2**100)
    dtype(bigint)

    # floats and bools
    >>> dtype(1.0)
    dtype('float64')
    >>> dtype(True)
    dtype('bool')

    # string dtypes
    >>> dtype("str")
    dtype('<U')

    # fallback to numpy.dtype
    >>> dtype("int32")
    dtype('int32')
    """
    import builtins

    import numpy as np

    # ---- Arkouda bigint family (catch these FIRST) ----
    if (
        (isinstance(x, str) and x.lower() == "bigint")
        or isinstance(x, bigint)  # sentinel instance
        or x is bigint  # sentinel class object
        or getattr(x, "name", "").lower() == "bigint"
        or (isinstance(x, type) and x.__name__ == "bigint")  # class by name
        or (bigint_ is not None and isinstance(x, bigint_))  # scalar instance
        or (isinstance(x, type) and x.__name__ == "bigint_")  # scalar class object
    ):
        return bigint()

    # ---- String dtype spellings ----
    if isinstance(x, str) and x.lower() in {"str", "str_", "strings", "string"}:
        return np.dtype(np.str_)
    if x in (str, np.str_):
        return np.dtype(np.str_)

    # ---- Core Python scalar types ----
    if x is float:
        return np.dtype(np.float64)
    if isinstance(x, (bool, builtins.bool, np.bool_)):
        return np.dtype(np.bool_)

    # Normalize NumPy integer scalars to Python int so they reuse the same path
    if isinstance(x, np.integer):
        x = int(x)

    # Magnitude-aware routing for Python ints
    if isinstance(x, int):
        if x < 0:
            # negative: fits in int64?
            return bigint() if x < _INT64_MIN else np.dtype(np.int64)
        else:
            # non-negative: prefer int64 up to max, then uint64 window, else bigint
            if x <= _INT64_MAX:
                return np.dtype(np.int64)
            if x <= _UINT64_MAX:
                return np.dtype(np.uint64)
            return bigint()

    if isinstance(x, float):
        return np.dtype(np.float64)
    if isinstance(x, bool):
        return np.dtype(np.bool_)

    # ---- Fallback to NumPy dtype for everything else ----
    try:
        return np.dtype(x)
    except TypeError as e:
        # Re-raise with a clearer message including the repr of x
        raise TypeError(f"Unsupported dtype-like object for arkouda.numpy.dtype: {x!r}") from e


_dtype_for_chapel = dict()  # type: ignore


_dtype_name_for_chapel = {  # see DType
    "real": "float64",
    "real(32)": "float32",
    "real(64)": "float64",
    "complex": "complex128",
    "complex(64)": "complex64",
    "complex(128)": "complex128",
    "int": "int64",
    "int(8)": "int8",
    "int(16)": "int16",
    "int(32)": "int32",
    "int(64)": "int64",
    "uint": "uint64",
    "uint(8)": "uint8",
    "uint(16)": "uint16",
    "uint(32)": "uint32",
    "uint(64)": "uint64",
    "bool": "bool",
    "bigint": "bigint",
    "string": "str",
}


def dtype_for_chapel(type_name: str):
    """
    Returns dtype() for the given Chapel type.

    Parameters
    ----------
    type_name : str
        The name of the Chapel type, with or without the bit width

    Returns
    -------
    dtype
        The corresponding Arkouda dtype object

    Raises
    ------
    TypeError
        Raised if Arkouda does not have a type that corresponds to `type_name`

    """
    try:
        return _dtype_for_chapel[type_name]
    except KeyError:
        try:
            dtype_name = _dtype_name_for_chapel[type_name]
        except KeyError:
            raise TypeError(f"Arkouda does not have a dtype that corresponds to '{type_name}' in Chapel")
        result = dtype(dtype_name)
        _dtype_for_chapel[type_name] = result
        return result


def _is_bigint_like(x) -> builtins.bool:
    if x is bigint or isinstance(x, bigint):
        return True
    if getattr(x, "name", "").lower() == "bigint":
        return True
    if isinstance(x, str) and x.lower() == "bigint":
        return True
    _bigint_scalar = globals().get("bigint_")
    if _bigint_scalar is not None and isinstance(x, _bigint_scalar):
        return True
    if isinstance(x, type) and x.__name__ in ("bigint", "bigint_"):
        return True
    return False


def can_cast(from_dt, to_dt, casting: Literal["safe",] | None = "safe") -> builtins.bool:
    """
    Determine whether a value of one dtype can be safely cast to another,
    following NumPy-like rules but including Arkouda-specific handling for
    ``bigint`` and ``bigint_``.

    The default ``"safe"`` mode uses the following logic:

    * ``bigint`` → ``bigint``: always allowed.
    * ``bigint`` → float dtypes: allowed (may lose precision, but magnitude fits).
    * ``bigint`` → fixed-width signed/unsigned integers: not allowed, due to
      potential overflow.
    * int64 / uint64 → ``bigint``: allowed (widening).
    * float → ``bigint``: not allowed (information loss).
    * All other cases fall back to ``numpy.can_cast`` semantics.

    Arguments
    ---------
    from_dt : Any
        Source dtype or scalar-like object.
    to_dt : Any
        Target dtype or scalar-like object.
    casting : str, optional
        Casting rule, matching NumPy’s ``can_cast`` API. Only ``"safe"``
        is currently implemented. Other values are accepted for API
        compatibility but routed through the same logic.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.numpy import can_cast, dtype

    # bigint → bigint
    >>> can_cast(dtype("bigint"), dtype("bigint"))
    True

    # bigint → float64
    >>> can_cast(dtype("bigint"), dtype("float64"))
    True

    # bigint → int64 (unsafe)
    >>> can_cast(dtype("bigint"), dtype("int64"))
    False

    # int64 → bigint (widening)
    >>> can_cast(dtype("int64"), dtype("bigint"))
    True

    # float → bigint (lossy)
    >>> can_cast(dtype("float64"), dtype("bigint"))
    False

    # Standard NumPy cases
    >>> can_cast(dtype("int64"), dtype("float64"))
    True
    >>> can_cast(dtype("float64"), dtype("int64"))
    False
    """
    import numpy as np

    def _to_np_dtype(x):
        """Normalize a dtype-ish into np.dtype, but NEVER feed bigint to NumPy."""
        import numpy as np

        # 1) already a NumPy dtype
        if isinstance(x, np.dtype):
            return x

        # 2) type objects (np.float64, int, bool, etc.)
        if isinstance(x, type):
            return np.dtype(x)

        # 3) plain Python / NumPy scalars
        if isinstance(x, (int, float, bool, np.number)):
            # Let NumPy infer from the value; this keeps us away from np.dtype(0) etc.
            return np.asarray(x).dtype

        # 4) instances with a dtype attribute (arrays, pdarray, numpy scalars)
        if hasattr(x, "dtype") and not isinstance(x, type):
            dt = getattr(x, "dtype")
            if isinstance(dt, np.dtype):
                return dt
            # If .dtype is bigint-like, proxy as object
            if getattr(dt, "name", "").lower() == "bigint" or (dt is bigint) or isinstance(dt, bigint):
                return np.dtype("O")
            return np.dtype(dt)

        # 5) last resort
        return np.dtype(x)

    def _scalar_int_can_cast_safe(value: int, np_to) -> builtins.bool | None:
        """
        Emulate NumPy 1.x `can_cast(value, to, casting="safe")` for Python ints.

        (NumPy 2.x does not allow casting for Pyton ints.)

        Returns True/False if handled (int → integer dtypes), otherwise None.
        """
        import numpy as np

        if not np.issubdtype(np_to, np.integer):
            return None  # let dtype-based logic handle non-integer targets

        # Unsigned targets
        if np.issubdtype(np_to, np.unsignedinteger):
            info = np.iinfo(np_to)
            return 0 <= value <= info.max

        # Signed targets
        info = np.iinfo(np_to)
        return info.min <= value <= info.max

    from_is_big = _is_bigint_like(from_dt)
    to_is_big = _is_bigint_like(to_dt)

    # bigint→bigint
    if from_is_big and to_is_big:
        return True

    # bigint→non-bigint
    if from_is_big:
        np_to = _to_np_dtype(to_dt)
        if np.issubdtype(np_to, np.floating):
            return True
        if np_to.kind in ("i", "u"):
            return False
        if np_to.kind == "O":
            return True
        return False

    # non-bigint→bigint
    if to_is_big:
        try:
            np_from = _to_np_dtype(from_dt)
        except TypeError:
            np_from = None
        if isinstance(np_from, np.dtype):
            if np.issubdtype(np_from, np.integer):
                return True
            if np.issubdtype(np_from, np.floating):
                return False
        return False

    # Neither side is bigint → NumPy-like semantics

    np_to = _to_np_dtype(to_dt)

    # ① Python int scalar special-case: emulate old scalar rules for "safe"
    if casting == "safe" and isinstance(from_dt, int):
        scalar_result = _scalar_int_can_cast_safe(from_dt, np_to)
        if scalar_result is not None:
            return builtins.bool(scalar_result)

    # ② Fallback: pure dtype-based NEP 50-style semantics
    np_from = _to_np_dtype(from_dt)
    return builtins.bool(np.can_cast(np_from, np_to, casting=casting))


def result_type(*args):
    """
    Determine the result dtype from one or more inputs, following NumPy’s
    promotion rules but extended to support Arkouda ``bigint`` semantics.

    This function mirrors ``numpy.result_type`` for standard NumPy dtypes,
    scalars, and arrays, but additionally recognizes Arkouda ``bigint`` and
    ``bigint_`` values, promoting them according to Arkouda-specific rules.

    In mixed-type expressions, the following logic is applied:

    * Any presence of ``bigint`` or ``bigint_`` promotes the result to:
        - ``float64`` if any float is also present,
        - otherwise ``bigint``.
    * Python integers first pass through Arkouda's magnitude-aware ``dtype()``
      routing, so extremely large integers may promote to ``bigint``.
    * Booleans promote to ``bool`` as in NumPy.
    * Mixed signed/unsigned integers follow NumPy rules, except that a
      non-negative signed scalar combined with unsigned scalars promotes to
      the widest unsigned dtype.
    * All remaining cases defer to ``numpy.result_type``.

    Arguments
    ---------
    *args : Any
        One or more dtype-like objects, scalars, NumPy arrays, Arkouda arrays,
        or any value accepted by ``numpy.result_type`` or Arkouda’s
        ``dtype()`` conversion.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.numpy import result_type, dtype

    # bigint wins unless floats appear
    >>> result_type(dtype("bigint"), dtype("int64"))
    dtype(bigint)

    >>> result_type(dtype("bigint"), dtype("float64"))
    dtype('float64')

    # magnitude-aware routing: this becomes bigint, so result is bigint
    >>> result_type(2**100, 5)
    dtype(bigint)

    # standard NumPy integer promotions
    >>> result_type(dtype("int32"), dtype("int64"))
    dtype('int64')

    # unsigned with non-negative signed scalar → largest unsigned
    >>> result_type(np.uint32(3), 7)   # 7 is non-negative signed scalar
    dtype('uint32')

    # float promotion
    >>> result_type(1.0, 5)
    dtype('float64')

    # boolean stays boolean
    >>> result_type(True, False)
    dtype('bool')
    """
    import numpy as np

    has_bigint = False
    has_float = False
    np_args: list[np.dtype] = []

    saw_unsigned = False
    signed_from_nonneg_scalar = False
    all_integer = True

    for a in args:
        # 0) bigint-like sentinel/scalar/class
        if _is_bigint_like(a):
            has_bigint = True
            continue

        # 1) explicit NumPy dtype
        if isinstance(a, np.dtype):
            np_dt = a

        # 2) Python / NumPy type objects
        elif isinstance(a, type):
            np_dt = np.dtype(a)

        # 3) objects with a real .dtype (pdarray, numpy scalars/arrays, etc.)
        elif hasattr(a, "dtype"):
            dt = getattr(a, "dtype")
            if _is_bigint_like(dt):
                has_bigint = True
                continue
            np_dt = np.dtype(dt)

        # 4) plain scalars  —— BOOL BEFORE INT ——
        elif isinstance(a, (bool, np.bool_)):
            np_dt = np.dtype(np.bool_)
        elif isinstance(a, (int, np.integer)):
            ak_dt = dtype(a)  # magnitude-aware routing
            if _is_bigint_like(ak_dt):
                has_bigint = True
                continue
            np_dt = np.dtype(ak_dt)
            if np_dt.kind == "i" and int(a) >= 0:
                signed_from_nonneg_scalar = True
        elif isinstance(a, (float, np.floating)):
            np_dt = np.result_type(a)

        # 5) generic fallback
        else:
            np_dt = np.result_type(a)

        np_dt = np.dtype(np_dt)
        np_args.append(np_dt)

        if not np.issubdtype(np_dt, np.integer):
            all_integer = False
        if np_dt.kind == "u":
            saw_unsigned = True
        if np.issubdtype(np_dt, np.floating):
            has_float = True

    if has_bigint:
        return np.dtype(np.float64) if has_float else bigint()

    if all_integer and saw_unsigned and signed_from_nonneg_scalar:
        unsigneds = [dt for dt in np_args if dt.kind == "u"]
        return max(unsigneds, key=lambda d: d.itemsize)

    return np.result_type(*np_args)


def _is_dtype_in_union(dtype, union_type) -> builtins.bool:
    """
    Check if a given type is in a typing.Union.

    Args
    ----
        dtype (type): The type to check for.
        union_type (type): The typing.Union type to check against.

    Returns
    -------
        bool True if the dtype is in the union_type, False otherwise.
    """
    return hasattr(union_type, "__args__") and dtype in union_type.__args__


def _val_isinstance_of_union(val, union_type) -> builtins.bool:
    """
    Check if a given val is an instance of one of the types in the typing.Union.

    Args
    ----
        val: The val to do the isinstance check on.
        union_type (type): The typing.Union type to check against.

    Returns
    -------
        bool: True if the val is an instance of one
            of the types in the union_type, False otherwise.
    """
    return hasattr(union_type, "__args__") and isinstance(val, union_type.__args__)


intTypes = frozenset((dtype("int64"), dtype("uint64"), dtype("uint8")))
bitType = uint64

# Union aliases used for static and runtime type checking
bool_scalars = Union[builtins.bool, np.bool_]  # type: TypeAlias

float_scalars = Union[float, np.float64, np.float32]  # type: TypeAlias


int_scalars = Union[
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]  # type: TypeAlias


numeric_scalars = Union[float_scalars, int_scalars]  # type: TypeAlias


numeric_and_bool_scalars = Union[bool_scalars, numeric_scalars]  # type: TypeAlias


numpy_scalars = Union[
    np.float64,
    np.float32,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.bool_,
    np.str_,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]  # type: TypeAlias

str_scalars = Union[str, np.str_]  # type: TypeAlias

all_scalars = Union[bool_scalars, numeric_scalars, numpy_scalars, str_scalars]  # type: TypeAlias


"""
The DType enum defines the supported Arkouda data types in string form.
"""


class DType(Enum):
    FLOAT = "float"
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"
    INT = "int"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT = "uint"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    BOOL = "bool"
    BIGINT = "bigint"
    STR = "str"

    def __str__(self) -> str:
        """
        Overridden method returns value, which is useful in outputting
        a DType as a request parameter.
        """
        return self.value

    def __repr__(self) -> str:
        """
        Overridden method returns value, which is useful in outputting
        a DType as a request parameter.
        """
        return self.value


ARKOUDA_SUPPORTED_BOOLS = (builtins.bool, np.bool_)


ARKOUDA_SUPPORTED_INTS = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    bigint,
    bigint_,
)

ARKOUDA_SUPPORTED_FLOATS = (float, np.float64, np.float32)
ARKOUDA_SUPPORTED_NUMBERS = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    float,
    np.float32,
    np.float64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    bigint,
    bigint_,
)

# TODO: bring supported data types into parity with all numpy dtypes
# missing full support for: float32, int32, int16, int8, uint32, uint16, complex64, complex128
# ARKOUDA_SUPPORTED_DTYPES = frozenset([member.value for _, member in DType.__members__.items()])
ARKOUDA_SUPPORTED_DTYPES = (
    bool_,
    float,
    float64,
    int,
    int64,
    uint64,
    uint8,
    bigint,
    str,
)

DTypes = frozenset([member.value for _, member in DType.__members__.items()])
DTypeObjects = frozenset([bool_, float, float64, int, int64, str, str_, uint8, uint64])
NumericDTypes = frozenset(["bool_", "bool", "float", "float64", "int", "int64", "uint64", "bigint"])
SeriesDTypes = {
    "string": np.str_,
    "<class 'str'>": np.str_,
    "int64": np.int64,
    "uint64": np.uint64,
    "<class 'numpy.int64'>": np.int64,
    "float64": np.float64,
    "<class 'numpy.float64'>": np.float64,
    "bool": np.bool_,
    "<class 'bool'>": np.bool_,
    "datetime64[ns]": np.int64,
    "timedelta64[ns]": np.int64,
}
ScalarDTypes = frozenset(["bool_", "float64", "int64"])


def is_supported_int(num) -> TypeGuard[int_scalars]:
    """
    Whether a scalar is an arkouda supported integer dtype.

    Parameters
    ----------
    num: object
        A scalar.

    Returns
    -------
    bool
        True if scalar is an instance of an arkouda supported integer dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.is_supported_int(79)
    True
    >>> ak.is_supported_int(54.9)
    False

    """
    return isinstance(num, ARKOUDA_SUPPORTED_INTS)


def isSupportedInt(num):
    """
    Deprecated alias for :func:`is_supported_int`.

    This function exists for backward compatibility only. Use
    :func:`is_supported_int` instead.

    Parameters
    ----------
    num : object
        A scalar value to test.

    Returns
    -------
    bool
        True if ``num`` is an arkouda-supported integer dtype,
        otherwise False.

    See Also
    --------
    is_supported_int : Preferred replacement.
    """
    warnings.warn(
        "isSupportedInt is deprecated; use is_supported_int instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_supported_int(num)


def is_supported_float(num) -> TypeGuard[float_scalars]:
    """
    Whether a scalar is an arkouda supported float dtype.

    Parameters
    ----------
    num: object
        A scalar.

    Returns
    -------
    bool
        True if scalar is an instance of an arkouda supported float dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.is_supported_float(56)
    False
    >>> ak.is_supported_float(56.7)
    True

    """
    return isinstance(num, ARKOUDA_SUPPORTED_FLOATS)


def isSupportedFloat(num):
    """
    Deprecated alias for :func:`is_supported_float`.

    This function exists for backward compatibility only. Use
    :func:`is_supported_float` instead.

    Parameters
    ----------
    num : object
        A scalar value to test.

    Returns
    -------
    bool
        True if ``num`` is an arkouda-supported float dtype,
        otherwise False.

    See Also
    --------
    is_supported_float : Preferred replacement.
    """
    warnings.warn(
        "isSupportedFloat is deprecated; use is_supported_float instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_supported_float(num)


def is_supported_number(num) -> TypeGuard[numeric_scalars]:
    """
    Whether a scalar is an arkouda supported numeric dtype.

    Parameters
    ----------
    num: object
        A scalar.

    Returns
    -------
    bool
        True if scalar is an instance of an arkouda supported numeric dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.is_supported_number(45.9)
    True
    >>> ak.is_supported_number("string")
    False

    """
    return isinstance(num, ARKOUDA_SUPPORTED_NUMBERS)


def isSupportedNumber(num):
    """
    Deprecated alias for :func:`is_supported_number`.

    This function exists for backward compatibility only. Use
    :func:`is_supported_number` instead.

    Parameters
    ----------
    num : object
        A scalar value to test.

    Returns
    -------
    bool
        True if ``num`` is an arkouda-supported numeric dtype,
        otherwise False.

    See Also
    --------
    is_supported_number : Preferred replacement.
    """
    warnings.warn(
        "isSupportedNumber is deprecated; use is_supported_number instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_supported_number(num)


def is_supported_bool(num) -> TypeGuard[bool_scalars]:
    """
    Whether a scalar is an arkouda supported boolean dtype.

    Parameters
    ----------
    num: object
        A scalar.

    Returns
    -------
    bool
        True if scalar is an instance of an arkouda supported boolean dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.is_supported_bool("True")
    False
    >>> ak.is_supported_bool(True)
    True

    """
    return isinstance(num, ARKOUDA_SUPPORTED_BOOLS)


def isSupportedBool(num):
    """
    Deprecated alias for :func:`is_supported_bool`.

    This function exists for backward compatibility only. Use
    :func:`is_supported_bool` instead.

    Parameters
    ----------
    num : object
        A scalar value to test.

    Returns
    -------
    bool
        True if ``num`` is an arkouda-supported boolean dtype,
        otherwise False.

    See Also
    --------
    is_supported_bool : Preferred replacement.
    """
    warnings.warn(
        "isSupportedBool is deprecated; use is_supported_bool instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_supported_bool(num)


def is_supported_dtype(scalar: object) -> builtins.bool:
    """
    Whether a scalar is an arkouda supported dtype.

    Parameters
    ----------
    scalar: object

    Returns
    -------
    builtins.bool
        True if scalar is an instance of an arkouda supported dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.is_supported_dtype(ak.int64(64))
    True
    >>> ak.is_supported_dtype(np.complex128(1+2j))
    False

    """
    return isinstance(scalar, ARKOUDA_SUPPORTED_DTYPES)


def isSupportedDType(num):
    """
    Deprecated alias for :func:`is_supported_dtype`.

    This function exists for backward compatibility only. Use
    :func:`is_supported_dtype` instead.

    Parameters
    ----------
    num : object
        A scalar value to test.

    Returns
    -------
    bool
        True if ``num`` is an arkouda-supported dtype,
        otherwise False.

    See Also
    --------
    is_supported_dtype : Preferred replacement.
    """
    warnings.warn(
        "isSupportedDType is deprecated; use is_supported_dtype instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_supported_dtype(num)


def resolve_scalar_dtype(val: object) -> str:
    """Try to infer what dtype arkouda_server should treat val as."""
    # ---- 1. Arkouda bigint scalar instances ----
    # (bigint_ is the scalar class; bigint is the dtype class)
    if isinstance(val, bigint_):
        return "bigint"

    # ---- 2. Python bool or numpy bool ----
    if isinstance(val, (builtins.bool, np.bool_)):
        return "bool"

    # ---- 3. Python float or numpy float ----
    if isinstance(val, (float, np.floating)):
        return "float64"

    # ---- 4. Python complex or numpy complex (mapped to float64 for now) ----
    if isinstance(val, (complex, np.complexfloating)):
        return "float64"  # TODO: support complex backend

    # ---- 5. Python / NumPy ints and uints with magnitude-aware routing ----
    if isinstance(val, (int, np.integer)):
        # Arkouda bigint family (if someone passes a bigint scalar here)
        if isinstance(val, (bigint, bigint_)):
            return "bigint"

        # bigint magnitude detection: val >= 2**64
        if val >= 2**64:
            return "bigint"

        # uint64 range: [2**63, 2**64) or explicit np.uint64
        if isinstance(val, np.uint64) or val >= 2**63:
            return "uint64"

        # everything else fits into int64
        return "int64"

    # ---- 6. Python / NumPy string ----
    if isinstance(val, (builtins.str, np.str_)):
        return "str"

    # ---- 7. Objects with dtype attribute ----
    if hasattr(val, "dtype"):
        # normalize through numpy's dtype machinery
        dt = np.dtype(getattr(val, "dtype"))
        # Map core kinds to our names, fall back to dtype.name
        if dt.kind == "b":
            return "bool"
        if dt.kind == "f":
            return "float64"
        if dt.kind in "iu":
            if dt == np.dtype("uint64"):
                return "uint64"
            return "int64"
        return dt.name

    # ---- 8. Fallback ----
    return builtins.str(type(val))


def get_byteorder(dt: np.dtype) -> str:
    """
    Get a concrete byteorder (turns '=' into '<' or '>') on the client.

    Parameters
    ----------
    dt: np.dtype
        The numpy dtype to determine the byteorder of.

    Return
    ------
    str
        Returns "<" for little endian and ">" for big endian.

    Raises
    ------
    ValueError
        Returned if sys.byteorder is not "little" or "big"

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.get_byteorder(ak.dtype(ak.int64))
    '<'

    """
    if dt.byteorder == "=":
        if sys.byteorder == "little":
            return "<"
        elif sys.byteorder == "big":
            return ">"
        else:
            raise ValueError("Client byteorder must be 'little' or 'big'")
    else:
        return dt.byteorder


def get_server_byteorder() -> str:
    """
    Get the server's byteorder.

    Return
    ------
    str
        Returns "little" for little endian and "big" for big endian.

    Raises
    ------
    ValueError
        Raised if Server byteorder is not 'little' or 'big'

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.get_server_byteorder()
    'little'

    """
    from arkouda.client import get_config

    order = get_config()["byteorder"]
    if order not in ("little", "big"):
        raise ValueError("Server byteorder must be 'little' or 'big'")
    return cast("str", order)
