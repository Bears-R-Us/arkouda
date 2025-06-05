from __future__ import annotations

import builtins
import sys
from enum import Enum
from typing import TYPE_CHECKING, List, Union, cast

import numpy as np
from numpy import (
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

if TYPE_CHECKING:
    from arkouda.pdarrayclass import pdarray

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
    "bitType",
    "bool_scalars",
    "can_cast",
    "complex128",
    "complex64",
    "dtype",
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
    "isSupportedBool",
    "isSupportedDType",
    "isSupportedFloat",
    "isSupportedInt",
    "isSupportedNumber",
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


def _datatype_check(the_dtype, allowed_list, name):
    if the_dtype not in allowed_list:
        raise TypeError(f"{name} only implements types {allowed_list}")


def dtype(dtype):
    """
    Create a data type object.

    Parameters
    ----------
    dtype: object
        Object to be converted to a data type object.

    Returns
    -------
    type

    """
    # we had to create our own bigint type since numpy
    # gives them dtype=object there's no np equivalent
    if (
        (isinstance(dtype, str) and dtype == "bigint")
        or isinstance(dtype, bigint)
        or (hasattr(dtype, "name") and dtype.name == "bigint")
    ):
        return bigint()
    if isinstance(dtype, str) and dtype in ["Strings"]:
        return np.dtype(np.str_)

    if isinstance(dtype, int):
        if 0 < dtype and dtype < 2**64:
            return np.dtype(np.uint64)
        if dtype >= 2**64:
            return bigint()
        else:
            return np.dtype(np.int64)
    if isinstance(dtype, float):
        return np.dtype(np.float64)
    if isinstance(dtype, bool):
        return np.dtype(np.bool)
    return np.dtype(dtype)


def can_cast(from_, to) -> builtins.bool:
    """
    Returns True if cast between data types can occur according to the casting rule.

    Parameters
    __________

    from_: dtype, dtype specifier, NumPy scalar, or pdarray
        Data type, NumPy scalar, or array to cast from.
    to: dtype or dtype specifier
        Data type to cast to.

    Return
    ------
    bool
        True if cast can occur according to the casting rule.

    """
    if isSupportedInt(from_):
        if (from_ < 2**64) and (from_ >= 0) and (to == dtype(uint64)):
            return True

    if (np.isscalar(from_) or _is_dtype_in_union(from_, numeric_scalars)) and not isinstance(
        from_, (int, float, complex)
    ):
        return np.can_cast(from_, to)

    return False


def result_type(*args: Union[pdarray, np.dtype, type]) -> Union[np.dtype, type]:
    """
    Determine the promoted result dtype of inputs, including support for Arkouda's bigint.

    Determine the result dtype that would be returned by a NumPy-like operation
    on the provided input arguments, accounting for Arkouda's extended types
    such as ak.bigint.

    This function mimics numpy.result_type, with support for Arkouda types.

    Parameters
    ----------
    *args: Union[pdarray, np.dtype, type]
        One or more input objects. These can be NumPy arrays, dtypes, Python
        scalar types, or Arkouda pdarrays.

    Returns
    -------
    Union[np.dtype, type]
        The dtype (or equivalent Arkouda type) that results from applying
        type promotion rules to the inputs.

    Notes
    -----
    This function is meant to be a drop-in replacement for numpy.result_type
    but includes logic to support Arkouda's bigint types.
    """
    from numpy.typing import DTypeLike

    has_bigint = False
    has_float = False
    np_dtypes: List[DTypeLike] = []

    for arg in args:
        if isinstance(arg, (np.dtype, type)):
            dt = arg
        elif hasattr(arg, "dtype"):
            dt = arg.dtype
        else:
            dt = np.result_type(arg)

        # Normalize Arkouda custom dtypes
        if dt == bigint:
            has_bigint = True
        elif _is_dtype_in_union(dt, Union[float, float64]):
            has_float = True
            np_dtypes.append(np.dtype(np.float64))
        elif _is_dtype_in_union(dt, Union[int, int64]):
            np_dtypes.append(np.dtype(np.int64))
        elif isinstance(dt, np.dtype):
            if dt.kind == "f":
                has_float = True
            np_dtypes.append(dt)
        else:
            # Fallback for unrecognized types
            np_dtypes.append(np.result_type(dt))

    if has_bigint:
        if has_float:
            return float64
        return bigint
    else:
        return np.result_type(*np_dtypes)


def _is_dtype_in_union(dtype, union_type) -> builtins.bool:
    """
    Check if a given type is in a typing.Union.

    Args
    ----
        dtype (type): The type to check for.
        union_type (type): The typing.Union type to check against.

    Returns
    -------
        bool: True if the dtype is in the union_type, False otherwise.
    """
    return hasattr(union_type, "__args__") and dtype in union_type.__args__


def _val_isinstance_of_union(val, union_type) -> builtins.bool:
    """
    Check if a given val is an instance of one of the types in the typing.Union

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


class bigint:
    """
    Datatype for representing integers of variable size.

    May be used for integers that exceed 64 bits.
    """

    # an estimate of the itemsize of bigint (128 bytes)
    itemsize = 128
    name = "bigint"
    ndim = 0
    shape = ()

    def __init__(self):
        self.kind = "ui"

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"dtype({self.name})"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(dtype(other), bigint):
            return True
        return False

    def __neq__(self, other):
        return not (self == other)

    def type(self, x):
        return int(x)


intTypes = frozenset((dtype("int64"), dtype("uint64"), dtype("uint8")))
bitType = uint64

# Union aliases used for static and runtime type checking
bool_scalars = Union[builtins.bool, np.bool_]
float_scalars = Union[float, np.float64, np.float32]
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
]
numeric_scalars = Union[float_scalars, int_scalars]
numeric_and_bool_scalars = Union[bool_scalars, numeric_scalars]
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
]
str_scalars = Union[str, np.str_]
all_scalars = Union[bool_scalars, numeric_scalars, numpy_scalars, str_scalars]

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
        a DType as a request parameter
        """
        return self.value

    def __repr__(self) -> str:
        """
        Overridden method returns value, which is useful in outputting
        a DType as a request parameter
        """
        return self.value


ARKOUDA_SUPPORTED_BOOLS = (bool, np.bool_)

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


def isSupportedInt(num):
    """
    Whether a scalar is an arkouda supported integer dtype.

    Parameters
    ----------
    scalar: object

    Returns
    -------
    bool
        True if scalar is an instance of an arkouda supported integer dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.isSupportedInt(79)
    True
    >>> ak.isSupportedInt(54.9)
    False

    """
    return isinstance(num, ARKOUDA_SUPPORTED_INTS)


def isSupportedFloat(num):
    """
    Whether a scalar is an arkouda supported float dtype.

    Parameters
    ----------
    scalar: object

    Returns
    -------
    bool
        True if scalar is an instance of an arkouda supported float dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.isSupportedFloat(56)
    False
    >>> ak.isSupportedFloat(56.7)
    True

    """
    return isinstance(num, ARKOUDA_SUPPORTED_FLOATS)


def isSupportedNumber(num):
    """
    Whether a scalar is an arkouda supported numeric dtype.

    Parameters
    ----------
    scalar: object

    Returns
    -------
    bool
        True if scalar is an instance of an arkouda supported numeric dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.isSupportedNumber(45.9)
    True
    >>> ak.isSupportedNumber("string")
    False

    """
    return isinstance(num, ARKOUDA_SUPPORTED_NUMBERS)


def isSupportedBool(num):
    """
    Whether a scalar is an arkouda supported boolean dtype.

    Parameters
    ----------
    scalar: object

    Returns
    -------
    bool
        True if scalar is an instance of an arkouda supported boolean dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.isSupportedBool("True")
    False
    >>> ak.isSupportedBool(True)
    True

    """
    return isinstance(num, ARKOUDA_SUPPORTED_BOOLS)


def isSupportedDType(scalar: object) -> bool:
    """
    Whether a scalar is an arkouda supported dtype.

    Parameters
    ----------
    scalar: object

    Returns
    -------
    bool
        True if scalar is an instance of an arkouda supported dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.isSupportedDType(ak.int64(64))
    True
    >>> ak.isSupportedDType(np.complex128(1+2j))
    False

    """
    return isinstance(scalar, ARKOUDA_SUPPORTED_DTYPES)


def resolve_scalar_dtype(val: object) -> str:
    """
    Try to infer what dtype arkouda_server should treat val as.

    Parameters
    ----------
    val: object
        The object to determine the dtype of.

    Return
    ------
    str
        The dtype name, if it can be resolved, otherwise the type (as str).

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.resolve_scalar_dtype(1)
    'int64'
    >>> ak.resolve_scalar_dtype(2.0)
    'float64'

    """

    # Python bool or np.bool
    if isinstance(val, builtins.bool) or (
        hasattr(val, "dtype") and cast(np.bool_, val).dtype.kind == "b"
    ):
        return "bool"
    # Python int or np.int* or np.uint*
    elif isinstance(val, int) or (hasattr(val, "dtype") and cast(np.uint, val).dtype.kind in "ui"):
        # we've established these are int, uint, or bigint,
        # so we can do comparisons
        if isSupportedInt(val) and val >= 2**64:  # type: ignore
            return "bigint"
        elif isinstance(val, np.uint64) or val >= 2**63:  # type: ignore
            return "uint64"
        else:
            return "int64"
    # Python float or np.float*
    elif isinstance(val, float) or (hasattr(val, "dtype") and cast(np.float64, val).dtype.kind == "f"):
        return "float64"
    elif isinstance(val, complex) or (hasattr(val, "dtype") and cast(np.float64, val).dtype.kind == "c"):
        return "float64"  # TODO: actually support complex values in the backend
    elif isinstance(val, builtins.str) or isinstance(val, np.str_):
        return "str"
    # Other numpy dtype
    elif hasattr(val, "dtype"):
        return cast(np.dtype, val).name
    # Other python type
    else:
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
    Get the server's byteorder

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
