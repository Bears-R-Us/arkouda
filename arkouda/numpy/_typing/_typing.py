from __future__ import annotations

import builtins
from typing import Any, Iterable, Literal, TypeAlias, TypeGuard
from typing import Union as _Union

import numpy as np

from arkouda.numpy.dtypes import bigint
from arkouda.numpy.dtypes import bool_ as ak_bool
from arkouda.numpy.dtypes import float64 as ak_float64
from arkouda.numpy.dtypes import int64 as ak_int64
from arkouda.numpy.dtypes import str_
from arkouda.numpy.dtypes import uint64 as ak_uint64
from arkouda.numpy.strings import Strings


BuiltinNumericTypes: TypeAlias = _Union[type[int], type[float], type[bool]]

ArkoudaNumericTypes: TypeAlias = _Union[type[ak_int64], type[ak_float64], type[ak_uint64], type[ak_bool]]

NumericDTypeTypes: TypeAlias = _Union[
    Literal["bigint", "float64", "int8", "int64", "uint8", "uint64", "bool", "bool_"],
    ArkoudaNumericTypes,
    BuiltinNumericTypes,
    np.dtype[Any],
    bigint,
    None,
]

StringDTypeTypes: TypeAlias = _Union[Literal["str", "str_"], type[str_], type[str], type[Strings]]

_ArrayLikeNum: TypeAlias = _Union[
    np.ndarray,  # keeps it simple; or list your NDArray[...]
    _Union[Iterable[int], Iterable[float], Iterable[bool]],
]

_ArrayLikeStr: TypeAlias = _Union[
    np.ndarray,  # or NDArray[np.str_]
    Iterable[str],
]


_StringDType: TypeAlias = _Union[
    Literal["str", "str_"],
    type[str_],
    type[str],
    type[Strings],
]

_NumericLikeDType: TypeAlias = _Union[
    # string literals for common names
    Literal[
        "bigint",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
        "bool_",
    ],
    type[builtins.bool],
    type[np.bool_],
    type[float],
    type[int],
    type[np.float64],
    type[np.float32],
    type[np.int8],
    type[np.int16],
    type[np.int32],
    type[np.int64],
    type[np.uint8],
    type[np.uint16],
    type[np.uint32],
    type[np.uint64],
    type[ak_int64],
    type[ak_uint64],
    type[ak_float64],
    type[ak_bool],
    type[bigint],
]


def is_string_dtype_hint(x: object) -> TypeGuard["_StringDType"]:
    # accept the spellings you want to map to Arkouda Strings
    return x in ("str", "str_") or x is str_ or x is str_ or x is Strings
