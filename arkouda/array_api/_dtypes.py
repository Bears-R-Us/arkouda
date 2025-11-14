import arkouda as ak

from arkouda import dtype as akdtype


__all__: list[str] = []


# Note: we use dtype objects instead of dtype classes. The spec does not
# require any behavior on dtypes other than equality.
int8 = ak.int8
int16 = ak.int16
int32 = ak.int32
int64 = ak.int64
uint8 = ak.uint8
uint16 = ak.uint16
uint32 = ak.uint32
uint64 = ak.uint64
float32 = ak.float32
float64 = ak.float64
complex64 = ak.complex64
complex128 = ak.complex128
bool_ = ak.bool_

_all_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
    bool,
)
_boolean_dtypes = (bool,)
_real_floating_dtypes = (float32, float64)
_floating_dtypes = (float32, float64, complex64, complex128)
_complex_floating_dtypes = (complex64, complex128)
_integer_dtypes = (int8, int16, int32, int64, uint8, uint16, uint32, uint64)
_signed_integer_dtypes = (int8, int16, int32, int64)
_unsigned_integer_dtypes = (uint8, uint16, uint32, uint64)
_integer_or_boolean_dtypes = (
    bool,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
_real_numeric_dtypes = (
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
_numeric_dtypes = (
    float32,
    float64,
    complex64,
    complex128,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

_dtype_categories = {
    "all": _all_dtypes,
    "real numeric": _real_numeric_dtypes,
    "numeric": _numeric_dtypes,
    "integer": _integer_dtypes,
    "integer or boolean": _integer_or_boolean_dtypes,
    "boolean": _boolean_dtypes,
    "real floating-point": _floating_dtypes,
    "complex floating-point": _complex_floating_dtypes,
    "floating-point": _floating_dtypes,
}


# Note: the spec defines a restricted type promotion table compared to NumPy.
# In particular, cross-kind promotions like integer + float or boolean +
# integer are not allowed, even for functions that accept both kinds.
# Additionally, NumPy promotes signed integer + uint64 to float64, but this
# promotion is not allowed here. To be clear, Python scalar int objects are
# allowed to promote to floating-point dtypes, but only in array operators
# (see Array._promote_scalar) method in _array_object.py.
_promotion_table = {
    (akdtype(int8), akdtype(int8)): int8,
    (akdtype(int8), akdtype(int16)): int16,
    (akdtype(int8), akdtype(int32)): int32,
    (akdtype(int8), akdtype(int64)): int64,
    (akdtype(int16), akdtype(int8)): int16,
    (akdtype(int16), akdtype(int16)): int16,
    (akdtype(int16), akdtype(int32)): int32,
    (akdtype(int16), akdtype(int64)): int64,
    (akdtype(int32), akdtype(int8)): int32,
    (akdtype(int32), akdtype(int16)): int32,
    (akdtype(int32), akdtype(int32)): int32,
    (akdtype(int32), akdtype(int64)): int64,
    (akdtype(int64), akdtype(int8)): int64,
    (akdtype(int64), akdtype(int16)): int64,
    (akdtype(int64), akdtype(int32)): int64,
    (akdtype(int64), akdtype(int64)): int64,
    (akdtype(uint8), akdtype(uint8)): uint8,
    (akdtype(uint8), akdtype(uint16)): uint16,
    (akdtype(uint8), akdtype(uint32)): uint32,
    (akdtype(uint8), akdtype(uint64)): uint64,
    (akdtype(uint16), akdtype(uint8)): uint16,
    (akdtype(uint16), akdtype(uint16)): uint16,
    (akdtype(uint16), akdtype(uint32)): uint32,
    (akdtype(uint16), akdtype(uint64)): uint64,
    (akdtype(uint32), akdtype(uint8)): uint32,
    (akdtype(uint32), akdtype(uint16)): uint32,
    (akdtype(uint32), akdtype(uint32)): uint32,
    (akdtype(uint32), akdtype(uint64)): uint64,
    (akdtype(uint64), akdtype(uint8)): uint64,
    (akdtype(uint64), akdtype(uint16)): uint64,
    (akdtype(uint64), akdtype(uint32)): uint64,
    (akdtype(uint64), akdtype(uint64)): uint64,
    (akdtype(int8), akdtype(uint8)): int16,
    (akdtype(int8), akdtype(uint16)): int32,
    (akdtype(int8), akdtype(uint32)): int64,
    (akdtype(int16), akdtype(uint8)): int16,
    (akdtype(int16), akdtype(uint16)): int32,
    (akdtype(int16), akdtype(uint32)): int64,
    (akdtype(int32), akdtype(uint8)): int32,
    (akdtype(int32), akdtype(uint16)): int32,
    (akdtype(int32), akdtype(uint32)): int64,
    (akdtype(int64), akdtype(uint8)): int64,
    (akdtype(int64), akdtype(uint16)): int64,
    (akdtype(int64), akdtype(uint32)): int64,
    (akdtype(uint8), akdtype(int8)): int16,
    (akdtype(uint16), akdtype(int8)): int32,
    (akdtype(uint32), akdtype(int8)): int64,
    (akdtype(uint8), akdtype(int16)): int16,
    (akdtype(uint16), akdtype(int16)): int32,
    (akdtype(uint32), akdtype(int16)): int64,
    (akdtype(uint8), akdtype(int32)): int32,
    (akdtype(uint16), akdtype(int32)): int32,
    (akdtype(uint32), akdtype(int32)): int64,
    (akdtype(uint8), akdtype(int64)): int64,
    (akdtype(uint16), akdtype(int64)): int64,
    (akdtype(uint32), akdtype(int64)): int64,
    (akdtype(float32), akdtype(float32)): float32,
    (akdtype(float32), akdtype(float64)): float64,
    (akdtype(float64), akdtype(float32)): float64,
    (akdtype(float64), akdtype(float64)): float64,
    (akdtype(complex64), akdtype(complex64)): complex64,
    (akdtype(complex64), akdtype(complex128)): complex128,
    (akdtype(complex128), akdtype(complex64)): complex128,
    (akdtype(complex128), akdtype(complex128)): complex128,
    (akdtype(float32), akdtype(complex64)): complex64,
    (akdtype(float32), akdtype(complex128)): complex128,
    (akdtype(float64), akdtype(complex64)): complex128,
    (akdtype(float64), akdtype(complex128)): complex128,
    (akdtype(complex64), akdtype(float32)): complex64,
    (akdtype(complex64), akdtype(float64)): complex128,
    (akdtype(complex128), akdtype(float32)): complex128,
    (akdtype(complex128), akdtype(float64)): complex128,
    (akdtype(bool_), akdtype(bool_)): bool_,
}


def _result_type(type1, type2):
    if (akdtype(type1), akdtype(type2)) in _promotion_table:
        return _promotion_table[akdtype(type1), akdtype(type2)]
    raise TypeError(f"{type1} and {type2} cannot be type promoted together")
