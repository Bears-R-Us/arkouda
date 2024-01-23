"""
Wrapper class around the ndarray object for the array API standard.

The array API standard defines some behaviors differently than ndarray, in
particular, type promotion rules are different (the standard has no
value-based casting). The standard also specifies a more limited subset of
array methods and functionalities than are implemented on ndarray. Since the
goal of the array_api namespace is to be a minimal implementation of the array
API standard, we need to define a separate wrapper class for the array_api
namespace.

The standard compliant class is only a wrapper class. It is *not* a subclass
of ndarray.
"""

from __future__ import annotations

from enum import IntEnum
from ._dtypes import (
    # _all_dtypes,
    _boolean_dtypes,
    _integer_dtypes,
    # _integer_or_boolean_dtypes,
    _floating_dtypes,
    _complex_floating_dtypes,
    # _numeric_dtypes,
    _result_type,
    _dtype_categories,
)

from typing import TYPE_CHECKING, Optional, Tuple, Union
import types

if TYPE_CHECKING:
    from ._typing import Device, Dtype

import arkouda as ak
import numpy as np

from arkouda import array_api


class Array:
    """
    n-d array object for the array API namespace.

    See the docstring of :py:obj:`np.ndarray <numpy.ndarray>` for more
    information.

    This is a wrapper around numpy.ndarray that restricts the usage to only
    those things that are required by the array API namespace. Note,
    attributes on this object that start with a single underscore are not part
    of the API specification and should only be used internally. This object
    should not be constructed directly. Rather, use one of the creation
    functions, such as asarray().

    """

    _array: ak.pdarray
    _empty: bool

    # Use a custom constructor instead of __init__, as manually initializing
    # this class is not supported API.
    @classmethod
    def _new(cls, x, /, empty: bool = False):
        """
        This is a private method for initializing the array API Array
        object.

        Functions outside of the array_api submodule should not use this
        method. Use one of the creation functions instead, such as
        ``asarray``.

        """
        obj = super().__new__(cls)
        obj._array = x
        obj._empty = empty
        return obj

    # Prevent Array() from working
    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "The array_api Array object should not be instantiated directly. \
            Use an array creation function, such as asarray(), instead."
        )

    def tolist(self):
        """
        Convert the array to a Python list or nested lists
        """
        x = self._array.to_list()
        if self.shape == ():
            # to match numpy, return a scalar for a 0-dimensional array
            return x[0]
        else:
            return x

    def to_ndarray(self):
        """
        Convert the array to a numpy ndarray
        """
        return self._array.to_ndarray()

    # These functions are not required by the spec, but are implemented for
    # the sake of usability.

    def __str__(self: Array, /) -> str:
        """
        Performs the operation __str__.
        """
        return self._array.__str__()

    def __repr__(self: Array, /) -> str:
        """
        Performs the operation __repr__.
        """
        return self._array.__repr__()

    # # This function is not required by the spec, but we implement it here for
    # # convenience so that np.asarray(np.array_api.Array) will work.
    # def __array__(self, dtype: None | np.dtype[Any] = None) -> npt.NDArray[Any]:
    #     """
    #     Warning: this method is NOT part of the array API spec. Implementers
    #     of other libraries need not include it, and users should not assume it
    #     will be present in other implementations.

    #     """
    #     return ak.asarray(self._array, dtype=dtype)

    # These are various helper functions to make the array behavior match the
    # spec in places where it either deviates from or is more strict than
    # NumPy behavior

    def _check_allowed_dtypes(
        self, other: bool | int | float | Array, dtype_category: str, op: str
    ) -> Array:
        """
        Helper function for operators to only allow specific input dtypes

        Use like

            other = self._check_allowed_dtypes(other, 'numeric', '__add__')
            if other is NotImplemented:
                return other
        """

        if self.dtype not in _dtype_categories[dtype_category]:
            raise TypeError(f"Only {dtype_category} dtypes are allowed in {op}")
        if isinstance(other, (int, complex, float, bool)):
            other = self._promote_scalar(other)
        elif isinstance(other, Array):
            if other.dtype not in _dtype_categories[dtype_category]:
                raise TypeError(f"Only {dtype_category} dtypes are allowed in {op}")
        else:
            return NotImplemented

        assert isinstance(other, Array)

        # This will raise TypeError for type combinations that are not allowed
        # to promote in the spec (even if the NumPy array operator would
        # promote them).
        res_dtype = _result_type(self.dtype, other.dtype)
        if op.startswith("__i"):
            # Note: NumPy will allow in-place operators in some cases where
            # the type promoted operator does not match the left-hand side
            # operand. For example,

            # >>> a = np.array(1, dtype=np.int8)
            # >>> a += np.array(1, dtype=np.int16)

            # The spec explicitly disallows this.
            if res_dtype != self.dtype:
                raise TypeError(f"Cannot perform {op} with dtypes {self.dtype} and {other.dtype}")

        return other

    # Helper function to match the type promotion rules in the spec
    def _promote_scalar(self, scalar) -> Array:
        """
        Returns a promoted version of a Python scalar appropriate for use with
        operations on self.

        This may raise an OverflowError in cases where the scalar is an
        integer that is too large to fit in a NumPy integer dtype, or
        TypeError when the scalar type is incompatible with the dtype of self.
        """
        # Note: Only Python scalar types that match the array dtype are
        # allowed.
        if isinstance(scalar, bool):
            if self.dtype not in _boolean_dtypes:
                raise TypeError("Python bool scalars can only be promoted with bool arrays")
        elif isinstance(scalar, int):
            if self.dtype in _boolean_dtypes:
                raise TypeError("Python int scalars cannot be promoted with bool arrays")
            if self.dtype in _integer_dtypes:
                info = np.iinfo(int)
                if not (info.min <= scalar <= info.max):
                    raise OverflowError(
                        "Python int scalars must be within the bounds of the dtype for integer arrays"
                    )
            # int + array(floating) is allowed
        elif isinstance(scalar, float):
            if self.dtype not in _floating_dtypes:
                raise TypeError("Python float scalars can only be promoted with floating-point arrays.")
        elif isinstance(scalar, complex):
            if self.dtype not in _complex_floating_dtypes:
                raise TypeError(
                    "Python complex scalars can only be promoted with complex floating-point arrays."
                )
        else:
            raise TypeError("'scalar' must be a Python scalar")

        # Note: scalars are unconditionally cast to the same dtype as the
        # array.

        # Note: the spec only specifies integer-dtype/int promotion
        # behavior for integers within the bounds of the integer dtype.
        # Outside of those bounds we use the default NumPy behavior (either
        # cast or raise OverflowError).
        return Array._new(np.array(scalar, self.dtype))

    @staticmethod
    def _normalize_two_args(x1, x2) -> Tuple[Array, Array]:
        """
        Normalize inputs to two arg functions to fix type promotion rules

        NumPy deviates from the spec type promotion rules in cases where one
        argument is 0-dimensional and the other is not. For example:

        >>> import numpy as np
        >>> a = np.array([1.0], dtype=np.float32)
        >>> b = np.array(1.0, dtype=np.float64)
        >>> np.add(a, b) # The spec says this should be float64
        array([2.], dtype=float32)

        To fix this, we add a dimension to the 0-dimension array before passing it
        through. This works because a dimension would be added anyway from
        broadcasting, so the resulting shape is the same, but this prevents NumPy
        from not promoting the dtype.
        """
        # Another option would be to use signature=(x1.dtype, x2.dtype, None),
        # but that only works for ufuncs, so we would have to call the ufuncs
        # directly in the operator methods. One should also note that this
        # sort of trick wouldn't work for functions like searchsorted, which
        # don't do normal broadcasting, but there aren't any functions like
        # that in the array API namespace.
        if x1.ndim == 0 and x2.ndim != 0:
            # The _array[None] workaround was chosen because it is relatively
            # performant. broadcast_to(x1._array, x2.shape) is much slower. We
            # could also manually type promote x2, but that is more complicated
            # and about the same performance as this.
            x1 = Array._new(x1._array[None])
        elif x2.ndim == 0 and x1.ndim != 0:
            x2 = Array._new(x2._array[None])
        return (x1, x2)

    # Note: A large fraction of allowed indices are disallowed here (see the
    # docstring below)
    def _validate_index(self, key):
        raise IndexError("not implemented")

    # Everything below this line is required by the spec.

    def __abs__(self: Array, /) -> Array:
        """
        Performs the operation __abs__.
        """
        return self

    def __add__(self: Array, other: Union[int, float, Array], /) -> Array:
        if isinstance(other, (int, float)):
            return Array._new(self._array + other)
        else:
            return Array._new(self._array + other._array)

    def __and__(self: Array, other: Union[int, bool, Array], /) -> Array:
        return self

    def __array_namespace__(self: Array, /, *, api_version: Optional[str] = None) -> types.ModuleType:
        if api_version is not None:
            raise ValueError(f"Unrecognized array API version: {api_version!r}")
        return array_api

    def __bool__(self: Array, /) -> bool:
        # TODO: retrieve the value from a 0D array as a boolean
        return True

    def __complex__(self: Array, /) -> complex:
        return complex(1)

    def __dlpack_device__(self: Array, /) -> Tuple[IntEnum, int]:
        raise ValueError("Not implemented")

    def __eq__(self: object, other: object, /) -> bool:
        raise ValueError("Not implemented")

    def __float__(self: Array, /) -> float:
        # TODO: retrieve the value from a 0D array as a float
        return 1.0

    def __floordiv__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __ge__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __getitem__(
        self: Array,
        key: Union[int, slice, Tuple[Union[int, slice], ...], Array],
        /,
    ) -> Array:
        if isinstance(key, Array):
            # TODO: hack for testing
            return self._array[key._array]
        else:
            return self._array[key]

    def __gt__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __int__(self: Array, /) -> int:
        return 0

    def __index__(self: Array, /) -> int:
        return 0

    def __invert__(self: Array, /) -> Array:
        return self

    def __le__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __lshift__(self: Array, other: Union[int, Array], /) -> Array:
        return self

    def __lt__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __matmul__(self: Array, other: Array, /) -> Array:
        return self

    def __mod__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __mul__(self: Array, other: Union[int, float, Array], /) -> Array:
        if isinstance(other, (int, float)):
            return Array._new(self._array * other)
        else:
            return Array._new(self._array * other._array)

    def __ne__(self: object, other: object, /) -> bool:
        raise ValueError("Not implemented")

    def __neg__(self: Array, /) -> Array:
        return self

    def __or__(self: Array, other: Union[int, bool, Array], /) -> Array:
        return self

    def __pos__(self: Array, /) -> Array:
        return self

    def __pow__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __rshift__(self: Array, other: Union[int, Array], /) -> Array:
        return self

    def __setitem__(
        self,
        key: Union[int, slice, Tuple[Union[int, slice], ...], Array],
        value: Union[int, float, bool, Array],
        /,
    ) -> None:
        raise ValueError("Not implemented")

    def __sub__(self: Array, other: Union[int, float, Array], /) -> Array:
        if isinstance(other, (int, float)):
            return Array._new(self._array - other)
        else:
            return Array._new(self._array - other._array)

    # PEP 484 requires int to be a subtype of float, but __truediv__ should
    # not accept int.
    def __truediv__(self: Array, other: Union[float, Array], /) -> Array:
        if isinstance(other, (int, float)):
            return Array._new(self._array / other)
        else:
            return Array._new(self._array / other._array)

    def __xor__(self: Array, other: Union[int, bool, Array], /) -> Array:
        return self

    def __iadd__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __radd__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __iand__(self: Array, other: Union[int, bool, Array], /) -> Array:
        return self

    def __rand__(self: Array, other: Union[int, bool, Array], /) -> Array:
        return self

    def __ifloordiv__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __rfloordiv__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __ilshift__(self: Array, other: Union[int, Array], /) -> Array:
        return self

    def __rlshift__(self: Array, other: Union[int, Array], /) -> Array:
        return self

    def __imatmul__(self: Array, other: Array, /) -> Array:
        return self

    def __rmatmul__(self: Array, other: Array, /) -> Array:
        return self

    def __imod__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __rmod__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __imul__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __rmul__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __ior__(self: Array, other: Union[int, bool, Array], /) -> Array:
        return self

    def __ror__(self: Array, other: Union[int, bool, Array], /) -> Array:
        return self

    def __ipow__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __rpow__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __irshift__(self: Array, other: Union[int, Array], /) -> Array:
        return self

    def __rrshift__(self: Array, other: Union[int, Array], /) -> Array:
        return self

    def __isub__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __rsub__(self: Array, other: Union[int, float, Array], /) -> Array:
        return self

    def __itruediv__(self: Array, other: Union[float, Array], /) -> Array:
        return self

    def __rtruediv__(self: Array, other: Union[float, Array], /) -> Array:
        return self

    def __ixor__(self: Array, other: Union[int, bool, Array], /) -> Array:
        return self

    def __rxor__(self: Array, other: Union[int, bool, Array], /) -> Array:
        return self

    def to_device(self: Array, device: Device, /, stream: None = None) -> Array:
        raise ValueError("Not implemented")

    @property
    def dtype(self) -> Dtype:
        return self._array.dtype

    @property
    def device(self) -> Device:
        return "cpu"

    # Note: mT is new in array API spec (see matrix_transpose)
    @property
    def mT(self) -> Array:
        return self

    @property
    def ndim(self) -> int:
        return len(self._array.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._array.shape)

    @property
    def size(self) -> int:
        return int(self._array.size)

    @property
    def T(self) -> Array:
        raise ValueError("Not implemented")
