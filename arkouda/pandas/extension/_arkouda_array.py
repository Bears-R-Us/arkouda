from __future__ import annotations

from types import NotImplementedType
from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar, Union, overload
from typing import cast as type_cast

import numpy as np
import pandas as pd

from numpy import ndarray
from numpy.typing import NDArray
from pandas.api.extensions import ExtensionArray
from pandas.core.dtypes.dtypes import ExtensionDtype

from arkouda.numpy.dtypes import dtype as ak_dtype

from ._arkouda_extension_array import ArkoudaExtensionArray
from ._dtypes import (
    ArkoudaBigintDtype,
    ArkoudaBoolDtype,
    ArkoudaFloat64Dtype,
    ArkoudaInt64Dtype,
    ArkoudaUint8Dtype,
    ArkoudaUint64Dtype,
)


if TYPE_CHECKING:
    from arkouda.numpy.pdarrayclass import pdarray
else:
    pdarray = TypeVar("pdarray")

__all__ = ["ArkoudaArray"]


class ArkoudaArray(ArkoudaExtensionArray, ExtensionArray):
    """
    Arkouda-backed numeric/bool pandas ExtensionArray.

    Wraps or converts supported inputs into an Arkouda ``pdarray`` to serve as the
    backing store. Ensures the underlying array is 1-D and lives on the Arkouda server.

    Parameters
    ----------
    data : pdarray | ndarray | Sequence[Any] | ArkoudaArray
        Input to wrap or convert.
        - If an Arkouda ``pdarray``, it is used directly unless ``dtype`` is given
          or ``copy=True``, in which case a new array is created via ``ak.array``.
        - If a NumPy array, it is transferred to Arkouda via ``ak.array``.
        - If a Python sequence, it is converted to NumPy then to Arkouda.
        - If another ``ArkoudaArray``, its underlying ``pdarray`` is reused.
    dtype : Any, optional
        Desired dtype to cast to (NumPy dtype or Arkouda dtype string). If omitted,
        dtype is inferred from ``data``.
    copy : bool
        If True, attempt to copy the underlying data when converting/wrapping.
        Default is False.

    Raises
    ------
    TypeError
        If ``data`` cannot be interpreted as an Arkouda array-like object.
    ValueError
        If the resulting array is not one-dimensional.

    Attributes
    ----------
    default_fill_value : int
        Sentinel used when filling missing values (default: -1).

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.pandas.extension import ArkoudaArray
    >>> ArkoudaArray(ak.arange(5))
    ArkoudaArray([0 1 2 3 4])
    >>> ArkoudaArray([10, 20, 30])
    ArkoudaArray([10 20 30])
    """

    default_fill_value: int = -1

    def __init__(
        self,
        data: pdarray | ndarray | Sequence[Any] | ArkoudaArray,
        dtype: Any = None,
        copy: bool = False,
    ):
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array

        if isinstance(data, ArkoudaArray):
            data = data._data
        elif isinstance(data, (list, tuple)):
            data = type_cast(pdarray, ak_array(np.asarray(data), dtype=dtype))
        elif isinstance(data, np.ndarray):
            data = type_cast(pdarray, ak_array(data, dtype=dtype, copy=copy))
        elif not isinstance(data, pdarray):
            raise TypeError(
                f"Expected arkouda.pdarray, ndarray, or ArkoudaArray, got {type(data).__name__}"
            )
        elif dtype is not None or copy:
            data = type_cast(pdarray, ak_array(data, dtype=dtype, copy=copy))

        if getattr(data, "ndim", 1) != 1:
            raise ValueError(
                f"ArkoudaArray must be 1-dimensional, got shape {getattr(data, 'shape', None)}"
            )

        self._data = data

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        from arkouda.numpy.numeric import cast as ak_cast
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array
        from arkouda.pandas.categorical import Categorical

        from ._dtypes import ArkoudaBigintDtype

        if (
            dtype is not None
            and (
                getattr(dtype, "name", None) in {"bigint", "ak.bigint"}
                or str(dtype) in {"bigint", "ak.bigint"}
            )
            or dtype is ArkoudaBigintDtype
            or isinstance(dtype, ArkoudaBigintDtype)
        ):
            dtype = "bigint"

        if dtype is not None and hasattr(dtype, "numpy_dtype"):
            dtype = dtype.numpy_dtype

        if isinstance(scalars, Categorical):
            codes = scalars.codes

            # Some implementations might return an ArkoudaArray here
            if isinstance(codes, ArkoudaArray):
                codes = codes._data

            if not isinstance(codes, pdarray):
                raise TypeError(f"Categorical.codes expected pdarray, got {type(codes).__name__}")

            if dtype is not None:
                codes = ak_cast(codes, dtype)

            return cls(codes)

        return cls(ak_array(scalars, dtype=dtype, copy=copy))

    def __getitem__(self, key: Any) -> Any:
        """
        Retrieve one or more values using a pandas/NumPy-style indexer.

        Parameters
        ----------
        key : Any
            A valid indexer for 1D array-like data. This may be:
            - A scalar integer position (e.g. ``1``)
            - A Python ``slice`` (e.g. ``1:3``)
            - A list-like of integer positions
            - A boolean mask (NumPy array, pandas Series, or Arkouda ``pdarray``)
            - A NumPy array, pandas Index/Series, or Arkouda ``pdarray``/``Strings``.

        Returns
        -------
        Any
            A scalar value for scalar indexers, or an ``ArkoudaArray`` for sequence-like
            indexers.

        Raises
        ------
        TypeError
            If ``key`` is not a supported indexer type, or if a NumPy array or
            list-like indexer has an unsupported dtype.
        NotImplementedError
            If a list-like indexer contains mixed element dtypes (e.g. a mixture
            of booleans and integers), which is not supported.

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaArray
        >>> data = ak.arange(5)
        >>> arr = ArkoudaArray(data)

        Scalar integer index returns a Python scalar:

        >>> arr[1]
        np.int64(1)

        Slicing returns another ArkoudaArray:

        >>> arr[1:4]
        ArkoudaArray([1 2 3])

        List-like integer positions:

        >>> arr[[0, 2, 4]]
        ArkoudaArray([0 2 4])

        Boolean mask (NumPy array):

        >>> import numpy as np
        >>> mask = np.array([True, False, True, False, True])
        >>> arr[mask]
        ArkoudaArray([0 2 4])
        """
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array

        # Normalize NumPy ndarray indexers
        if isinstance(key, np.ndarray):
            if key.dtype == bool or key.dtype == np.bool_:
                key = ak_array(key, dtype=bool)
            elif np.issubdtype(key.dtype, np.integer):
                key = ak_array(key, dtype="int64")
            elif np.issubdtype(key.dtype, np.unsignedinteger):
                key = ak_array(key, dtype="uint64")
            else:
                raise TypeError(f"Unsupported NumPy index type {key.dtype}")

        # Normalize Python lists
        elif isinstance(key, list):
            if len(key) == 0:
                # Empty selection -> empty ArkoudaArray of same dtype
                empty = ak_array([], dtype=self._data.dtype)
                return self.__class__(empty)

            first = key[0]
            first_dtype = ak_dtype(first)
            for item in key:
                item_dtype = ak_dtype(item)
                if first_dtype != item_dtype:
                    raise NotImplementedError(
                        f"Mixed dtypes are not supported: {item_dtype} vs {first_dtype}"
                    )

            if isinstance(first, (bool, np.bool_)):
                key = ak_array(np.array(key, dtype=bool))
            elif isinstance(first, (int, np.integer)):
                key = ak_array(np.array(key, dtype=np.int64))
            else:
                raise TypeError(f"Unsupported list index type: {type(first)}")

        # Perform the indexing operation
        result = self._data[key]

        # Scalar key → return Python scalar
        if np.isscalar(key):
            # If server returned a pdarray of length 1, extract scalar
            if isinstance(result, pdarray) and result.size == 1:
                return result[0]
            return result

        # All other cases → wrap result in same class
        return self.__class__(result)

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Assign one or more values to the underlying Arkouda array in-place.

        Parameters
        ----------
        key : Any
            A positional indexer selecting the locations to modify. Supported forms include:

            - Scalar integer position (e.g. ``arr[3] = ...``)
            - Slice (e.g. ``arr[1:4] = ...``)
            - Boolean mask (NumPy ``ndarray`` of bools, or Python ``list`` of bools)
            - Integer indexer (NumPy ``ndarray`` of integers, Python ``list`` of ints)
            - Arkouda ``pdarray`` indexer (integer index array or boolean mask)

            For Python ``list`` indexers, all elements must be of a single supported type:
            all-bool or all-int. Mixed bool/int lists are rejected. Boolean-mask lists
            must have length equal to ``len(self)``.

        value : Any
            The value(s) to assign.

            - If a scalar (NumPy scalar or Python scalar), it is broadcast to all selected
              positions.
            - If an ``ArkoudaArray`` or Arkouda ``pdarray``, it is assigned directly.
            - Otherwise, array-like inputs (e.g. Python lists, NumPy arrays) are converted
              to an Arkouda ``pdarray`` and must be aligned with ``key``.

        Raises
        ------
        TypeError
            If a Python list indexer contains unsupported element types.
        NotImplementedError
            If a Python list indexer mixes boolean and integer elements.
        IndexError
            If a Python list boolean mask has length different from ``len(self)``.

        Notes
        -----
        This operation mutates the underlying server-side array in-place.

        Empty indexers (e.g. an empty Python list, or an empty NumPy integer indexer
        after normalization) are treated as a no-op.

        Examples
        --------
        Basic scalar assignment by position:

        >>> import arkouda as ak
        >>> import numpy as np
        >>> from arkouda.pandas.extension import ArkoudaArray
        >>> arr = ArkoudaArray(ak.arange(5))
        >>> arr[0] = 42
        >>> arr
        ArkoudaArray([42 1 2 3 4])

        Assigning with a Python list of integer positions:

        >>> arr = ArkoudaArray(ak.arange(5))
        >>> arr[[1, 3]] = 99
        >>> arr
        ArkoudaArray([0 99 2 99 4])

        Assigning with a NumPy boolean mask:

        >>> arr = ArkoudaArray(ak.arange(5))
        >>> mask = arr.to_ndarray() % 2 == 0
        >>> arr[mask] = -1
        >>> arr
        ArkoudaArray([-1 1 -1 3 -1])

        Assigning with a NumPy integer indexer:

        >>> arr = ArkoudaArray(ak.arange(5))
        >>> idx = np.array([1, 3], dtype=np.int64)
        >>> arr[idx] = 7
        >>> arr
        ArkoudaArray([0 7 2 7 4])

        Assigning from another ArkoudaArray:

        >>> arr = ArkoudaArray(ak.arange(5))
        >>> other = ArkoudaArray(ak.arange(10, 15))
        >>> idx = [1, 3, 4]
        >>> arr[idx] = other[idx]
        >>> arr
        ArkoudaArray([0 11 2 13 14])

        Python list boolean masks must match the array length:

        >>> arr = ArkoudaArray(ak.arange(5))
        >>> arr[[True, False, True]] = 0
        Traceback (most recent call last):
        ...
        IndexError: Boolean indexer has wrong length: 3 instead of 5
        """
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array

        if isinstance(key, list):
            if len(key) == 0:
                return  # empty list => noop

            # validate element types + detect mixed
            has_bool = False
            has_int = False
            for k in key:
                if isinstance(k, (bool, np.bool_)):
                    has_bool = True
                elif isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)):
                    has_int = True
                else:
                    raise TypeError(
                        "Only lists of ints or bools are supported for __setitem__ indexers."
                    )

                if has_bool and has_int:
                    raise NotImplementedError("Mixed index list dtypes (bool + int) are not supported.")

            if has_bool:
                # boolean mask must match array length
                if len(key) != len(self):
                    raise IndexError(
                        f"Boolean indexer has wrong length: {len(key)} instead of {len(self)}"
                    )
                key = np.array(key, dtype=bool)
            else:
                key = np.array(key, dtype=np.int64)

        # Normalize NumPy / Python indexers into Arkouda pdarrays where needed
        if isinstance(key, np.ndarray):
            # NumPy bool mask or integer indexer
            if key.dtype == bool or key.dtype == np.bool_ or np.issubdtype(key.dtype, np.integer):
                key = ak_array(key)
        elif isinstance(key, list):
            # Python list of bools or ints - convert to NumPy then to pdarray
            if key and isinstance(key[0], (bool, np.bool_)):
                key = ak_array(np.array(key, dtype=bool))
            elif key and isinstance(key[0], (int, np.integer)):
                key = ak_array(np.array(key, dtype=np.int64))

        if _is_empty_indexer(key):
            # Setting nothing is a no-op, consistent with numpy/pandas
            return

        # Normalize the value into something the underlying pdarray understands
        if isinstance(value, ArkoudaArray):
            value = value._data
        elif isinstance(value, pdarray):
            # already an Arkouda pdarray; nothing to do
            pass
        elif np.isscalar(value):
            # Fast path for scalar assignment

            self._data[key] = value
            return
        else:
            # Convert generic array-likes (Python lists, NumPy arrays, etc.)
            # into Arkouda pdarrays.
            value = ak_array(value)

        self._data[key] = value

    # -------------------------------------------------------------------------
    # pandas comparison protocol hook
    # -------------------------------------------------------------------------

    def _cmp_method(
        self,
        other: Any,
        op: Callable[[Any, Any], Any],
    ) -> ArkoudaArray | NotImplementedType:
        """
        Perform an elementwise comparison operation.

        This method implements the pandas ``ExtensionArray`` comparison
        protocol and may be invoked internally by pandas for comparison
        operations (e.g., ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``).

        Parameters
        ----------
        other : Any
            The right-hand operand. Supported inputs include another
            ``ArkoudaArray``, an Arkouda ``pdarray``, a NumPy ``ndarray``,
            a Python sequence (list/tuple), or a scalar value. Unsupported
            types result in ``NotImplemented``.

        op : Callable[[Any, Any], Any]
            A binary operator implementing the comparison (for example
            functions from the ``operator`` module such as ``operator.eq``
            or ``operator.lt``).

        Returns
        -------
        ArkoudaArray | NotImplementedType
            A boolean ``ArkoudaArray`` containing the elementwise comparison
            result, or ``NotImplemented`` if the operation cannot be performed.

        Notes
        -----
        Length compatibility is enforced for elementwise comparisons.
        Scalar operands are broadcast. Comparison results are always boolean.
        """
        result = self._binary_op(other, lambda a, b: op(a, b))
        if result is NotImplemented:
            return NotImplemented
        return result

    def _coerce_other_for_binop(self, other: Any) -> tuple[Any, str]:
        """
        Normalize ``other`` for binary operations.

        Parameters
        ----------
        other : Any
            The right-hand operand to normalize. Supported inputs include
            ``ArkoudaArray``, Arkouda ``pdarray``, NumPy ``ndarray``, Python
            sequences (list/tuple), and scalars.

        Returns
        -------
        tuple[Any, str]
            A pair ``(other_norm, kind)`` where:

            - ``other_norm`` is the normalized operand (a scalar or an Arkouda ``pdarray``),
              or ``None`` when unsupported.
            - ``kind`` is one of ``"scalar"``, ``"pdarray"``, or ``"notimpl"``.
        """
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array

        if isinstance(other, ArkoudaArray):
            return other._data, "pdarray"

        if isinstance(other, pdarray):
            return other, "pdarray"

        if np.isscalar(other):
            return other, "scalar"

        if isinstance(other, (list, tuple, np.ndarray)):
            return ak_array(other), "pdarray"

        return None, "notimpl"

    def _check_compatible_lengths(self, other_pdarray) -> None:
        """
        Enforce elementwise length compatibility.
        Allow scalar-broadcast pdarray of size 1.
        """
        if getattr(other_pdarray, "size", None) not in (1, len(self)):
            raise ValueError("Lengths must match for elementwise operation")

    def _binary_op(self, other: Any, op, *, require_bool: bool = False):
        """
        Core binary operator for self <op> other.
        `op` should be a callable accepting (lhs, rhs) returning a pdarray/scalar.
        """
        other_norm, kind = self._coerce_other_for_binop(other)
        if kind == "notimpl":
            return NotImplemented

        if require_bool and self._data.dtype != "bool":
            return NotImplemented

        if kind == "pdarray":
            if require_bool and getattr(other_norm, "dtype", None) != "bool":
                return NotImplemented
            # elementwise length check unless scalar-broadcast pdarray
            self._check_compatible_lengths(other_norm)
            return type(self)(op(self._data, other_norm))

        # scalar
        if require_bool and not isinstance(other_norm, (bool, np.bool_)):
            return NotImplemented
        return type(self)(op(self._data, other_norm))

    def _rbinary_op(self, other: Any, op, *, require_bool: bool = False):
        """Core binary operator for other <op> self (reverse op)."""
        other_norm, kind = self._coerce_other_for_binop(other)
        if kind == "notimpl":
            return NotImplemented

        if require_bool and self._data.dtype != "bool":
            return NotImplemented

        if kind == "pdarray":
            if require_bool and getattr(other_norm, "dtype", None) != "bool":
                return NotImplemented
            self._check_compatible_lengths(other_norm)
            return type(self)(op(other_norm, self._data))

        # scalar
        if require_bool and not isinstance(other_norm, (bool, np.bool_)):
            return NotImplemented
        return type(self)(op(other_norm, self._data))

    def _unary_op(self, op):
        """Core unary operator, returning ArkoudaArray or NotImplemented."""
        try:
            return type(self)(op(self._data))
        except Exception:
            return NotImplemented

    # -------------------------------------------------------------------------
    # Arithmetic dunders
    # -------------------------------------------------------------------------
    def __add__(self, other: Any):
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: Any):
        return self._rbinary_op(other, lambda a, b: a + b)

    def __sub__(self, other: Any):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: Any):
        return self._rbinary_op(other, lambda a, b: a - b)

    def __mul__(self, other: Any):
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other: Any):
        return self._rbinary_op(other, lambda a, b: a * b)

    def __truediv__(self, other: Any):
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other: Any):
        return self._rbinary_op(other, lambda a, b: a / b)

    def __floordiv__(self, other: Any):
        return self._binary_op(other, lambda a, b: a // b)

    def __rfloordiv__(self, other: Any):
        return self._rbinary_op(other, lambda a, b: a // b)

    def __mod__(self, other: Any):
        return self._binary_op(other, lambda a, b: a % b)

    def __rmod__(self, other: Any):
        return self._rbinary_op(other, lambda a, b: a % b)

    def __pow__(self, other: Any):
        return self._binary_op(other, lambda a, b: a**b)

    def __rpow__(self, other: Any):
        return self._rbinary_op(other, lambda a, b: a**b)

    # Unary arithmetic
    def __neg__(self):
        return self._unary_op(lambda a: -a)

    def __pos__(self):
        return self._unary_op(lambda a: +a)

    def __abs__(self):
        return self._unary_op(lambda a: abs(a))

    # -------------------------------------------------------------------------
    # Comparison dunders (elementwise, return ArkoudaArray[bool])
    # -------------------------------------------------------------------------

    def __eq__(self, other: Any):
        return self._binary_op(other, lambda a, b: a == b)

    def __ne__(self, other: Any):
        return self._binary_op(other, lambda a, b: a != b)

    def __lt__(self, other: Any):
        return self._binary_op(other, lambda a, b: a < b)

    def __le__(self, other: Any):
        return self._binary_op(other, lambda a, b: a <= b)

    def __gt__(self, other: Any):
        return self._binary_op(other, lambda a, b: a > b)

    def __ge__(self, other: Any):
        return self._binary_op(other, lambda a, b: a >= b)

    # -------------------------------------------------------------------------
    # Bitwise / logical dunders (only for bool dtype)
    # -------------------------------------------------------------------------
    def __and__(self, other: Any):
        return self._binary_op(other, lambda a, b: a & b, require_bool=True)

    def __rand__(self, other: Any):
        return self._rbinary_op(other, lambda a, b: a & b, require_bool=True)

    def __or__(self, other: Any):
        return self._binary_op(other, lambda a, b: a | b, require_bool=True)

    def __ror__(self, other: Any):
        return self._rbinary_op(other, lambda a, b: a | b, require_bool=True)

    def __xor__(self, other: Any):
        return self._binary_op(other, lambda a, b: a ^ b, require_bool=True)

    def __rxor__(self, other: Any):
        return self._rbinary_op(other, lambda a, b: a ^ b, require_bool=True)

    def __invert__(self):
        # ~ only makes sense for boolean arrays here (or integer bitwise if you later want it)
        if self._data.dtype != "bool":
            return NotImplemented
        return type(self)(~self._data)

    # docstr-coverage:excused `typing-only overload stub`
    @overload
    def astype(self, dtype: np.dtype[Any], copy: bool = True) -> NDArray[Any]: ...

    # docstr-coverage:excused `typing-only overload stub`
    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = True) -> ExtensionArray: ...

    # docstr-coverage:excused `typing-only overload stub`
    @overload
    def astype(self, dtype: Any, copy: bool = True) -> Union[ExtensionArray, NDArray[Any]]: ...

    def astype(
        self,
        dtype: Any,
        copy: bool = True,
    ) -> Union[ExtensionArray, NDArray[Any]]:
        """
        Cast the array to a specified dtype.

        Casting rules:

        * If ``dtype`` requests ``object``, returns a NumPy ``NDArray[Any]`` of
          dtype ``object`` containing the array values.
        * Otherwise, the target dtype is normalized using Arkouda's dtype
          resolution rules.
        * If the normalized dtype matches the current dtype and ``copy=False``,
          returns ``self``.
        * In all other cases, casts the underlying Arkouda array to the target
          dtype and returns an Arkouda-backed ``ArkoudaExtensionArray``.

        Parameters
        ----------
        dtype : Any
            Target dtype. May be a NumPy dtype, pandas dtype, Arkouda dtype,
            or any dtype-like object accepted by Arkouda.
        copy : bool
            Whether to force a copy when the target dtype matches the current dtype.
            Default is True.

        Returns
        -------
        Union[ExtensionArray, NDArray[Any]]
            The cast result. Returns a NumPy array only when casting to ``object``;
            otherwise returns an Arkouda-backed ExtensionArray.

        Examples
        --------
        Basic numeric casting returns an Arkouda-backed array:

        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaArray
        >>> a = ArkoudaArray(ak.array([1, 2, 3], dtype="int64"))
        >>> a.astype("float64").to_ndarray()
        array([1., 2., 3.])

        Casting to the same dtype with ``copy=False`` returns the original object:

        >>> b = a.astype("int64", copy=False)
        >>> b is a
        True

        Forcing a copy when the dtype is unchanged returns a new array:

        >>> c = a.astype("int64", copy=True)
        >>> c is a
        False
        >>> c.to_ndarray()
        array([1, 2, 3])

        Casting to ``object`` materializes the data to a NumPy array:

        >>> a.astype(object)
        array([1, 2, 3], dtype=object)

        NumPy and pandas dtype objects are also accepted:

        >>> import numpy as np
        >>> a.astype(np.dtype("bool")).to_ndarray()
        array([ True,  True,  True])
        """
        from arkouda.numpy.dtypes import dtype as ak_dtype

        # --- 1) ExtensionDtype branch (satisfies overload #2) ---
        if isinstance(dtype, ExtensionDtype):
            # pandas extension dtypes typically have .numpy_dtype
            if hasattr(dtype, "numpy_dtype"):
                dtype = dtype.numpy_dtype

            if copy is False and self.dtype.numpy_dtype == dtype:
                return self

            casted = self._data.astype(dtype)
            return type_cast(ExtensionArray, ArkoudaExtensionArray._from_sequence(casted))

        # --- 2) object -> numpy (satisfies overload #1 / general) ---

        if dtype in (object, np.object_, "object", np.dtype("O")):
            return self.to_ndarray().astype(object, copy=False)

        dtype = ak_dtype(dtype)

        if copy is False and self.dtype.numpy_dtype == dtype:
            return self

        casted = self._data.astype(dtype)
        return ArkoudaExtensionArray._from_sequence(casted)

    @property
    def dtype(self):
        if self._data.dtype == "int64":
            return ArkoudaInt64Dtype()
        elif self._data.dtype == "float64":
            return ArkoudaFloat64Dtype()
        elif self._data.dtype == "bool":
            return ArkoudaBoolDtype()
        elif self._data.dtype == "uint64":
            return ArkoudaUint64Dtype()
        elif self._data.dtype == "uint8":
            return ArkoudaUint8Dtype()
        elif self._data.dtype == "bigint":
            return ArkoudaBigintDtype()
        else:
            raise TypeError(f"Unsupported dtype {self._data.dtype}")

    @property
    def nbytes(self):
        return self._data.nbytes

    def equals(self, other):
        if not isinstance(other, ArkoudaArray):
            return False
        return self._data.equals(other._data)

    def _reduce(self, name: str, skipna: bool = True, **kwargs: Any) -> Any:
        """
        Reduce the array to a single value (or a small array result) using a named reduction.

        This implements the pandas ExtensionArray reduction protocol and is called by pandas
        for operations like ``Series.sum()`` and ``Series.min()``.

        Parameters
        ----------
        name : str
            Name of the reduction to perform (e.g., ``"sum"``, ``"min"``, ``"std"``).
        skipna : bool
            Whether to ignore missing values. Accepted for pandas compatibility.
            Default is True.

            NOTE
            ----
            ``skipna`` semantics are **not fully supported** for Arkouda-backed arrays.
            Except where explicitly implemented (e.g., ``count`` for float64),
            reductions are delegated directly to Arkouda operations, which typically
            propagate ``NaN`` values rather than skipping them.

            As a result, reductions such as ``sum``, ``mean``, ``min``, and ``max`` on
            float arrays may return ``NaN`` even when ``skipna=True``.
        **kwargs : Any
            Additional keyword arguments forwarded by pandas. Currently unused unless
            explicitly supported.

        Returns
        -------
        Any
            A scalar result for scalar reductions (e.g., ``sum``, ``min``, ``mean``), or an
            ``ArkoudaArray`` for array-returning reductions such as ``mode``.

        Raises
        ------
        TypeError
            If ``name`` is not a recognized reduction.
        """
        from arkouda.numpy import isnan

        ddof = int(kwargs.get("ddof", 1))

        op = name.lower()
        data = self._data

        def _count_nonmissing() -> int:
            # Minimal NA handling: treat NaN as missing only for float64.
            if data.dtype == "float64":
                return int((~isnan(data)).sum())
            return int(data.size)

        def _first() -> Any:
            if data.size == 0:
                # Throw an error for now; pandas often raises or returns NA depending on context.
                raise ValueError("Reduction 'first' requires at least one element")
            return data[0]

        def _var() -> Any:
            return data.var(ddof=ddof)

        def _std() -> Any:
            return data.std(ddof=ddof)

        # All listed reductions are guaranteed to exist on pdarray for all dtypes
        scalar_fns: dict[str, Callable[[], Any]] = {
            "sum": data.sum,
            "count": _count_nonmissing,
            "prod": data.prod,
            "min": data.min,
            "max": data.max,
            "mean": data.mean,
            "var": _var,
            "std": _std,
            "argmin": data.argmin,
            "argmax": data.argmax,
            "first": _first,
            "any": data.any,
            "or": data.any,  # "any" and "or" are the same op
            "all": data.all,
            "and": data.all,  # "all" and "and" are the same op
        }

        fn = scalar_fns.get(op)
        if fn is not None:
            return fn()
        else:
            #   op was not in the keys of scalar_fns:
            raise TypeError(f"Unknown reduction '{name}'")

    def __repr__(self):
        return f"ArkoudaArray({self._data})"

    #   TODO:  refine this.
    def _values_for_factorize(self):
        """
        Return (values, na_value) as NumPy for pandas.factorize.
        Ensure 'values' is 1-D numpy array and 'na_value' is the sentinel to use.
        """
        vals = self.to_ndarray()  # materialize to numpy
        if vals.dtype.kind in {"U", "S", "O"}:
            na = ""  # strings: empty as sentinel is OK for factorize
        elif vals.dtype.kind in {"i", "u"}:
            na = -1
        else:
            na = np.nan
        return vals, na

    @classmethod
    def _from_factorized(cls, uniques, original):
        # pandas gives us numpy uniques; preserve dtype by deferring to _from_sequence
        return cls._from_sequence(uniques)

    def all(self, axis=0, skipna=True, **kwargs):
        """
        Return whether all elements are True.

        This is mainly to support pandas' BaseExtensionArray.equals, which
        calls `.all()` on the result of a boolean expression.
        """
        return bool(self._data.all())

    def any(self, axis=0, skipna=True, **kwargs):
        """
        Return whether any element is True.

        Added for symmetry with `.all()` and to support potential pandas
        boolean-reduction calls.
        """
        return bool(self._data.any())

    def isna(self) -> np.ndarray:
        """
        Return a boolean mask indicating missing values.

        This method implements the pandas ExtensionArray.isna contract
        and always returns a NumPy ndarray of dtype ``bool`` with the
        same length as the array.

        Returns
        -------
        np.ndarray
            A boolean mask where ``True`` marks elements considered missing.

        Raises
        ------
        TypeError
            If the underlying data buffer does not support missing-value
            detection or cannot produce a boolean mask.
        """
        from arkouda.numpy import isnan
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import full
        from arkouda.numpy.segarray import SegArray
        from arkouda.pandas.categorical import Categorical

        data = self._data

        # SegArray
        if isinstance(data, SegArray):
            raise TypeError("isna is not supported for SegArray-backed ArkoudaArray")

        # Categorical
        if isinstance(data, Categorical):
            return (data.codes == -1).to_ndarray()
        # pdarray
        if isinstance(data, pdarray):
            if data.dtype in ("float64", "float32"):
                return (isnan(data)).to_ndarray()

            return (full(data.size, False, dtype=bool)).to_ndarray()

        return NotImplemented

    def isnull(self):
        """Alias for isna()."""
        return self.isna()

    def value_counts(self, dropna: bool = True) -> pd.Series:
        """
        Return counts of unique values as a pandas Series.

        This method computes the frequency of each distinct value in the
        underlying Arkouda array and returns the result as a pandas
        ``Series``, with the unique values as the index and their counts
        as the data.

        Parameters
        ----------
        dropna : bool
            Whether to exclude missing values. Currently, missing-value
            handling is supported only for floating-point data, where
            ``NaN`` values are treated as missing. Default is True.

        Returns
        -------
        pd.Series
            A Series containing the counts of unique values.
            The index is an ``ArkoudaArray`` of unique values, and the
            values are an ``ArkoudaArray`` of counts.

        Notes
        -----
        - Only ``dropna=True`` is supported.
        - The following pandas options are not yet implemented:
          ``normalize``, ``sort``, and ``bins``.
        - Counting is performed server-side in Arkouda; only the small
          result (unique values and counts) is materialized on the client.

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaArray
        >>>
        >>> a = ArkoudaArray(ak.array([1, 2, 1, 3, 2, 1]))
        >>> a.value_counts()
        1    3
        2    2
        3    1
        dtype: int64

        Floating-point data with NaN values:

        >>> b = ArkoudaArray(ak.array([1.0, 2.0, float("nan"), 1.0]))
        >>> b.value_counts()
        1.0    2
        2.0    1
        dtype: int64
        """
        from arkouda.numpy.numeric import isnan as ak_isnan

        data = self._data

        # Handle NA only for floats (pandas-compatible)
        if dropna and data.dtype == "float64":
            mask = ~ak_isnan(data)
            data = data[mask]

        if data.size == 0:
            return pd.Series(dtype="int64")

        keys, counts = data.value_counts()

        return_index = ArkoudaArray._from_sequence(keys)
        return_values = ArkoudaArray._from_sequence(counts)

        return pd.Series(return_values, index=return_index)


def _is_empty_indexer(key) -> bool:
    from arkouda.numpy.pdarrayclass import pdarray

    # Python containers
    if isinstance(key, (list, tuple)):
        return len(key) == 0

    # NumPy arrays
    if isinstance(key, np.ndarray):
        return key.size == 0

    # Arkouda arrays
    if isinstance(key, pdarray):
        return key.size == 0

    # Pandas Index/Series often implement __len__ and are safe here,
    # but we keep it conservative (optional):
    if isinstance(key, Sequence) and not isinstance(key, (str, bytes)):
        try:
            return len(key) == 0
        except TypeError:
            return False

    return False
