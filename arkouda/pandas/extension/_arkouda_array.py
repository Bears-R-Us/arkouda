from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar
from typing import cast as type_cast

import numpy as np

from numpy import ndarray
from numpy.typing import NDArray
from pandas.api.extensions import ExtensionArray

from arkouda.numpy.dtypes import dtype as ak_dtype

from ._arkouda_extension_array import ArkoudaExtensionArray
from ._dtypes import (
    ArkoudaBigintDtype,
    ArkoudaBoolDtype,
    ArkoudaFloat64Dtype,
    ArkoudaInt64Dtype,
    ArkoudaUint8Dtype,
    ArkoudaUint64Dtype,
    _ArkoudaBaseDtype,
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
        from arkouda.numpy.pdarraycreation import array as ak_array

        # If pandas passes our own EA dtype, ignore it and infer from data
        if isinstance(dtype, _ArkoudaBaseDtype):
            dtype = dtype.numpy_dtype

        if dtype is not None and hasattr(dtype, "numpy_dtype"):
            dtype = dtype.numpy_dtype

        # If scalars is already a numpy array, we can preserve its dtype
        return cls(ak_array(scalars, dtype=dtype, copy=copy))

    def __getitem__(self, key):
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array

        # Convert numpy boolean mask to arkouda pdarray
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                key = ak_array(key)
            elif key.dtype.kind in {"i"}:
                key = ak_array(key, dtype="int64")
            elif key.dtype.kind in {"u"}:
                key = ak_array(key, dtype="uint64")
            else:
                raise TypeError(f"Unsupported numpy index type {key.dtype}")

        result = self._data[key]
        if np.isscalar(key):
            if isinstance(result, pdarray):
                return result[0]
            else:
                return result
        return self.__class__(result)

    #   TODO:  Simplify to use underlying array setter
    def __setitem__(self, key, value):
        from arkouda.numpy.dtypes import is_supported_int
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array

        # Convert numpy mask to pdarray if necessary
        if isinstance(key, np.ndarray) and key.dtype == bool:
            key = ak_array(key)
        elif isinstance(key, np.ndarray) and is_supported_int(key.dtype):
            key = ak_array(key)
        if isinstance(value, ArkoudaArray):
            value = value._data
        elif isinstance(value, pdarray):
            pass
        elif isinstance(value, (int, float, bool)):  # Add scalar check
            self._data[key] = value  # assign scalar to scalar position
            return
        else:
            value = ak_array(value)

        self._data[key] = value

    def astype(self, dtype, copy: bool = False):
        # Always hand back a real object-dtype ndarray when object is requested
        if dtype in (object, np.object_, "object", np.dtype("O")):
            return self.to_ndarray().astype(object, copy=copy)

        if isinstance(dtype, _ArkoudaBaseDtype):
            dtype = dtype.numpy_dtype

        # Server-side cast for numeric/bool
        try:
            npdt = np.dtype(dtype)
        except Exception:
            return self.to_ndarray().astype(dtype, copy=copy)

        from arkouda.numpy.numeric import cast as ak_cast

        if npdt.kind in {"i", "u", "f", "b"}:
            return type(self)(ak_cast(self._data, ak_dtype(npdt.name)))

        # Fallback: local cast
        return self.to_ndarray().astype(npdt, copy=copy)

    def isna(self) -> NDArray[np.bool_]:
        from arkouda.numpy import isnan
        from arkouda.numpy.pdarraycreation import full as ak_full
        from arkouda.numpy.util import is_float

        if not is_float(self._data):
            return (
                ak_full(self._data.size, False, dtype=bool).to_ndarray().astype(dtype=bool, copy=False)
            )

        return isnan(self._data).to_ndarray().astype(dtype=bool, copy=False)

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

    def __eq__(self, other):
        """
        Elementwise equality with correct pandas ExtensionArray semantics.
        Returns an ArkoudaArray of booleans.
        """
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array

        # Case 1: comparing with another ArkoudaArray
        if isinstance(other, ArkoudaArray):
            if len(self) != len(other):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other._data)

        # Case 2: comparing with an arkouda pdarray
        if isinstance(other, pdarray):
            if other.size != 1 and len(other) != len(self):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other)

        # Case 3: scalar broadcasting
        if np.isscalar(other):
            return ArkoudaArray(self._data == other)

        # Case 4: Python iterable / numpy array comparison
        if isinstance(other, (list, tuple, np.ndarray)):
            other_ak = ak_array(other)
            if other_ak.size not in (1, len(self)):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other_ak)

        return NotImplemented

    def __or__(self, other):
        """
        Elementwise boolean OR.

        This is only defined for boolean ArkoudaArray instances and returns
        an ArkoudaArray[bool]. For unsupported operand types or dtypes,
        returns NotImplemented so Python can fall back appropriately.
        """
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array

        # Only defined for boolean arrays
        if self._data.dtype != "bool":
            return NotImplemented

        # ArkoudaArray | ArkoudaArray
        if isinstance(other, ArkoudaArray):
            if other._data.dtype != "bool":
                return NotImplemented
            if len(self) != len(other):
                raise ValueError("Lengths must match for elementwise boolean operations")
            return ArkoudaArray(self._data | other._data)

        # ArkoudaArray | pdarray
        if isinstance(other, pdarray):
            if other.dtype != "bool":
                return NotImplemented
            if other.size not in (1, len(self)):
                raise ValueError("Lengths must match for elementwise boolean operations")
            return ArkoudaArray(self._data | other)

        # ArkoudaArray | scalar bool
        if isinstance(other, (bool, np.bool_)):
            return ArkoudaArray(self._data | other)

        # ArkoudaArray | numpy array / Python sequence
        if isinstance(other, (list, tuple, np.ndarray)):
            other_ak = ak_array(other, dtype=bool)
            if other_ak.size not in (1, len(self)):
                raise ValueError("Lengths must match for elementwise boolean operations")
            return ArkoudaArray(self._data | other_ak)

        return NotImplemented

    def __ror__(self, other):
        """
        Elementwise boolean OR with reversed operands.

        This allows expressions like `pdarray | ArkoudaArray` to be handled
        by ArkoudaArray when appropriate.
        """
        result = self.__or__(other)
        if result is NotImplemented:
            return NotImplemented
        return result

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
