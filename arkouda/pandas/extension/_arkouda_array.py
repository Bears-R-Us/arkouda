from typing import Any, Iterable

import numpy as np

from numpy import ndarray
from pandas.api.extensions import ExtensionArray

from arkouda.numpy.dtypes import dtype as ak_dtype
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.pdarraycreation import full as ak_full
from arkouda.numpy.pdarraycreation import pdarray

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


__all__ = ["ArkoudaArray"]


class ArkoudaArray(ArkoudaExtensionArray, ExtensionArray):
    default_fill_value = -1

    def __init__(self, data):
        if isinstance(data, (np.ndarray, Iterable)):
            data = ak_array(data)
        if not isinstance(data, pdarray):
            raise TypeError("Expected an Arkouda pdarray")
        self._data = data

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        # If pandas passes our own EA dtype, ignore it and infer from data
        if isinstance(dtype, _ArkoudaBaseDtype):
            dtype = dtype.numpy_dtype

        if dtype is not None and hasattr(dtype, "numpy_dtype"):
            dtype = dtype.numpy_dtype

        # If scalars is already a numpy array, we can preserve its dtype
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
            If ``key`` is not a supported indexer type.

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
            A valid positional indexer for the array. This may be a scalar integer,
            slice, list-like of integers, boolean mask, NumPy array, pandas Index/Series,
            or Arkouda ``pdarray``.
        value : Any
            A scalar value broadcast to the selected positions, or an array-like
            (NumPy array, Arkouda ``pdarray``, or ``ArkoudaArray``) that is
            aligned with ``key``.

        Notes
        -----
        This operation mutates the underlying server-side array in-place.

        Examples
        --------
        Basic scalar assignment by position:

        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaArray
        >>> data = ak.arange(5)
        >>> arr = ArkoudaArray(data)
        >>> arr[0] = 42
        >>> arr
        ArkoudaArray([42 1 2 3 4])

        Using a NumPy boolean mask:

        >>> data = ak.arange(5)
        >>> arr = ArkoudaArray(data)
        >>> mask = arr.to_ndarray() % 2 == 0  # even positions
        >>> arr[mask] = -1
        >>> arr
        ArkoudaArray([-1 1 -1 3 -1])

        Using a NumPy integer indexer:

        >>> data = ak.arange(5)
        >>> arr = ArkoudaArray(data)
        >>> idx = np.array([1, 3], dtype=np.int64)
        >>> arr[idx] = 99
        >>> arr
        ArkoudaArray([0 99 2 99 4])

        Assigning from another ArkoudaArray:

        >>> data = ak.arange(5)
        >>> arr = ArkoudaArray(data)
        >>> other = ArkoudaArray(ak.arange(10, 15))
        >>> idx = [1, 3, 4]
        >>> arr[idx] = other[idx]
        >>> arr
        ArkoudaArray([0 11 2 13 14])
        """
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

    def isna(self) -> ExtensionArray | ndarray[Any, Any]:
        from arkouda.numpy import isnan
        from arkouda.numpy.util import is_float

        if not is_float(self._data):
            return ak_full(self._data.size, False, dtype=bool)

        return isnan(self._data)

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

    def _reduce(self, name, skipna=True, **kwargs):
        if name == "all":
            return self._data.all()
        elif name == "any":
            return self._data.any()
        elif name == "sum":
            return self._data.sum()
        elif name == "prod":
            return self._data.prod()
        elif name == "min":
            return self._data.min()
        elif name == "max":
            return self._data.max()
        else:
            raise TypeError(f"'ArkoudaArray' with dtype arkouda does not support reduction '{name}'")

    def __eq__(self, other):
        if isinstance(other, ArkoudaArray):
            return self._data == other._data
        return self._data == other

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
