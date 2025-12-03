from typing import Any

import numpy as np

from pandas.api.extensions import ExtensionArray

from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.strings import Strings

from ._arkouda_extension_array import ArkoudaExtensionArray
from ._dtypes import ArkoudaStringDtype


__all__ = ["ArkoudaStringArray"]


class ArkoudaStringArray(ArkoudaExtensionArray, ExtensionArray):
    default_fill_value = ""

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            from arkouda.numpy.pdarraycreation import array as ak_array

            data = ak_array(data)

        if isinstance(data, ArkoudaStringArray):
            self._data = data._data
        elif isinstance(data, Strings):
            self._data = data
        else:
            raise TypeError(f"Expected arkouda Strings. Instead received {type(data)}.")

    @property
    def dtype(self):
        return ArkoudaStringDtype()

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(ak_array(scalars))

    def __getitem__(self, key: Any) -> Any:
        """
        Retrieve one or more string values.

        Parameters
        ----------
        key : Any
            Positional indexer. Supports:
            * scalar integer positions
            * slice objects
            * NumPy integer arrays (signed/unsigned)
            * NumPy boolean masks
            * Python lists of integers / booleans
            * Arkouda pdarray indexers (int / uint / bool)

        Returns
        -------
        Any
            A Python string for scalar access, or a new ArkoudaStringArray
            for non-scalar indexers.

        Raises
        ------
        TypeError
            If ``key`` is a NumPy array with an unsupported dtype (for example,
            a floating point or object dtype).

        Examples
        --------
        Basic scalar access:

        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaStringArray
        >>> arr = ArkoudaStringArray(ak.array(["a", "b", "c", "d"]))
        >>> arr[1]
        np.str_('b')

        Negative indexing:

        >>> arr[-1]
        np.str_('d')

        Slice indexing (returns a new ArkoudaStringArray):

        >>> arr[1:3]
        ArkoudaStringArray(['b', 'c'])

        NumPy integer array indexing:

        >>> idx = np.array([0, 2], dtype=np.int64)
        >>> arr[idx]
        ArkoudaStringArray(['a', 'c'])

        NumPy boolean mask:

        >>> mask = np.array([True, False, True, False])
        >>> arr[mask]
        ArkoudaStringArray(['a', 'c'])

        Arkouda integer indexer:

        >>> ak_idx = ak.array([3, 1])
        >>> arr[ak_idx]
        ArkoudaStringArray(['d', 'b'])

        Empty indexer returns an empty ArkoudaStringArray:

        >>> empty_idx = np.array([], dtype=np.int64)
        >>> arr[empty_idx]
        ArkoudaStringArray([])
        """
        # Normalize NumPy indexers to Arkouda pdarrays, mirroring ArkoudaArray.__getitem__
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                key = ak_array(key)
            elif key.dtype.kind in {"i"}:
                # signed integer
                key = ak_array(key, dtype="int64")
            elif key.dtype.kind in {"u"}:
                # unsigned integer
                key = ak_array(key, dtype="uint64")
            else:
                raise TypeError(f"Unsupported numpy index type {key.dtype}")

        result = self._data[key]

        # Scalar access: return a plain Python str (or scalar) instead of a Strings object
        if np.isscalar(key):
            return result

        # Non-scalar: expect an Arkouda Strings, wrap it
        if isinstance(result, Strings):
            return ArkoudaStringArray(result)

        # Fallback: if Arkouda returned something array-like but not Strings,
        # materialize via ak.array and wrap again as Strings.
        return ArkoudaStringArray(ak_array(result))

    def astype(self, dtype, copy: bool = False):
        if dtype in (object, np.object_, "object", np.dtype("O")):
            return self.to_ndarray().astype(object, copy=copy)
        # Let pandas do the rest locally
        return self.to_ndarray().astype(dtype, copy=copy)

    def isna(self):
        from arkouda.numpy.pdarraycreation import zeros

        return zeros(self._data.size, dtype="bool")

    def __eq__(self, other):
        return self._data == (other._data if isinstance(other, ArkoudaStringArray) else other)

    def __repr__(self):
        return f"ArkoudaStringArray({self._data})"
