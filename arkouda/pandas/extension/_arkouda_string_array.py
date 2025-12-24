from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, TypeVar
from typing import cast as type_cast

import numpy as np

from numpy import ndarray
from pandas.api.extensions import ExtensionArray

from arkouda.numpy.dtypes import str_
from arkouda.pandas.extension import ArkoudaArray

from ._arkouda_extension_array import ArkoudaExtensionArray
from ._dtypes import ArkoudaStringDtype


if TYPE_CHECKING:
    from arkouda.numpy.strings import Strings
else:
    Strings = TypeVar("Strings")

__all__ = ["ArkoudaStringArray"]


class ArkoudaStringArray(ArkoudaExtensionArray, ExtensionArray):
    """
    Arkouda-backed string pandas ExtensionArray.

    Ensures the underlying data is an Arkouda ``Strings`` object. Accepts existing
    ``Strings`` or converts from NumPy arrays and Python sequences of strings.

    Parameters
    ----------
    data : Strings | ndarray | Sequence[Any] | ArkoudaStringArray
        Input to wrap or convert.
        - If ``Strings``, used directly.
        - If NumPy/sequence, converted via ``ak.array``.
        - If another ``ArkoudaStringArray``, its backing ``Strings`` is reused.

    Raises
    ------
    TypeError
        If ``data`` cannot be converted to Arkouda ``Strings``.

    Attributes
    ----------
    default_fill_value : str
        Sentinel used when filling missing values (default: "").
    """

    default_fill_value: str = ""

    def __init__(self, data: Strings | ndarray | Sequence[Any] | "ArkoudaStringArray"):
        from arkouda.numpy.pdarraycreation import array as ak_array
        from arkouda.numpy.strings import Strings

        if isinstance(data, ArkoudaStringArray):
            self._data = data._data
            return

        if isinstance(data, (np.ndarray, list, tuple)):
            data = type_cast(Strings, ak_array(data, dtype="str_"))

        if not isinstance(data, Strings):
            raise TypeError(f"Expected arkouda.Strings, got {type(data).__name__}")

        self._data = data

    @property
    def dtype(self):
        return ArkoudaStringDtype()

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        from arkouda.numpy.pdarraycreation import array as ak_array

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
        from arkouda.numpy.pdarraycreation import array as ak_array
        from arkouda.numpy.strings import Strings

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
        """
        Elementwise equality for string arrays using pandas ExtensionArray semantics.
        Returns ArkoudaArray of booleans.
        """
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array

        # Case 1: ArkoudaStringArray
        if isinstance(other, ArkoudaStringArray):
            if len(self) != len(other):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other._data)

        # Case 2: arkouda pdarray (should contain encoded string indices)
        if isinstance(other, pdarray):
            if other.size not in (1, len(self)):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other)

        # Case 3: scalar (string or bytes)
        if isinstance(other, (str, str_)):
            return ArkoudaArray(self._data == other)

        # Case 4: numpy array or Python sequence
        if isinstance(other, (list, tuple, np.ndarray)):
            other_ak = ak_array(other)
            if other_ak.size == 1:
                return ArkoudaArray(self._data == other_ak[0])
            if other_ak.size != len(self):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other_ak)

        # Case 5: unsupported type
        return NotImplemented

    def __repr__(self):
        return f"ArkoudaStringArray({self._data})"
