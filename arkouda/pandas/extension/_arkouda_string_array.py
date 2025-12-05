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

    def __getitem__(self, key):
        result = self._data[key]
        if np.isscalar(key):
            if hasattr(result, "to_ndarray"):
                return result.to_ndarray()[()]
            else:
                return result
        return ArkoudaStringArray(result)

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
        from arkouda.numpy.pdarraycreation import full as ak_full

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
            if other_ak.size != len(self):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other_ak)

        # Case 5: unsupported type → all False
        return ArkoudaArray(ak_full(len(self), False, dtype=bool))

    def __repr__(self):
        return f"ArkoudaStringArray({self._data})"
