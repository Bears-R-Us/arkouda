from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, TypeVar
from typing import cast as type_cast

import numpy as np

from numpy import ndarray
from pandas.api.extensions import ExtensionArray

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
        return self._data == (other._data if isinstance(other, ArkoudaStringArray) else other)

    def __repr__(self):
        return f"ArkoudaStringArray({self._data})"
