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
