import numpy as np
from pandas.api.extensions import ExtensionArray

import arkouda as ak

from ._arkouda_extension_array import ArkoudaExtensionArray
from ._dtypes import ArkoudaStringDtype


__all__ = ["ArkoudaStringArray"]


class ArkoudaStringArray(ArkoudaExtensionArray, ExtensionArray):
    default_fill_value = ""

    def __init__(self, data):
        if not isinstance(data, ak.Strings):
            raise TypeError("Expected arkouda Strings")
        self._data = data

    @property
    def dtype(self):
        return ArkoudaStringDtype()

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(ak.array(scalars))

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
        return ak.zeros(self._data.size, dtype=ak.bool)

    def copy(self):
        return ArkoudaStringArray(self._data[:])

    def __eq__(self, other):
        return self._data == (other._data if isinstance(other, ArkoudaStringArray) else other)

    def __repr__(self):
        return f"ArkoudaStringArray({self._data})"

    def factorize(self, *, sort=False, use_na_sentinel=True, **kwargs):
        import numpy as np
        import pandas as pd

        codes, uniques = pd.factorize(
            np.asarray(self.to_numpy()),
            sort=sort,
            use_na_sentinel=use_na_sentinel,
        )
        return codes, uniques
