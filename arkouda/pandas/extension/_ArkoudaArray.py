import numpy as np
from pandas.api.extensions import ExtensionArray, ExtensionDtype

import arkouda as ak
from arkouda.numpy.pdarraysetops import concatenate as ak_concat
from arkouda.pdarraycreation import array

from ._base_array import ArkoudaBaseArray

__all__ = ["ArkoudaArray", "ArkoudaDtype"]


class ArkoudaDtype(ExtensionDtype):
    # implement required properties/methods here
    name = "arkouda"
    type = object  # or the underlying Python type, like int/str
    kind = "O"

    @classmethod
    def construct_array_type(cls):
        from ._ArkoudaArray import ArkoudaArray

        return ArkoudaArray

    @property
    def na_value(self):
        return -1  # or np.nan or None depending on your use case


class ArkoudaArray(ArkoudaBaseArray, ExtensionArray):
    default_fill_value = -1

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            data = ak.array(data)
        if not isinstance(data, ak.pdarray):
            raise TypeError("Expected an Arkouda pdarray")
        self._data = data

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(ak.array(scalars))

    def __getitem__(self, key):
        # Convert numpy boolean mask to arkouda pdarray
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                key = ak.array(key)
            elif key.dtype.kind in {"i", "u"}:
                key = ak.array(key.astype("int64"))
            else:
                raise TypeError(f"Unsupported numpy index type {key.dtype}")

        result = self._data[key]
        if np.isscalar(key):
            if hasattr(result, "to_ndarray"):
                return result.to_ndarray()[()]
            else:
                return result
        return self.__class__(result)

    def __setitem__(self, key, value):
        from arkouda.numpy.dtypes import isSupportedInt

        # Convert numpy mask to pdarray if necessary
        if isinstance(key, np.ndarray) and key.dtype == bool:
            key = array(key)
        elif isinstance(key, np.ndarray) and isSupportedInt(key.dtype):
            key = array(key)
        if isinstance(value, ArkoudaArray):
            value = value._data
        elif isinstance(value, ak.pdarray):
            pass
        elif isinstance(value, (int, float, bool)):  # Add scalar check
            value = ak.full(1, value, dtype=self._data.dtype)
            self._data[key] = value[0]  # assign scalar to scalar position
            return
        else:
            value = ak.array(value)

        self._data[key] = value

    def __len__(self):
        return self._data.size

    def isna(self):
        return ak.zeros(self._data.size, dtype=ak.bool)

    def copy(self):
        return ArkoudaArray(self._data[:])

    @property
    def dtype(self):
        return ArkoudaDtype()

    @property
    def nbytes(self):
        return self._data.nbytes

    def to_numpy(self, dtype=None, copy=False, na_value=np.nan):
        return self._data.to_ndarray()

    def astype(self, dtype, copy=True):
        if isinstance(dtype, ArkoudaDtype):
            return self if not copy else self.copy()
        return self.to_numpy().astype(dtype)

    def equals(self, other):
        if not isinstance(other, ArkoudaArray):
            return False
        return ak.all(self._data == other._data)

    def argsort(self, ascending=True, kind="quicksort", na_position="last"):
        if na_position != "last":
            raise NotImplementedError("na_position != 'last' not supported")
        perm = ak.argsort(self._data)
        if not ascending:
            perm = perm[::-1]
        return perm

    def _reduce(self, name, skipna=True, **kwargs):
        if name == "all":
            return ak.all(self._data)
        elif name == "any":
            return ak.any(self._data)
        elif name == "sum":
            return ak.sum(self._data)
        elif name == "prod":
            return ak.prod(self._data)
        elif name == "min":
            return ak.min(self._data)
        elif name == "max":
            return ak.max(self._data)
        else:
            raise TypeError(f"'ArkoudaArray' with dtype arkouda does not support reduction '{name}'")

    @classmethod
    def _concat_same_type(cls, to_concat):
        # Concatenate the internal _data fields of all arrays
        data = ak_concat([x._data for x in to_concat])
        return cls(data)

    def __eq__(self, other):
        if isinstance(other, ArkoudaArray):
            return self._data == other._data
        return self._data == other

    def __repr__(self):
        return f"ArkoudaArray({self._data})"

    @classmethod
    def _from_factorized(cls, values, original):
        # This assumes factorization didn't alter values in a way we need to recover.
        # You can refine it later if you want real deduplication + recoding.
        return cls(ak.array(values))

    def factorize(self, *, sort=False, use_na_sentinel=True, **kwargs):
        import numpy as np
        import pandas as pd

        codes, uniques = pd.factorize(
            np.asarray(self.to_numpy()),
            sort=sort,
            use_na_sentinel=use_na_sentinel,
        )
        return codes, uniques

    def _fill_missing(self, mask, fill_value):
        arr = self.to_ndarray()

        np_mask = mask.to_ndarray() if hasattr(mask, "to_ndarray") else mask

        # Ensure arr is the same shape as mask
        if np_mask.shape[0] != arr.shape[0]:
            arr = np.resize(arr, np_mask.shape)

        out = arr.copy()
        out[np_mask] = fill_value
        return out
