import numpy as np
from pandas.api.extensions import ExtensionArray, ExtensionDtype

from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.pdarraycreation import full as ak_full
from arkouda.numpy.pdarraycreation import pdarray
from arkouda.numpy.pdarraycreation import zeros as ak_zeros
from arkouda.numpy.pdarraysetops import concatenate as ak_concat
from arkouda.pdarraycreation import array

from ._arkouda_base_array import ArkoudaBaseArray

__all__ = ["ArkoudaArray", "ArkoudaDtype"]


class ArkoudaDtype(ExtensionDtype):
    # implement required properties/methods here
    name = "arkouda"
    type = object  # or the underlying Python type, like int/str
    kind = "O"

    @classmethod
    def construct_array_type(cls):
        from ._arkouda_array import ArkoudaArray

        return ArkoudaArray

    @property
    def na_value(self):
        return -1  # or np.nan or None depending on your use case


class ArkoudaArray(ArkoudaBaseArray, ExtensionArray):
    default_fill_value = -1

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            data = ak_array(data)
        if not isinstance(data, pdarray):
            raise TypeError("Expected an Arkouda pdarray")
        self._data = data

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(ak_array(scalars))

    def __getitem__(self, key):
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
        from arkouda.numpy.dtypes import isSupportedInt

        # Convert numpy mask to pdarray if necessary
        if isinstance(key, np.ndarray) and key.dtype == bool:
            key = array(key)
        elif isinstance(key, np.ndarray) and isSupportedInt(key.dtype):
            key = array(key)
        if isinstance(value, ArkoudaArray):
            value = value._data
        elif isinstance(value, pdarray):
            pass
        elif isinstance(value, (int, float, bool)):  # Add scalar check
            value = ak_full(1, value, dtype=self._data.dtype)
            self._data[key] = value[0]  # assign scalar to scalar position
            return
        else:
            value = ak_array(value)

        self._data[key] = value

    def __len__(self):
        return self._data.size

    #   TODO:  Fix isna
    def isna(self):
        return ak_zeros(self._data.size, dtype=bool)

    #   TODO:  use pdarray.copy()
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

    #   TODO:  fix this
    def astype(self, dtype, copy=True):
        if isinstance(dtype, ArkoudaDtype):
            return self if not copy else self.copy()
        return self.to_numpy().astype(dtype)

    def equals(self, other):
        if not isinstance(other, ArkoudaArray):
            return False
        return self._data.equals(other._data)

    #   TODO:  add pandas arguments
    def argsort(self, ascending=True):
        perm = self._data.argsort(ascending=ascending)
        return perm

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

    #   TODO:  refine this.
    @classmethod
    def _from_factorized(cls, values, original):
        # This assumes factorization didn't alter values in a way we need to recover.
        # You can refine it later if you want real deduplication + recoding.
        return cls(array(values))

    def factorize(self, *, sort=False, use_na_sentinel=True, **kwargs):
        import numpy as np
        import pandas as pd

        codes, uniques = pd.factorize(
            np.asarray(self.to_numpy()),
            sort=sort,
            use_na_sentinel=use_na_sentinel,
        )
        return codes, uniques
