from typing import Any

import numpy as np
from numpy import ndarray
from pandas.api.extensions import ExtensionArray

from arkouda.numpy.dtypes import dtype as ak_dtype
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.pdarraycreation import full as ak_full
from arkouda.numpy.pdarraycreation import pdarray

from ._arkouda_base_array import ArkoudaBaseArray
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


class ArkoudaArray(ArkoudaBaseArray, ExtensionArray):
    default_fill_value = -1

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            data = ak_array(data)
        if not isinstance(data, pdarray):
            raise TypeError("Expected an Arkouda pdarray")
        self._data = data

    # in _arkouda_array.py

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        # If pandas passes our own EA dtype, ignore it and infer from data
        if isinstance(dtype, _ArkoudaBaseDtype):
            dtype = dtype.numpy_dtype
        # If scalars is already a numpy array, we can preserve its dtype
        return cls(ak_array(scalars, dtype=dtype, copy=copy))

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
            key = ak_array(key)
        elif isinstance(key, np.ndarray) and isSupportedInt(key.dtype):
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

    def __len__(self):
        return int(self._data.size)

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

    #   TODO:  use pdarray.copy()
    def copy(self):
        return ArkoudaArray(self._data[:])

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

    def to_numpy(self, dtype=None, copy=False, na_value=np.nan):
        return self._data.to_ndarray()
        # return ak_array(self._data)

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

    def factorize(self, *, sort=False, use_na_sentinel=True, **kwargs):
        import numpy as np
        import pandas as pd

        codes, uniques = pd.factorize(
            np.asarray(self.to_numpy()),
            sort=sort,
            use_na_sentinel=use_na_sentinel,
        )
        return codes, uniques
