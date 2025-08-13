"""
Base extension array infrastructure for Arkouda-backed pandas objects.

This module defines :class:`ArkoudaBaseArray`, an abstract base class implementing
common logic for pandas ``ExtensionArray`` subclasses that store their data in Arkouda
server-side arrays. It provides core methods like construction from sequences,
concatenation, conversion to NumPy, and a generalized ``take`` implementation
supporting missing-value fills.

Classes in this module are not intended to be instantiated directly—subclasses should
wrap specific Arkouda types such as numeric arrays, strings, or categoricals, and
must ensure that the internal ``_data`` attribute references the correct Arkouda
array type.

Classes
-------
ArkoudaBaseArray(ExtensionArray)
    Base class for Arkouda-backed pandas extension arrays, implementing shared
    behaviors and bridging pandas’ extension array API with Arkouda server arrays.

Notes
-----
This module is designed to integrate with pandas' extension array interface. It
is used internally by Arkouda's pandas extension types and not generally part of
the public user-facing API. Subclasses must implement certain abstract or
datatype-specific methods, such as ``_fill_missing``.

Examples
--------
>>> import arkouda as ak
>>> from arkouda.pandas.extension._arkouda_base_array import ArkoudaBaseArray
>>> arr = ak.array([1, 2, 3])
>>> class MyArray(ArkoudaBaseArray): pass
>>> a = MyArray(arr)
>>> len(a)
3
>>> a.to_numpy()
array([1, 2, 3])

The ``take`` method supports missing values via ``allow_fill=True``:

>>> import numpy as np
>>> idx = np.array([0, -1, 2])
>>> a.take(idx, allow_fill=True, fill_value=99).to_numpy()
array([ 1, 99,  3])

"""

import numpy as np
from pandas.api.extensions import ExtensionArray

from arkouda.numpy.pdarrayclass import pdarray
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.pdarraysetops import concatenate as ak_concat
from arkouda.numpy.strings import Strings

__all__ = ["_ensure_numpy", "ArkoudaBaseArray"]


def _ensure_numpy(x):
    if hasattr(x, "to_ndarray"):
        return x.to_ndarray()
    return np.asarray(x)


class ArkoudaBaseArray(ExtensionArray):
    def __init__(self, data):
        # Subclasses should ensure this is the correct ak object
        self._data = data

    def __len__(self):
        return int(self._data.size)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        from arkouda.numpy.pdarraycreation import array as ak_array

        # If caller already passed an arkouda array, honor it
        if isinstance(scalars, pdarray) or isinstance(scalars, Strings):
            return cls(scalars.copy() if copy else scalars)
        # Else build in arkouda
        if dtype is None:
            return cls(ak_array(scalars))

        return cls(ak_array(scalars, dtype=dtype, copy=copy))

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls(ak_concat([x._data for x in to_concat]))

    def take(self, indexer, fill_value=None, allow_fill=False, axis=None):
        import numpy as np

        import arkouda as ak
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.strings import Strings

        # normalize indexer to ak int64
        if isinstance(indexer, pdarray):
            idx_ak = indexer.astype("int64")
        else:
            idx_ak = ak.array(np.asarray(indexer, dtype="int64"))

        if not allow_fill:
            gathered = self._data[idx_ak]
            return type(self)(gathered)

        # allow_fill=True
        mask = idx_ak == -1
        idx_fix = ak.where(mask, 0, idx_ak)  # valid placeholder

        # server-side gather
        gathered = self._data[idx_fix]

        # choose default fill if needed
        if fill_value is None:
            # rely on EA dtype name if you exposed it; fall back to ak dtype
            from arkouda.dtypes import dtype as ak_dtype

            dtype = ak_dtype(self._data.dtype)
            if dtype == "str_":
                fill_value = ""
            elif dtype in ["int64", "uint64"]:
                fill_value = -1
            elif dtype == "bool_":
                fill_value = False
            else:
                # safest: require explicit fill for weird/non-numeric types
                raise ValueError("Specify fill_value explicitly for this dtype when allow_fill=True")

        # Categorical: pandas returns strings when fill may not be in categories
        try:
            from arkouda.pandas.extension._arkouda_categorical_array import (
                ArkoudaCategoricalArray,
            )
        except Exception:
            ArkoudaCategoricalArray = ()  # not available yet

        if isinstance(self, ArkoudaCategoricalArray):
            # Convert to strings for mixing category + fill
            gathered_str = self._data.to_strings()[idx_fix]  # subset as strings
            if not isinstance(fill_value, str):
                fill_value = str(fill_value)
            fill_vec = ak.full(mask.size, fill_value, dtype="str_")
            out = ak.where(mask, fill_vec, gathered_str)
            return ArkoudaCategoricalArray(ak.Categorical(out))

        # Strings: use string fill
        if isinstance(self._data, Strings):
            if not isinstance(fill_value, str):
                fill_value = str(fill_value)
            fill_vec = ak.full(mask.size, fill_value, dtype="str_")
            out = ak.where(mask, fill_vec, gathered)
            return type(self)(out)

        # Numeric/bool: scalar fill is fine
        # Make sure the fill is castable to the underlying dtype
        from arkouda.numpy.dtypes import can_cast

        fv = (
            ak.cast(ak_array([fill_value]), dt=self._data.dtype)[0]
            if can_cast(fill_value, self._data.dtype)
            else fill_value
        )
        out = ak.where(
            mask, ak.full(mask.size, fv, dtype=str(getattr(self._data, "dtype", "int64"))), gathered
        )
        return type(self)(out)

    def _fill_missing(self, mask, fill_value):
        raise NotImplementedError("Subclasses must implement _fill_missing")

    def to_numpy(self, dtype=None, copy=False):
        out = self._data.to_ndarray()
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    def to_ndarray(self):
        return self._data.to_ndarray()
