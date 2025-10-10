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

>>> import numpy as np
>>> idx = np.array([0, -1, 2])
>>> a.take(idx, allow_fill=True, fill_value=99).to_numpy()
array([ 1, 99,  3])

"""

from typing import Optional, Union

import numpy as np
from pandas.api.extensions import ExtensionArray

from arkouda.numpy.dtypes import all_scalars
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.pdarraysetops import concatenate as ak_concat


__all__ = ["_ensure_numpy", "ArkoudaBaseArray"]


def _ensure_numpy(x):
    if hasattr(x, "to_ndarray"):
        return x.to_ndarray()
    return np.asarray(x)


class ArkoudaBaseArray(ExtensionArray):
    default_fill_value: Optional[Union[all_scalars, str]] = -1

    def __init__(self, data):
        # Subclasses should ensure this is the correct ak object
        self._data = data

    def __len__(self):
        """
        Return the number of elements in the array.

        Returns
        -------
        int
            The length of the underlying Arkouda array.

        Notes
        -----
        This method delegates to ``len(self._data)``, which uses the length
        protocol defined by the underlying Arkouda object (e.g., ``pdarray`` or
        ``Strings``). Equivalent to querying ``self._data.size``.
        """
        return len(self._data)

    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate a sequence of same-type extension arrays into a new instance.

        Parameters
        ----------
        to_concat : Sequence[cls]
            Iterable of arrays of the same subclass.

        Returns
        -------
        cls
            New array whose data are the concatenation of all inputs.

        Raises
        ------
        ValueError
            If `to_concat` is empty, contains mixed types, or has incompatible dtypes.
        """
        seq = list(to_concat)
        if not seq:
            raise ValueError("to_concat must contain at least one array")
        if not all(isinstance(x, cls) for x in seq):
            raise ValueError("All items passed to _concat_same_type must be instances of the same class")

        data = [x._data for x in seq]
        out = ak_concat(data)
        return cls(out)

    def _fill_missing(self, mask, fill_value):
        raise NotImplementedError("Subclasses must implement _fill_missing")

    def take(self, indexer, fill_value=None, allow_fill=False):
        """
        Take elements by (0-based) position, returning a new array.

        This implementation:
          * normalizes the indexer to Arkouda int64,
          * explicitly emulates NumPy-style negative wrapping when allow_fill=False,
          * If ``allow_fill=True``, then **only** ``-1`` is allowed as a sentinel
            for missing; those positions are filled with ``fill_value``. Any other
            negative index raises ``ValueError``.
          * validates bounds (raising IndexError) when allow_fill=True,
          * gathers once, then fills masked positions in a single pass.
        """
        import arkouda as ak
        from arkouda.numpy.pdarrayclass import pdarray

        # Normalize indexer to ak int64
        if not isinstance(indexer, pdarray):
            indexer = ak_array(np.asarray(indexer, dtype="int64"))
        elif indexer.dtype != "int64":
            indexer = indexer.astype("int64")

        n = len(self)

        # Trivial fast-path: empty indexer → empty result of same dtype
        if indexer.size == 0:
            return type(self)(self._data[indexer])

        if not allow_fill:
            # Explicit NumPy-like negative wrapping
            # TODO:  Remove negative indexing work-around when #4878 is resolved.
            neg = indexer < 0
            # Wrap negatives: idx = idx + n (vectorized) where needed
            idx_fix = ak.where(neg, indexer + n, indexer)

            # Bounds check (pandas/NumPy behavior): any remaining OOB → IndexError
            oob = (idx_fix < 0) | (idx_fix >= n)
            if oob.any():
                raise IndexError("indexer out of bounds in take")

            return type(self)(self._data[idx_fix])

        # allow_fill=True

        # Only -1 is allowed as a sentinel for missing
        if (indexer < -1).any():
            raise ValueError("Invalid negative indexer when allow_fill=True")

        # Resolve fill_value
        if fill_value is None:
            fill_value = self.default_fill_value

        # cast once to ensure dtype match
        fv = self._data.dtype.type(fill_value)

        # Mask missing, replace -1 with 0 for a safe gather (any valid in-bounds dummy)
        mask = indexer == -1
        idx_fix = ak.where(mask, 0, indexer)

        # Bounds check for the non-missing positions
        oob = ((idx_fix < 0) | (idx_fix >= n)) & (~mask)
        if oob.any():
            raise IndexError("indexer out of bounds in take with allow_fill=True")

        gathered = ak.where(mask, fv, self._data[idx_fix])
        return type(self)(gathered)

    def to_numpy(self, dtype=None, copy=False, na_value=None):
        """
        Convert the array to a NumPy ndarray.

        Parameters
        ----------
        dtype : str, numpy.dtype, optional
            Desired dtype for the result. If None, the underlying dtype is preserved.
        copy : bool, default False
            Whether to ensure a copy is made:
            - If False, a view of the underlying buffer may be returned when possible.
            - If True, always return a new NumPy array.

        Returns
        -------
        numpy.ndarray
            NumPy array representation of the data.
        """
        out = self._data.to_ndarray()

        if dtype is not None and out.dtype != np.dtype(dtype):
            out = out.astype(dtype, copy=False)

        if copy:
            out = out.copy()

        return out

    def to_ndarray(self) -> np.ndarray:
        """
        Convert to a NumPy ndarray, without any dtype conversion or copy options.

        Returns
        -------
        numpy.ndarray
            A new NumPy array materialized from the underlying Arkouda data.

        Notes
        -----
        This is a lightweight convenience wrapper around the backend's
        ``.to_ndarray()`` method. Unlike :meth:`to_numpy`, this method does
        not accept ``dtype`` or ``copy`` arguments and always performs a
        materialization step.
        """
        return self._data.to_ndarray()

    def broadcast_arrays(self, *arrays):
        raise NotImplementedError("ArkoudaBaseArray.broadcast_arrays is not implemented in Arkouda yet")

    def broadcast_to(self, x, shape, /):
        raise NotImplementedError("ArkoudaBaseArray.broadcast_to is not implemented in Arkouda yet")

    def concat(self, arrays, /, *, axis=0):
        raise NotImplementedError("ArkoudaBaseArray.concat is not implemented in Arkouda yet")

    def duplicated(self, arrays, /, *, axis=0):
        raise NotImplementedError("ArkoudaBaseArray.duplicated is not implemented in Arkouda yet")

    def expand_dims(self, x, /, *, axis):
        raise NotImplementedError("ArkoudaBaseArray.expand_dims is not implemented in Arkouda yet")

    def permute_dims(self, x, /, axes):
        raise NotImplementedError("ArkoudaBaseArray.permute_dims is not implemented in Arkouda yet")

    def reshape(self, x, /, shape):
        raise NotImplementedError("ArkoudaBaseArray.reshape is not implemented in Arkouda yet")

    def split(self, x, indices_or_sections, /, *, axis=0):
        raise NotImplementedError("ArkoudaBaseArray.split is not implemented in Arkouda yet")

    def squeeze(self, x, /, *, axis=None):
        raise NotImplementedError("ArkoudaBaseArray.squeeze is not implemented in Arkouda yet")

    def stack(self, arrays, /, *, axis=0):
        raise NotImplementedError("ArkoudaBaseArray.stack is not implemented in Arkouda yet")

    def __hash__(self):
        raise NotImplementedError(
            "__hash__ is not yet implemented for ArkoudaBaseArray. "
            "Use .to_numpy() and hash the result if needed."
        )

    def argmax(self, axis=None, out=None):
        raise NotImplementedError("argmax is not yet implemented for ArkoudaBaseArray.")

    def argmin(self, axis=None, out=None):
        raise NotImplementedError("argmin is not yet implemented for ArkoudaBaseArray.")

    def _mode(self, dropna=True):
        raise NotImplementedError("_mode is not yet implemented for ArkoudaBaseArray.")

    def _quantile(self, q, interpolation="linear"):
        raise NotImplementedError("_quantile is not yet implemented for ArkoudaBaseArray.")
