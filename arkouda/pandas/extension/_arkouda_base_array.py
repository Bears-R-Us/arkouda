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
from arkouda.numpy.pdarrayclass import pdarray
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
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        Construct an instance from a 1D sequence of scalars or an Arkouda object.

        Parameters
        ----------
        scalars : Sequence | pdarray | Strings | Categorical | pandas.Series | pandas.Index | scalar
            Input data. Inputs are converted via ``ak.array``.
        dtype : optional
            Target dtype for the underlying Arkouda array. If ``scalars`` is already
            an Arkouda object with a different dtype, a cast is performed.
        copy : bool, default False
            If True and ``scalars`` is an Arkouda object, make a physical copy
            before wrapping. For non-Arkouda inputs, this is forwarded to ``ak.array``.

        Returns
        -------
        cls
            An instance of the subclass wrapping an Arkouda array.
        """
        import pandas as pd

        from arkouda.pandas.categorical import Categorical

        if isinstance(scalars, (pd.Series, pd.Index)):
            scalars = scalars.to_numpy(copy=False)

        if np.isscalar(scalars):
            scalars = [scalars]

        if isinstance(scalars, Categorical):
            return cls(scalars.copy() if copy is True else scalars)

        return cls(ak_array(scalars, dtype=dtype, copy=copy))

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

        See module docstring for full semantics. This implementation:
          * normalizes the indexer to Arkouda int64,
          * explicitly emulates NumPy-style negative wrapping when allow_fill=False,
          * validates bounds (raising IndexError) when allow_fill=True,
          * gathers once, then fills masked positions in a single pass.
        """
        import numpy as np

        import arkouda as ak
        from arkouda.numpy.pdarrayclass import pdarray as ak_pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array

        # --- Normalize indexer to ak int64 ------------------------------------------------------------
        if not isinstance(indexer, ak_pdarray):
            indexer = ak_array(np.asarray(indexer, dtype="int64"))
        elif indexer.dtype != "int64":
            indexer = indexer.astype("int64")

        n = len(self)

        # Trivial fast-path: empty indexer → empty result of same dtype
        if indexer.size == 0:
            return type(self)(self._data[indexer])

        if not allow_fill:
            # Explicit NumPy-like negative wrapping
            neg = indexer < 0
            # Wrap negatives: idx = idx + n (vectorized) where needed
            idx = ak.where(neg, indexer + n, indexer)

            # Bounds check (pandas/NumPy behavior): any remaining OOB → IndexError
            oob = (idx < 0) | (idx >= n)
            if oob.any():
                raise IndexError("indexer out of bounds in take")

            return type(self)(self._data[idx])

        # --- allow_fill=True --------------------------------------------------------------------------
        # Only -1 is allowed as a sentinel for missing
        if (indexer < -1).any():
            raise ValueError("Invalid negative indexer when allow_fill=True")

        # Mask missing, replace -1 with 0 for a safe gather (any valid in-bounds dummy)
        mask = indexer == -1
        idx = ak.where(mask, 0, indexer)

        # Bounds check for the non-missing positions
        oob = ((idx < 0) | (idx >= n)) & (~mask)
        if oob.any():
            raise IndexError("indexer out of bounds in take with allow_fill=True")

        # Gather once
        taken = self._data[idx]

        # Resolve fill_value
        if fill_value is None:
            fill_value = self.default_fill_value

        # Try to cast fill_value to backend dtype when that concept exists
        fv = fill_value
        dt = getattr(self._data, "dtype", None)
        if dt is not None and hasattr(dt, "type"):
            try:
                fv = dt.type(fill_value)
            except Exception:
                # Fall back to original fill_value (e.g., Strings/Categorical cases)
                fv = fill_value

        # Fill masked positions; keep others from gathered data
        out = ak.where(mask, fv, taken)
        return type(self)(out)

    def to_numpy(self, dtype=None, copy=False):
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

    def broadcast_arrays(*arrays):
        raise NotImplementedError("ArkoudaBaseArray.broadcast_arrays is not implemented in Arkouda yet")

    def broadcast_to(x, shape, /):
        raise NotImplementedError("ArkoudaBaseArray.broadcast_to is not implemented in Arkouda yet")

    def concat(arrays, /, *, axis=0):
        raise NotImplementedError("ArkoudaBaseArray.concat is not implemented in Arkouda yet")

    def duplicated(arrays, /, *, axis=0):
        raise NotImplementedError("ArkoudaBaseArray.duplicated is not implemented in Arkouda yet")

    def expand_dims(x, /, *, axis):
        raise NotImplementedError("ArkoudaBaseArray.expand_dims is not implemented in Arkouda yet")

    def permute_dims(x, /, axes):
        raise NotImplementedError("ArkoudaBaseArray.permute_dims is not implemented in Arkouda yet")

    def reshape(x, /, shape):
        raise NotImplementedError("ArkoudaBaseArray.reshape is not implemented in Arkouda yet")

    def split(x, indices_or_sections, /, *, axis=0):
        raise NotImplementedError("ArkoudaBaseArray.split is not implemented in Arkouda yet")

    def squeeze(x, /, *, axis=None):
        raise NotImplementedError("ArkoudaBaseArray.squeeze is not implemented in Arkouda yet")

    def stack(arrays, /, *, axis=0):
        raise NotImplementedError("ArkoudaBaseArray.stack is not implemented in Arkouda yet")
