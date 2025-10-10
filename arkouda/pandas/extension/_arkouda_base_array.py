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

from typing import Optional, Tuple

import numpy as np
from pandas.api.extensions import ExtensionArray

from arkouda.numpy.dtypes import all_scalars
from arkouda.numpy.pdarrayclass import pdarray
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.pdarraysetops import concatenate as ak_concat
from arkouda.numpy.strings import Strings
from arkouda.pandas.categorical import Categorical


__all__ = ["_ensure_numpy", "ArkoudaBaseArray"]


def _ensure_numpy(x):
    if hasattr(x, "to_ndarray"):
        return x.to_ndarray()
    return np.asarray(x)


class ArkoudaBaseArray(ExtensionArray):
    default_fill_value: Optional[all_scalars] = -1

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

    def _fill_missing(self, mask, fill_value):
        raise NotImplementedError("Subclasses must implement _fill_missing")

    def take(self, indexer, fill_value=None, allow_fill=False):
        """
        Take elements by (0-based) position, returning a new array.

        This follows the pandas ``ExtensionArray.take`` semantics:

        * If ``allow_fill=False`` (default), negative indices wrap from the end
          (NumPy-style indexing).
        * If ``allow_fill=True``, then **only** ``-1`` is allowed as a sentinel
          for missing; those positions are filled with ``fill_value``. Any other
          negative index raises ``ValueError``.

        Parameters
        ----------
        indexer : Union[Sequence, ndarray, pdarray]
            Positions to take. May be a Python sequence, a NumPy integer array,
            or an Arkouda ``pdarray``. Internally normalized to ``int64``.
        fill_value : all_scalars
            Value used to fill positions where ``indexer == -1`` when
            ``allow_fill=True``. If ``None``, uses ``self.default_fill_value``.
            The value is cast to the underlying dtype.  Optional.
        allow_fill : bool
            If ``False`` (default), negative indices are interpreted as counting from the
            end. If ``True``, only ``-1`` is permitted as a sentinel for missing,
            and other negative values are invalid.

        Returns
        -------
        self.__class__
            A new array of the same logical dtype with length ``len(indexer)``.

        Raises
        ------
        ValueError
            If ``allow_fill=True`` and ``indexer`` contains values less than ``-1``.

        Notes
        -----
        ``indexer`` is coerced to an Arkouda ``int64`` ``pdarray``. When
        ``allow_fill=True``, masked positions (where ``indexer == -1``) are
        filled with ``fill_value`` (cast via ``self._data.dtype.type``), and all
        other positions are gathered from ``self._data`` in a single pass.

        See Also
        --------
        pandas.api.extensions.ExtensionArray.take : Reference semantics in pandas.

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaArray
        >>> arr = ArkoudaArray(ak.array([10, 20, 30, 40]))

        Basic take:
        >>> arr.take([0, 2, 3]).to_numpy().tolist()
        [10, 30, 40]

        Use ``-1`` as a sentinel with ``allow_fill=True``:
        >>> arr.take([0, -1, 2, -1], allow_fill=True, fill_value=-999).to_numpy().tolist()
        [10, -999, 30, -999]

        When ``fill_value`` is ``None``, the class default is used:
        >>> arr.default_fill_value
        -1
        >>> arr.take([1, -1, 3], allow_fill=True).to_numpy().tolist()
        [20, -1, 40]

        """
        import numpy as np

        import arkouda as ak
        from arkouda.numpy.pdarrayclass import pdarray

        # Normalize indexer to ak int64
        if not isinstance(indexer, pdarray):
            indexer = ak_array(np.asarray(indexer, dtype="int64"))
        elif indexer.dtype != "int64":
            indexer = indexer.astype("int64")

        if not allow_fill:
            return type(self)(self._data[indexer])

        # allow_fill=True
        if (indexer < -1).any():
            raise ValueError("Invalid negative indexer when allow_fill=True")

        if fill_value is None:
            fill_value = self.default_fill_value

        # cast once to ensure dtype match
        fv = self._data.dtype.type(fill_value)

        mask = indexer == -1
        idx_fix = ak.where(mask, 0, indexer)

        gathered = ak.where(mask, fv, self._data[idx_fix])
        return type(self)(gathered)

    def factorize(
        self, use_na_sentinel=True, sort=False, **kwargs
    ) -> Tuple[pdarray, "ArkoudaBaseArray"]:
        """
        Encode the values of this array as integer codes and uniques,
        similar to :func:`pandas.factorize`, but implemented with Arkouda.

        Each distinct non-missing value is assigned a unique integer code.
        Missing values (NaN in floating dtypes) are encoded as -1 by default.

        Parameters
        ----------
        use_na_sentinel : bool, default True
            If True, missing values are encoded as -1 in the codes array.
            If False, missing values are assigned a valid code equal to
            ``len(uniques)``.
        sort : bool, default False
            Whether to sort the unique values. If False, the unique values
            appear in the order of first appearance in the array. If True,
            the unique values are sorted, and codes are assigned accordingly.
        **kwargs
            Ignored for compatibility.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A pair ``(codes, uniques)`` where:
            - ``codes`` is a NumPy ``int64`` array of factor labels, one per element.
              Missing values are ``-1`` if ``use_na_sentinel=True``; otherwise they
              receive the code ``len(uniques)``.
            - ``uniques`` is a NumPy array of the unique values.

        Notes
        -----
        * Only floating-point dtypes treat ``NaN`` as missing; for other dtypes,
          no values are considered missing.
        * This method executes all grouping and factorization in Arkouda,
          returning results as NumPy arrays for compatibility with pandas.
        * Unlike pandas, string/None/null handling is not yet unified.

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaArray
        >>> arr = ArkoudaArray(ak.array([1, 2, 1, 3]))
        >>> codes, uniques = arr.factorize()
        >>> codes
        array([0, 1, 0, 2])
        >>> uniques
        ArkoudaArray([1 2 3])
        """
        from arkouda import isnan as ak_isnan
        from arkouda.numpy.dtypes import bool_, int64
        from arkouda.numpy.pdarraycreation import arange, full, ones, zeros
        from arkouda.numpy.sorting import argsort
        from arkouda.pandas.groupbyclass import GroupBy

        # Arkouda array backing
        arr = self._data
        n = arr.size

        # Only floats treat NaN as NA (pandas-like); others: no NA
        if arr.dtype in ("float32", "float64"):
            non_na = ~ak_isnan(arr)
        else:
            non_na = ones(n, dtype=bool_)

        arr_nn = arr[non_na]
        if arr_nn.size == 0:
            sent = -1 if use_na_sentinel else 0
            return np.full(n, sent, dtype=np.int64), type(self)(
                np.array([], dtype=self.to_numpy().dtype)
            )

        # Group non-missing values
        g = GroupBy(arr_nn)
        uniques_ak = g.unique_keys  # one per group
        if not isinstance(uniques_ak, (pdarray, Strings, Categorical)):
            from arkouda import concatenate

            uniques_ak = concatenate(uniques_ak)

        if sort:
            # Keys already sorted; group id -> 0..k-1
            groupid_to_code = arange(uniques_ak.size, dtype=int64)

            # Work around to account GroupBy not sorting Categorical properly
            if isinstance(arr, Categorical):
                perm = uniques_ak.argsort()
                #   Inverse argsort:
                groupid_to_code[perm] = arange(uniques_ak.size, dtype=int64)
                uniques_ak = uniques_ak[perm]

        else:
            # First-appearance order
            _keys, first_idx_per_group = g.min(arange(arr_nn.size, dtype=int64))
            order = argsort(first_idx_per_group)

            # Reorder uniques by first appearance
            uniques_ak = uniques_ak[order]

            # Map group_id -> code in first-appearance order
            groupid_to_code = zeros(order.size, dtype=int64)
            groupid_to_code[order] = arange(order.size, dtype=int64)

        # Per-element codes on the non-NA slice
        codes_nn = g.broadcast(groupid_to_code)

        # Assemble full codes with sentinel
        sentinel = -1 if use_na_sentinel else uniques_ak.size
        codes_ak = full(n, sentinel, dtype=int64)
        codes_ak[non_na] = codes_nn
        from ._arkouda_array import ArkoudaArray
        return ArkoudaArray(codes_ak), type(self)(uniques_ak)

    # In each EA
    def _values_for_factorize(self):
        # Return a small NumPy "codes-like" view + na_value
        # If you can't return codes here, still return a compact NumPy representation.
        return self.to_factorize_view(), np.nan  # both NumPy

    @classmethod
    def _from_factorized(cls, values, original):
        # Build EA back from factorized NumPy values
        return cls._from_numpy(values)

    def to_numpy(self, dtype=None, copy=False):
        out = self._data.to_ndarray()
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    def to_ndarray(self):
        return self._data.to_ndarray()


    def argsort(
            self,
            *,
            ascending: bool = True,
            kind = "quicksort",
            na_position: str = "last",
            **kwargs,
    )  :#-> np.ndarray:
        """
        Return NumPy indices that would sort the array.
        Pandas calls this with no args and expects a numpy.intp/int64 array.
        """
        # NOTE: our numeric/bool arrays have no NA; float NaNs are left to server implementation.
        perm = self._data.argsort(ascending=ascending)
        return perm
