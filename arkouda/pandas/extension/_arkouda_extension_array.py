"""
Base extension array infrastructure for Arkouda-backed pandas objects.

This module defines :class:`ArkoudaExtensionArray`, an abstract base class implementing
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
ArkoudaExtensionArray(ExtensionArray)
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
>>> from arkouda.pandas.extension import ArkoudaExtensionArray
>>> arr = ak.array([1, 2, 3])
>>> class MyArray(ArkoudaExtensionArray): pass
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

from __future__ import annotations

from types import NotImplementedType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Tuple, TypeVar, Union

import numpy as np

from numpy.typing import NDArray
from pandas.api.extensions import ExtensionArray
from pandas.core.arraylike import OpsMixin
from typing_extensions import Self

from arkouda.numpy.dtypes import all_scalars
from arkouda.numpy.pdarrayclass import pdarray
from arkouda.numpy.pdarraysetops import concatenate as ak_concat
from arkouda.pandas.categorical import Categorical


if TYPE_CHECKING:
    from arkouda.numpy.strings import Strings
else:
    Strings = TypeVar("Strings")

__all__ = ["_ensure_numpy", "ArkoudaExtensionArray"]


def _ensure_numpy(x):
    if hasattr(x, "to_ndarray"):
        return x.to_ndarray()
    return np.asarray(x)


class ArkoudaExtensionArray(OpsMixin, ExtensionArray):
    default_fill_value: Optional[Union[all_scalars, str]] = -1

    _data: Any

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
    def _from_data(cls: type[Self], data: Any) -> Self:
        return cls(data)

    def _arith_method(
        self,
        other: object,
        op: Callable[[Any, Any], Any],
    ) -> Union[Self, NotImplementedType]:
        """
        Apply an elementwise arithmetic operation between this ExtensionArray and
        ``other``.

        This is the pandas ExtensionArray arithmetic hook. Pandas uses this method
        (via its internal operator dispatch) to implement operators like ``+``,
        ``-``, ``*``, etc. for arrays/Series backed by Arkouda.

        Parameters
        ----------
        other : object
            The right-hand operand. Supported forms:

            * ExtensionArray with a ``_data`` attribute: the operand is unwrapped to
              its underlying Arkouda data.
            * scalar: any NumPy scalar / Python scalar supported by the underlying
              Arkouda operation.

            Any other type returns ``NotImplemented`` so that pandas/Python can fall
            back to alternate dispatch paths.
        op : callable
            A binary operator (e.g., ``operator.add``). Must accept
            ``(self._data, other)`` and return an Arkouda-backed result.

        Returns
        -------
        ExtensionArray or NotImplemented
            A new array of the same ExtensionArray class as ``self`` containing the
            elementwise result, or ``NotImplemented`` for unsupported operand types.

        Notes
        -----
        * This method does **not** perform index alignment; pandas handles alignment
          at the Series/DataFrame level before calling into the ExtensionArray.
        * Type coercion / promotion behavior is determined by the underlying Arkouda
          implementation of ``op``.
        """
        from arkouda.numpy.pdarraycreation import array as ak_array

        if isinstance(other, ExtensionArray) and hasattr(other, "_data"):
            other = other._data
            if isinstance(other, (np.ndarray, Iterable, pdarray, Strings)):
                other = ak_array(other, copy=False)
            elif isinstance(other, Categorical):
                other = other.to_strings()
            else:
                return NotImplemented
        elif np.isscalar(other):
            pass
        else:
            return NotImplemented

        result = op(self._data, other)
        return self._from_data(result)

    def _normalize_setitem_key(self, key):
        import numpy as np

        from arkouda.numpy.dtypes import is_supported_int
        from arkouda.numpy.pdarraycreation import array as ak_array

        if isinstance(key, np.ndarray) and key.dtype == bool:
            return ak_array(key)
        if isinstance(key, np.ndarray) and is_supported_int(key.dtype):
            return ak_array(key)
        return key

    def copy(self, deep: bool = True):
        """
        Return a copy of the array.

        Parameters
        ----------
        deep : bool, default True
            Whether to make a deep copy of the underlying Arkouda data.
            - If ``True``, the underlying server-side array is duplicated.
            - If ``False``, a new ExtensionArray wrapper is created but the
              underlying data is shared (no server-side copy).

        Returns
        -------
        ArkoudaExtensionArray
            A new instance of the same concrete subclass containing either a
            deep copy or a shared reference to the underlying data.

        Notes
        -----
        Pandas semantics:
            ``deep=False`` creates a new wrapper but may share memory.
            ``deep=True`` must create an independent copy of the data.

        Arkouda semantics:
            Arkouda arrays do not presently support views. Therefore:
            - ``deep=False`` returns a new wrapper around the *same*
              server-side array.
            - ``deep=True`` forces a full server-side copy.

        Examples
        --------
        Shallow copy (shared data):

        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaArray
        >>> arr = ArkoudaArray(ak.arange(5))
        >>> c1 = arr.copy(deep=False)
        >>> c1
        ArkoudaArray([0 1 2 3 4])

        Underlying data is the same object:

        >>> arr._data is c1._data
        True

        Deep copy (independent server-side data):

        >>> c2 = arr.copy(deep=True)
        >>> c2
        ArkoudaArray([0 1 2 3 4])

        Underlying data is a distinct pdarray on the server:

        >>> arr._data is c2._data
        False
        """
        data = self._data.copy() if deep else self._data
        return type(self)(data)

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

    @classmethod
    def _from_sequence(
        cls,
        scalars,
        dtype=None,
        copy: bool = False,
    ) -> "ArkoudaExtensionArray":
        """
        Construct an Arkouda-backed ExtensionArray from Arkouda objects or
        Python/NumPy scalars.

        This factory inspects ``scalars`` and returns an instance of the
        appropriate concrete subclass:

        * :class:`ArkoudaArray` for :class:`pdarray`
        * :class:`ArkoudaStringArray` for :class:`Strings`
        * :class:`ArkoudaCategoricalArray` for :class:`Categorical`

        If ``scalars`` is **not** already an Arkouda server-side array, it is
        interpreted as a sequence of Python/NumPy scalars, converted into a
        server-side ``pdarray`` via :func:`arkouda.numpy.pdarraycreation.array`,
        and wrapped in :class:`ArkoudaArray`.

        Parameters
        ----------
        scalars : object
            Either an Arkouda array type (``pdarray``, ``Strings``,
            or ``Categorical``) or a sequence of Python/NumPy scalars.
        dtype : object, optional
            Ignored. Present for pandas API compatibility.
        copy : bool, default False
            Ignored. Present for pandas API compatibility.

        Returns
        -------
        ArkoudaExtensionArray
            An instance of :class:`ArkoudaArray`,
            :class:`ArkoudaStringArray`, or
            :class:`ArkoudaCategoricalArray`, depending on the type of
            ``scalars``.

        Examples
        --------
        Constructing from Arkouda server-side arrays:

        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaExtensionArray
        >>> pda = ak.arange(5)
        >>> ea = ArkoudaExtensionArray._from_sequence(pda)
        >>> ea
        ArkoudaArray([0 1 2 3 4])

        From Arkouda Strings:

        >>> s = ak.array(["red", "green", "blue"])
        >>> ea = ArkoudaExtensionArray._from_sequence(s)
        >>> ea
        ArkoudaStringArray(['red', 'green', 'blue'])

        From Python scalars:

        >>> ea = ArkoudaExtensionArray._from_sequence([10, 20, 30])
        >>> ea
        ArkoudaArray([10 20 30])

        From mixed Python/NumPy types:

        >>> import numpy as np
        >>> ea = ArkoudaExtensionArray._from_sequence([1, np.int64(2), 3])
        >>> ea
        ArkoudaArray([1 2 3])
        """
        # Local imports to avoid circular dependencies at module import time.
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array
        from arkouda.numpy.strings import Strings
        from arkouda.pandas.categorical import Categorical
        from arkouda.pandas.extension._arkouda_array import ArkoudaArray
        from arkouda.pandas.extension._arkouda_categorical_array import ArkoudaCategoricalArray
        from arkouda.pandas.extension._arkouda_string_array import ArkoudaStringArray

        # Fast path: already an Arkouda column. Pick the matching subclass.
        if isinstance(scalars, pdarray):
            return ArkoudaArray(scalars)
        if isinstance(scalars, Strings):
            return ArkoudaStringArray(scalars)
        if isinstance(scalars, Categorical):
            return ArkoudaCategoricalArray(scalars)

        # Fallback: treat as a sequence of scalars and build a pdarray.
        data = ak_array(scalars)
        return ArkoudaArray(data)

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
        from arkouda.numpy.pdarraycreation import array as ak_array

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

    def factorize(self, use_na_sentinel=True) -> Tuple[NDArray[np.intp], "ArkoudaExtensionArray"]:
        """
        Encode the values of this array as integer codes and unique values.

        This is similar to :func:`pandas.factorize`, but the grouping/factorization
        work is performed in Arkouda. The returned ``codes`` are a NumPy array for
        pandas compatibility, while ``uniques`` are returned as an ExtensionArray
        of the same type as ``self``.

        Each distinct non-missing value is assigned a unique integer code.
        For floating dtypes, ``NaN`` is treated as missing; for all other dtypes,
        no values are considered missing.

        Parameters
        ----------
        use_na_sentinel : bool, default True
            If True, missing values are encoded as ``-1`` in the returned codes.
            If False, missing values are assigned the code ``len(uniques)``.
            (Missingness is only detected for floating dtypes via ``NaN``.)

        Returns
        -------
        (numpy.ndarray, ExtensionArray)
            A pair ``(codes, uniques)`` where:

            * ``codes`` is a 1D NumPy array of dtype ``np.intp`` with the same length
              as this array, containing the factor codes for each element.
            * ``uniques`` is an ExtensionArray containing the unique (non-missing)
              values, with the same extension type as ``self``.

            If ``use_na_sentinel=True``, missing values in ``codes`` are ``-1``.
            Otherwise they receive the code ``len(uniques)``.

        Notes
        -----
        * Only floating-point dtypes treat ``NaN`` as missing; for other dtypes,
          all values are treated as non-missing.
        * ``uniques`` are constructed from Arkouda's unique keys and returned as
          ``type(self)(uniques_ak)`` so that pandas internals (e.g. ``groupby``)
          can treat them as an ExtensionArray.
        * String/None/null missing-value behavior is not yet unified with pandas.

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
        from arkouda.numpy.dtypes import ARKOUDA_SUPPORTED_FLOATS, bool_, int64
        from arkouda.numpy.numeric import isnan as ak_isnan
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import arange, full, ones, zeros
        from arkouda.numpy.pdarraycreation import array as ak_array
        from arkouda.numpy.sorting import argsort
        from arkouda.numpy.strings import Strings
        from arkouda.pandas.groupbyclass import GroupBy

        # Arkouda array backing
        arr = self._data
        n = arr.size

        # Only floats treat NaN as NA (pandas-like); others: no NA
        if arr.dtype in ARKOUDA_SUPPORTED_FLOATS:
            non_na = ~ak_isnan(arr)
        else:
            non_na = ones(n, dtype=bool_)

        arr_nn = arr[non_na]
        if arr_nn.size == 0:
            sent = -1 if use_na_sentinel else 0
            from arkouda.numpy.pdarraycreation import full as ak_full

            return ak_full(n, sent, dtype=int64).to_ndarray().astype(np.intp, copy=False), type(self)(
                ak_array([], dtype=self.to_numpy().dtype)
            )

        # Group non-missing values
        g = GroupBy(arr_nn)
        uniques_ak = g.unique_keys  # one per group
        if not isinstance(uniques_ak, (pdarray, Strings, Categorical)):
            from arkouda import concatenate

            uniques_ak = concatenate(uniques_ak)

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

        codes_np = codes_ak.to_ndarray().astype(np.intp, copy=False)

        return codes_np, type(self)(uniques_ak)

    # In each EA
    def _values_for_factorize(self):
        # Return a small NumPy "codes-like" view + na_value
        # If you can't return codes here, still return a compact NumPy representation.
        return self.to_factorize_view(), np.nan  # both NumPy

    @classmethod
    def _from_factorized(cls, values, original):
        # Build EA back from factorized NumPy values
        return cls._from_numpy(values)

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

    def argsort(
        self,
        *,
        ascending: bool = True,
        kind: str = "quicksort",
        **kwargs: object,
    ) -> NDArray[np.intp]:
        """
        Return the indices that would sort the array.

        This method computes the permutation indices that would sort the underlying
        Arkouda data and returns them as a NumPy array, in accordance with the
        pandas ``ExtensionArray`` contract. The indices can be used to reorder the
        array via ``take`` or ``iloc``.

        For floating-point data, ``NaN`` values are handled according to the
        ``na_position`` keyword argument.

        Parameters
        ----------
        ascending : bool, default True
            If True, sort values in ascending order. If False, sort in descending
            order.
        kind : str, default "quicksort"
            Sorting algorithm. Present for API compatibility with NumPy and pandas
            but currently ignored.
        **kwargs
            Additional keyword arguments for compatibility. Supported keyword:

            * ``na_position`` : {"first", "last"}, default "last"
              Where to place ``NaN`` values in the sorted result. This option is
              currently only applied for floating-point ``pdarray`` data; for
              ``Strings`` and ``Categorical`` data it has no effect.

        Returns
        -------
        numpy.ndarray
            A 1D NumPy array of dtype ``np.intp`` containing the indices that would
            sort the array.

        Raises
        ------
        ValueError
            If ``na_position`` is not "first" or "last".
        TypeError
            If the underlying data type does not support sorting.

        Notes
        -----
        * Supports Arkouda ``pdarray``, ``Strings``, and ``Categorical`` data.
        * For floating-point arrays, ``NaN`` values are repositioned according to
          ``na_position``.
        * The sorting computation occurs on the Arkouda server, but the resulting
          permutation indices are materialized on the client as a NumPy array, as
          required by pandas internals.

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaArray
        >>> a = ArkoudaArray(ak.array([3.0, float("nan"), 1.0]))
        >>> a.argsort() # NA last by default
        array([2, 0, 1])
        >>> a.argsort(na_position="first")
        array([1, 2, 0])
        """
        from arkouda.numpy import argsort
        from arkouda.numpy.numeric import isnan as ak_isnan
        from arkouda.numpy.pdarraysetops import concatenate
        from arkouda.numpy.strings import Strings
        from arkouda.numpy.util import is_float
        from arkouda.pandas.categorical import Categorical

        # Extract na_position from kwargs
        na_position = kwargs.pop("na_position", "last")

        if na_position not in {"first", "last"}:
            raise ValueError("na_position must be 'first' or 'last'.")

        perm: pdarray

        if isinstance(self._data, (Strings, Categorical, pdarray)):
            perm = argsort(self._data, ascending=ascending)

            if is_float(self._data):
                is_nan = ak_isnan(self._data)[perm]
                if na_position == "last":
                    perm = concatenate([perm[~is_nan], perm[is_nan]])
                else:
                    perm = concatenate([perm[is_nan], perm[~is_nan]])
        else:
            raise TypeError(f"Unsupported argsort dtype: {type(self._data)}")

        return perm.to_ndarray().astype(np.intp, copy=False)

    def broadcast_arrays(self, *arrays):
        raise NotImplementedError(
            "ArkoudaExtensionArray.broadcast_arrays is not implemented in Arkouda yet"
        )

    def broadcast_to(self, x, shape, /):
        raise NotImplementedError("ArkoudaExtensionArray.broadcast_to is not implemented in Arkouda yet")

    def concat(self, arrays, /, *, axis=0):
        raise NotImplementedError("ArkoudaExtensionArray.concat is not implemented in Arkouda yet")

    def duplicated(self, arrays, /, *, axis=0):
        raise NotImplementedError("ArkoudaExtensionArray.duplicated is not implemented in Arkouda yet")

    def expand_dims(self, x, /, *, axis):
        raise NotImplementedError("ArkoudaExtensionArray.expand_dims is not implemented in Arkouda yet")

    def permute_dims(self, x, /, axes):
        raise NotImplementedError("ArkoudaExtensionArray.permute_dims is not implemented in Arkouda yet")

    def reshape(self, x, /, shape):
        raise NotImplementedError("ArkoudaExtensionArray.reshape is not implemented in Arkouda yet")

    def split(self, x, indices_or_sections, /, *, axis=0):
        raise NotImplementedError("ArkoudaExtensionArray.split is not implemented in Arkouda yet")

    def squeeze(self, x, /, *, axis=None):
        raise NotImplementedError("ArkoudaExtensionArray.squeeze is not implemented in Arkouda yet")

    def stack(self, arrays, /, *, axis=0):
        raise NotImplementedError("ArkoudaExtensionArray.stack is not implemented in Arkouda yet")

    def __hash__(self):
        raise NotImplementedError(
            "__hash__ is not yet implemented for ArkoudaExtensionArray. "
            "Use .to_numpy() and hash the result if needed."
        )

    def argmax(self, axis=None, out=None):
        raise NotImplementedError("argmax is not yet implemented for ArkoudaExtensionArray.")

    def argmin(self, axis=None, out=None):
        raise NotImplementedError("argmin is not yet implemented for ArkoudaExtensionArray.")

    def _mode(self, dropna=True):
        raise NotImplementedError("_mode is not yet implemented for ArkoudaExtensionArray.")

    def _quantile(self, q, interpolation="linear"):
        raise NotImplementedError("_quantile is not yet implemented for ArkoudaExtensionArray.")

    def _empty(self, *args, **kwargs):
        raise NotImplementedError("_empty is not yet implemented for ArkoudaExtensionArray.")

    def _accumulate(self, name, *, skipna=True, **kwargs):
        raise NotImplementedError("_accumulate is not yet implemented for ArkoudaExtensionArray.")

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy=False):
        raise NotImplementedError(
            "_from_sequence_of_strings is not yet implemented for ArkoudaExtensionArray."
        )

    def _values_for_json(self):
        raise NotImplementedError("_values_for_json is not yet implemented for ArkoudaExtensionArray.")

    def _where(self, mask, other):
        raise NotImplementedError("_where is not yet implemented for ArkoudaExtensionArray.")

    def interpolate(self, method="linear", *, limit=None, **kwargs):
        raise NotImplementedError("interpolate is not yet implemented for ArkoudaExtensionArray.")

    def view(self, dtype=None):
        raise NotImplementedError("view is not yet implemented for ArkoudaExtensionArray.")

    def _pad_or_backfill(self, method, limit=None, mask=None):
        raise NotImplementedError("_pad_or_backfill is not yet implemented for ArkoudaExtensionArray.")

    # ------------------------------------------------------------------
    # Arithmetic / comparison / logical ops hooks
    # ------------------------------------------------------------------

    def _cmp_method(self, *args, **kwargs):
        raise NotImplementedError("_cmp_method is not yet implemented for ArkoudaExtensionArray.")

    def _logical_method(self, *args, **kwargs):
        raise NotImplementedError("_logical_method is not yet implemented for ArkoudaExtensionArray.")

    # ------------------------------------------------------------------
    # NDArray-backed construction / wrapping
    # ------------------------------------------------------------------

    @classmethod
    def _from_backing_data(cls, *args, **kwargs):
        raise NotImplementedError("_from_backing_data is not yet implemented for ArkoudaExtensionArray.")

    @classmethod
    def _simple_new(cls, *args, **kwargs):
        raise NotImplementedError("_simple_new is not yet implemented for ArkoudaExtensionArray.")

    def _box_func(self, *args, **kwargs):
        raise NotImplementedError("_box_func is not yet implemented for ArkoudaExtensionArray.")

    def _validate_scalar(self, *args, **kwargs):
        raise NotImplementedError("_validate_scalar is not yet implemented for ArkoudaExtensionArray.")

    def _validate_setitem_value(self, *args, **kwargs):
        raise NotImplementedError(
            "_validate_setitem_value is not yet implemented for ArkoudaExtensionArray."
        )

    def _wrap_ndarray_result(self, *args, **kwargs):
        raise NotImplementedError(
            "_wrap_ndarray_result is not yet implemented for ArkoudaExtensionArray."
        )

    def _wrap_reduction_result(self, *args, **kwargs):
        raise NotImplementedError(
            "_wrap_reduction_result is not yet implemented for ArkoudaExtensionArray."
        )

    # ------------------------------------------------------------------
    # Reductions (currently delegated to arkouda-side implementations or unsupported)
    # ------------------------------------------------------------------

    def _reduction_not_implemented(self, name: str):
        raise NotImplementedError(f"{name} is not yet implemented for ArkoudaExtensionArray.")

    def kurt(self, *args, **kwargs):
        self._reduction_not_implemented("kurt")

    def median(self, *args, **kwargs):
        self._reduction_not_implemented("median")

    def sem(self, *args, **kwargs):
        self._reduction_not_implemented("sem")

    def skew(self, *args, **kwargs):
        self._reduction_not_implemented("skew")

    def swapaxes(self, *args, **kwargs):
        self._reduction_not_implemented("swapaxes")

    def value_counts(self, *args, **kwargs):
        self._reduction_not_implemented("value_counts")

    # ------------------------------------------------------------------
    # String-like methods
    # ------------------------------------------------------------------

    def _string_not_supported(self, name: str):
        raise NotImplementedError(f"{name} is not supported for ArkoudaExtensionArray.")

    def _str_capitalize(self, *args, **kwargs):
        self._string_not_supported("_str_capitalize")

    def _str_casefold(self, *args, **kwargs):
        self._string_not_supported("_str_casefold")

    def _str_contains(self, *args, **kwargs):
        self._string_not_supported("_str_contains")

    def _str_count(self, *args, **kwargs):
        self._string_not_supported("_str_count")

    def _str_encode(self, *args, **kwargs):
        self._string_not_supported("_str_encode")

    def _str_endswith(self, *args, **kwargs):
        self._string_not_supported("_str_endswith")

    def _str_extract(self, *args, **kwargs):
        self._string_not_supported("_str_extract")

    def _str_find(self, *args, **kwargs):
        self._string_not_supported("_str_find")

    def _str_find_(self, *args, **kwargs):
        self._string_not_supported("_str_find_")

    def _str_findall(self, *args, **kwargs):
        self._string_not_supported("_str_findall")

    def _str_fullmatch(self, *args, **kwargs):
        self._string_not_supported("_str_fullmatch")

    def _str_get(self, *args, **kwargs):
        self._string_not_supported("_str_get")

    def _str_get_dummies(self, *args, **kwargs):
        self._string_not_supported("_str_get_dummies")

    def _str_getitem(self, *args, **kwargs):
        self._string_not_supported("_str_getitem")

    def _str_index(self, *args, **kwargs):
        self._string_not_supported("_str_index")

    def _str_isalnum(self, *args, **kwargs):
        self._string_not_supported("_str_isalnum")

    def _str_isalpha(self, *args, **kwargs):
        self._string_not_supported("_str_isalpha")

    def _str_isdecimal(self, *args, **kwargs):
        self._string_not_supported("_str_isdecimal")

    def _str_isdigit(self, *args, **kwargs):
        self._string_not_supported("_str_isdigit")

    def _str_islower(self, *args, **kwargs):
        self._string_not_supported("_str_islower")

    def _str_isnumeric(self, *args, **kwargs):
        self._string_not_supported("_str_isnumeric")

    def _str_isspace(self, *args, **kwargs):
        self._string_not_supported("_str_isspace")

    def _str_istitle(self, *args, **kwargs):
        self._string_not_supported("_str_istitle")

    def _str_isupper(self, *args, **kwargs):
        self._string_not_supported("_str_isupper")

    def _str_join(self, *args, **kwargs):
        self._string_not_supported("_str_join")

    def _str_len(self, *args, **kwargs):
        self._string_not_supported("_str_len")

    def _str_lower(self, *args, **kwargs):
        self._string_not_supported("_str_lower")

    def _str_lstrip(self, *args, **kwargs):
        self._string_not_supported("_str_lstrip")

    def _str_map(self, *args, **kwargs):
        self._string_not_supported("_str_map")

    def _str_match(self, *args, **kwargs):
        self._string_not_supported("_str_match")

    def _str_normalize(self, *args, **kwargs):
        self._string_not_supported("_str_normalize")

    def _str_pad(self, *args, **kwargs):
        self._string_not_supported("_str_pad")

    def _str_partition(self, *args, **kwargs):
        self._string_not_supported("_str_partition")

    def _str_removeprefix(self, *args, **kwargs):
        self._string_not_supported("_str_removeprefix")

    def _str_removesuffix(self, *args, **kwargs):
        self._string_not_supported("_str_removesuffix")

    def _str_repeat(self, *args, **kwargs):
        self._string_not_supported("_str_repeat")

    def _str_replace(self, *args, **kwargs):
        self._string_not_supported("_str_replace")

    def _str_rfind(self, *args, **kwargs):
        self._string_not_supported("_str_rfind")

    def _str_rindex(self, *args, **kwargs):
        self._string_not_supported("_str_rindex")

    def _str_rpartition(self, *args, **kwargs):
        self._string_not_supported("_str_rpartition")

    def _str_rsplit(self, *args, **kwargs):
        self._string_not_supported("_str_rsplit")

    def _str_rstrip(self, *args, **kwargs):
        self._string_not_supported("_str_rstrip")

    def _str_slice(self, *args, **kwargs):
        self._string_not_supported("_str_slice")

    def _str_slice_replace(self, *args, **kwargs):
        self._string_not_supported("_str_slice_replace")

    def _str_split(self, *args, **kwargs):
        self._string_not_supported("_str_split")

    def _str_startswith(self, *args, **kwargs):
        self._string_not_supported("_str_startswith")

    def _str_strip(self, *args, **kwargs):
        self._string_not_supported("_str_strip")

    def _str_swapcase(self, *args, **kwargs):
        self._string_not_supported("_str_swapcase")

    def _str_title(self, *args, **kwargs):
        self._string_not_supported("_str_title")

    def _str_translate(self, *args, **kwargs):
        self._string_not_supported("_str_translate")

    def _str_upper(self, *args, **kwargs):
        self._string_not_supported("_str_upper")

    def _str_wrap(self, *args, **kwargs):
        self._string_not_supported("_str_wrap")
