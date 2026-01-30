from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, TypeVar

import numpy as np  # new

from numpy import ndarray
from pandas.api.extensions import ExtensionArray

from arkouda.numpy.dtypes import bool_
from arkouda.numpy.pdarrayclass import pdarray

from ._arkouda_array import ArkoudaArray
from ._arkouda_extension_array import ArkoudaExtensionArray
from ._dtypes import ArkoudaCategoricalDtype


if TYPE_CHECKING:
    from arkouda.categorical import Categorical
else:
    Categorical = TypeVar("Categorical")


__all__ = ["ArkoudaCategoricalArray"]


class ArkoudaCategoricalArray(ArkoudaExtensionArray, ExtensionArray):
    """
    Arkouda-backed categorical pandas ExtensionArray.

    Ensures the underlying data is an Arkouda ``Categorical``. Accepts an existing
    ``Categorical`` or converts from Python/NumPy sequences of labels.

    Parameters
    ----------
    data : Categorical | ArkoudaCategoricalArray | ndarray | Sequence[Any]
        Input to wrap or convert.
        - If ``Categorical``, used directly.
        - If another ``ArkoudaCategoricalArray``, its backing object is reused.
        - If list/tuple/ndarray, converted via ``ak.Categorical(ak.array(data))``.

    Raises
    ------
    TypeError
        If ``data`` cannot be converted to Arkouda ``Categorical``.

    Attributes
    ----------
    default_fill_value : str
        Sentinel used when filling missing values (default: "").
    """

    default_fill_value: str = ""

    def __init__(self, data: Categorical | "ArkoudaCategoricalArray" | ndarray | Sequence[Any]):
        from arkouda import Categorical as AkCategorical
        from arkouda import array

        if isinstance(data, ArkoudaCategoricalArray):
            self._data = data._data
            return

        if not isinstance(data, AkCategorical):
            try:
                data = AkCategorical(array(data))
            except Exception as e:
                raise TypeError(
                    f"Expected arkouda.Categorical or sequence convertible to one, "
                    f"got {type(data).__name__}"
                ) from e

        self._data = data

    def __getitem__(self, key: Any) -> Any:
        """
        Retrieve one or more categorical values.

        Parameters
        ----------
        key : Any
            Location(s) to retrieve. Supported forms include:

            * scalar integer index
            * slice objects (e.g. ``1:3``)
            * NumPy integer array (any integer dtype)
            * NumPy boolean mask with the same length as the array
            * Python list of integers or booleans
            * Arkouda ``pdarray`` of integers or booleans

        Returns
        -------
        Any
            A Python scalar for scalar access, or a new
            :class:`ArkoudaCategoricalArray` for non-scalar indexers.

        Raises
        ------
        TypeError
            If a NumPy indexer with an unsupported dtype is provided.

        Examples
        --------
        >>> import numpy as np
        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaCategoricalArray
        >>> data = ak.Categorical(ak.array(["a", "b", "c", "d"]))
        >>> arr = ArkoudaCategoricalArray(data)

        Scalar access returns a Python string-like scalar:

        >>> arr[1]
        np.str_('b')

        Negative indexing:

        >>> arr[-1]
        np.str_('d')

        Slice indexing returns a new ArkoudaCategoricalArray:

        >>> result = arr[1:3]
        >>> type(result)
        <class 'arkouda.pandas.extension._arkouda_categorical_array.ArkoudaCategoricalArray'>

        NumPy integer array indexing:

        >>> idx = np.array([0, 2], dtype=np.int64)
        >>> sliced = arr[idx]
        >>> isinstance(sliced, ArkoudaCategoricalArray)
        True

        NumPy boolean mask:

        >>> mask = np.array([True, False, True, False])
        >>> masked = arr[mask]
        >>> isinstance(masked, ArkoudaCategoricalArray)
        True

        Empty integer indexer returns an empty ArkoudaCategoricalArray:

        >>> empty_idx = np.array([], dtype=np.int64)
        >>> empty = arr[empty_idx]
        >>> len(empty)
        0
        """
        import numpy as np

        from arkouda.numpy.pdarraycreation import array as ak_array
        from arkouda.pandas.categorical import Categorical

        # Handle empty indexer (list / tuple / ndarray of length 0)
        if isinstance(key, (list, tuple, np.ndarray)) and len(key) == 0:
            empty_strings = ak_array([], dtype="str_")
            return ArkoudaCategoricalArray(Categorical(empty_strings))

        # Scalar integers and slices: delegate directly to the underlying Categorical
        if isinstance(key, (int, np.integer, slice)):
            result = self._data[key]
            # For scalar keys, just return the underlying scalar
            if isinstance(key, (int, np.integer)):
                return result
            # For slices, underlying arkouda.Categorical returns a Categorical
            return ArkoudaCategoricalArray(result)

        # NumPy array indexers: normalize to Arkouda pdarrays
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                key = ak_array(key)
            elif np.issubdtype(key.dtype, np.signedinteger):
                key = ak_array(key, dtype="int64")
            elif np.issubdtype(key.dtype, np.unsignedinteger):
                key = ak_array(key, dtype="uint64")
            else:
                raise TypeError(f"Unsupported numpy index type {key.dtype}")
        elif not isinstance(key, (pdarray, Categorical)):
            # Convert generic indexers (e.g. Python lists of ints/bools) to an Arkouda pdarray
            key = ak_array(key)

        # Delegate to underlying arkouda.Categorical
        result = self._data[key]

        # Scalar result: just return the underlying scalar
        if isinstance(key, pdarray) and key.size == 1:
            # Categorical.__getitem__ will generally still give a Categorical here;
            # we normalize to a Python scalar by going through categories[codes].

            codes = result.codes if isinstance(result, Categorical) else result
            cats = self._data.categories
            # codes is length-1, so this is length-1 Strings
            labels = cats[codes]
            # Return a Python scalar string
            return labels[0]

        # Non-scalar: wrap Categorical in ArkoudaCategoricalArray
        if isinstance(result, Categorical):
            return ArkoudaCategoricalArray(result)

        # Fallback: if Categorical returned something array-like but not Categorical,
        # rebuild a Categorical from it.
        return ArkoudaCategoricalArray(Categorical(result))

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        from arkouda import Categorical, array

        # if 'scalars' are raw labels (strings), build ak.Categorical
        if not isinstance(scalars, Categorical):
            scalars = Categorical(array(scalars))
        return cls(scalars)

    def astype(self, x, dtype):
        raise NotImplementedError("array_api.astype is not implemented in Arkouda yet")

    def isna(self):
        from arkouda.numpy.pdarraycreation import zeros

        return zeros(self._data.size, dtype=bool_)

    @property
    def dtype(self):
        return ArkoudaCategoricalDtype()

    def __eq__(self, other):
        """Elementwise equality for ArkoudaCategoricalArray."""
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array
        from arkouda.pandas.categorical import Categorical

        # Case 1: Categorical vs Categorical
        if isinstance(other, ArkoudaCategoricalArray):
            if len(self) != len(other):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other._data)

        # Case 2: Categorical vs arkouda pdarray (e.g., codes or labels, depending on ak semantics)
        if isinstance(other, pdarray):
            if other.size not in (1, len(self)):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other)

        # Case 3: scalar (string / category label / code)
        if np.isscalar(other):
            return ArkoudaArray(self._data == other)

        # Case 4: numpy array or Python sequence
        if isinstance(other, (list, tuple, np.ndarray)):
            other_ak = Categorical(ak_array(other))
            if other_ak.size == 1:
                return ArkoudaArray(self._data == other_ak[0])
            if other_ak.size != len(self):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other_ak)

        # Case 5: unsupported type
        return NotImplemented

    def __repr__(self):
        return f"ArkoudaCategoricalArray({self._data})"

    # ------------------------------------------------------------------
    # pandas.Categorical-specific API that is not yet implemented
    # ------------------------------------------------------------------

    def _categorical_not_implemented(self, name: str):
        raise NotImplementedError(f"{name} is not yet implemented for ArkoudaCategoricalArray.")

    def _categories_match_up_to_permutation(self, *args, **kwargs):
        self._categorical_not_implemented("_categories_match_up_to_permutation")

    def _constructor(self, *args, **kwargs):
        self._categorical_not_implemented("_constructor")

    def _dir_additions(self, *args, **kwargs):
        self._categorical_not_implemented("_dir_additions")

    def _dir_deletions(self, *args, **kwargs):
        self._categorical_not_implemented("_dir_deletions")

    def _encode_with_my_categories(self, *args, **kwargs):
        self._categorical_not_implemented("_encode_with_my_categories")

    def _from_inferred_categories(self, *args, **kwargs):
        self._categorical_not_implemented("_from_inferred_categories")

    def _get_values_repr(self, *args, **kwargs):
        self._categorical_not_implemented("_get_values_repr")

    def _internal_get_values(self, *args, **kwargs):
        self._categorical_not_implemented("_internal_get_values")

    def _replace(self, *args, **kwargs):
        self._categorical_not_implemented("_replace")

    def _repr_categories(self, *args, **kwargs):
        self._categorical_not_implemented("_repr_categories")

    def _reset_cache(self, *args, **kwargs):
        self._categorical_not_implemented("_reset_cache")

    def _reverse_indexer(self, *args, **kwargs):
        self._categorical_not_implemented("_reverse_indexer")

    def _set_categories(self, *args, **kwargs):
        self._categorical_not_implemented("_set_categories")

    def _set_dtype(self, *args, **kwargs):
        self._categorical_not_implemented("_set_dtype")

    def _unbox_scalar(self, *args, **kwargs):
        self._categorical_not_implemented("_unbox_scalar")

    def _validate_codes_for_dtype(self, *args, **kwargs):
        self._categorical_not_implemented("_validate_codes_for_dtype")

    def _validate_listlike(self, *args, **kwargs):
        self._categorical_not_implemented("_validate_listlike")

    def _values_for_rank(self, *args, **kwargs):
        self._categorical_not_implemented("_values_for_rank")

    def add_categories(self, *args, **kwargs):
        self._categorical_not_implemented("add_categories")

    def as_ordered(self, *args, **kwargs):
        self._categorical_not_implemented("as_ordered")

    def as_unordered(self, *args, **kwargs):
        self._categorical_not_implemented("as_unordered")

    def check_for_ordered(self, *args, **kwargs):
        self._categorical_not_implemented("check_for_ordered")

    def describe(self, *args, **kwargs):
        self._categorical_not_implemented("describe")

    @classmethod
    def from_codes(cls, *args, **kwargs):
        raise NotImplementedError("from_codes is not yet implemented for ArkoudaCategoricalArray.")

    def isnull(self, *args, **kwargs):
        self._categorical_not_implemented("isnull")

    def memory_usage(self, *args, **kwargs):
        self._categorical_not_implemented("memory_usage")

    def notna(self, *args, **kwargs):
        self._categorical_not_implemented("notna")

    def notnull(self, *args, **kwargs):
        self._categorical_not_implemented("notnull")

    def remove_categories(self, *args, **kwargs):
        self._categorical_not_implemented("remove_categories")

    def remove_unused_categories(self, *args, **kwargs):
        self._categorical_not_implemented("remove_unused_categories")

    def rename_categories(self, *args, **kwargs):
        self._categorical_not_implemented("rename_categories")

    def reorder_categories(self, *args, **kwargs):
        self._categorical_not_implemented("reorder_categories")

    def set_categories(self, *args, **kwargs):
        self._categorical_not_implemented("set_categories")

    def set_ordered(self, *args, **kwargs):
        self._categorical_not_implemented("set_ordered")

    def sort_values(self, *args, **kwargs):
        self._categorical_not_implemented("sort_values")

    def to_list(self, *args, **kwargs):
        self._categorical_not_implemented("to_list")

    def max(self, *args, **kwargs):
        self._categorical_not_implemented("max")

    def min(self, *args, **kwargs):
        self._categorical_not_implemented("min")

    def value_counts(self, *args, **kwargs):
        self._categorical_not_implemented("value_counts")
