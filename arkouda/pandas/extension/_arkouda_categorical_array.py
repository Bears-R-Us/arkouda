from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, TypeVar

from numpy import ndarray
from pandas.api.extensions import ExtensionArray

import arkouda as ak

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

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        from arkouda import Categorical, array

        # if 'scalars' are raw labels (strings), build ak.Categorical
        if not isinstance(scalars, Categorical):
            scalars = Categorical(array(scalars))
        return cls(scalars)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._data[idx]
        return ArkoudaCategoricalArray(self._data[idx])

    def astype(self, x, dtype):
        raise NotImplementedError("array_api.astype is not implemented in Arkouda yet")

    def isna(self):
        return ak.zeros(self._data.size, dtype=ak.bool)

    @property
    def dtype(self):
        return ArkoudaCategoricalDtype()

    def __eq__(self, other):
        return self._data == (other._data if isinstance(other, ArkoudaCategoricalArray) else other)

    def __repr__(self):
        return f"ArkoudaCategoricalArray({self._data})"

    # ------------------------------------------------------------------
    # pandas.Categorical-specific API that is not yet implemented
    # ------------------------------------------------------------------

    def _categories_match_up_to_permutation(self, *args, **kwargs):
        raise NotImplementedError(
            "_categories_match_up_to_permutation is not yet implemented for ArkoudaCategoricalArray."
        )

    def _constructor(self, *args, **kwargs):
        raise NotImplementedError("_constructor is not yet implemented for ArkoudaCategoricalArray.")

    def _dir_additions(self, *args, **kwargs):
        raise NotImplementedError("_dir_additions is not yet implemented for ArkoudaCategoricalArray.")

    def _dir_deletions(self, *args, **kwargs):
        raise NotImplementedError("_dir_deletions is not yet implemented for ArkoudaCategoricalArray.")

    def _encode_with_my_categories(self, *args, **kwargs):
        raise NotImplementedError(
            "_encode_with_my_categories is not yet implemented for ArkoudaCategoricalArray."
        )

    def _from_inferred_categories(self, *args, **kwargs):
        raise NotImplementedError(
            "_from_inferred_categories is not yet implemented for ArkoudaCategoricalArray."
        )

    def _get_values_repr(self, *args, **kwargs):
        raise NotImplementedError("_get_values_repr is not yet implemented for ArkoudaCategoricalArray.")

    def _internal_get_values(self, *args, **kwargs):
        raise NotImplementedError(
            "_internal_get_values is not yet implemented for ArkoudaCategoricalArray."
        )

    def _replace(self, *args, **kwargs):
        raise NotImplementedError("_replace is not yet implemented for ArkoudaCategoricalArray.")

    def _repr_categories(self, *args, **kwargs):
        raise NotImplementedError("_repr_categories is not yet implemented for ArkoudaCategoricalArray.")

    def _reset_cache(self, *args, **kwargs):
        raise NotImplementedError("_reset_cache is not yet implemented for ArkoudaCategoricalArray.")

    def _reverse_indexer(self, *args, **kwargs):
        raise NotImplementedError("_reverse_indexer is not yet implemented for ArkoudaCategoricalArray.")

    def _set_categories(self, *args, **kwargs):
        raise NotImplementedError("_set_categories is not yet implemented for ArkoudaCategoricalArray.")

    def _set_dtype(self, *args, **kwargs):
        raise NotImplementedError("_set_dtype is not yet implemented for ArkoudaCategoricalArray.")

    def _unbox_scalar(self, *args, **kwargs):
        raise NotImplementedError("_unbox_scalar is not yet implemented for ArkoudaCategoricalArray.")

    def _validate_codes_for_dtype(self, *args, **kwargs):
        raise NotImplementedError(
            "_validate_codes_for_dtype is not yet implemented for ArkoudaCategoricalArray."
        )

    def _validate_listlike(self, *args, **kwargs):
        raise NotImplementedError(
            "_validate_listlike is not yet implemented for ArkoudaCategoricalArray."
        )

    def _values_for_rank(self, *args, **kwargs):
        raise NotImplementedError("_values_for_rank is not yet implemented for ArkoudaCategoricalArray.")

    def add_categories(self, *args, **kwargs):
        raise NotImplementedError("add_categories is not yet implemented for ArkoudaCategoricalArray.")

    def as_ordered(self, *args, **kwargs):
        raise NotImplementedError("as_ordered is not yet implemented for ArkoudaCategoricalArray.")

    def as_unordered(self, *args, **kwargs):
        raise NotImplementedError("as_unordered is not yet implemented for ArkoudaCategoricalArray.")

    def check_for_ordered(self, *args, **kwargs):
        raise NotImplementedError(
            "check_for_ordered is not yet implemented for ArkoudaCategoricalArray."
        )

    def describe(self, *args, **kwargs):
        raise NotImplementedError("describe is not yet implemented for ArkoudaCategoricalArray.")

    def from_codes(self, *args, **kwargs):
        raise NotImplementedError("from_codes is not yet implemented for ArkoudaCategoricalArray.")

    def isnull(self, *args, **kwargs):
        raise NotImplementedError("isnull is not yet implemented for ArkoudaCategoricalArray.")

    def memory_usage(self, *args, **kwargs):
        raise NotImplementedError("memory_usage is not yet implemented for ArkoudaCategoricalArray.")

    def notna(self, *args, **kwargs):
        raise NotImplementedError("notna is not yet implemented for ArkoudaCategoricalArray.")

    def notnull(self, *args, **kwargs):
        raise NotImplementedError("notnull is not yet implemented for ArkoudaCategoricalArray.")

    def remove_categories(self, *args, **kwargs):
        raise NotImplementedError(
            "remove_categories is not yet implemented for ArkoudaCategoricalArray."
        )

    def remove_unused_categories(self, *args, **kwargs):
        raise NotImplementedError(
            "remove_unused_categories is not yet implemented for ArkoudaCategoricalArray."
        )

    def rename_categories(self, *args, **kwargs):
        raise NotImplementedError(
            "rename_categories is not yet implemented for ArkoudaCategoricalArray."
        )

    def reorder_categories(self, *args, **kwargs):
        raise NotImplementedError(
            "reorder_categories is not yet implemented for ArkoudaCategoricalArray."
        )

    def set_categories(self, *args, **kwargs):
        raise NotImplementedError("set_categories is not yet implemented for ArkoudaCategoricalArray.")

    def set_ordered(self, *args, **kwargs):
        raise NotImplementedError("set_ordered is not yet implemented for ArkoudaCategoricalArray.")

    def sort_values(self, *args, **kwargs):
        raise NotImplementedError("sort_values is not yet implemented for ArkoudaCategoricalArray.")

    def to_list(self, *args, **kwargs):
        raise NotImplementedError("to_list is not yet implemented for ArkoudaCategoricalArray.")
