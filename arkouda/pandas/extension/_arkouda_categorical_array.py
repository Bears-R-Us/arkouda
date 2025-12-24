from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, TypeVar, Union, cast, overload

import numpy as np  # new

from numpy import ndarray
from numpy.typing import NDArray
from pandas import CategoricalDtype as pd_CategoricalDtype
from pandas.core.arrays.base import ExtensionArray
from pandas.core.dtypes.dtypes import ExtensionDtype

import arkouda as ak

from ._arkouda_array import ArkoudaArray
from ._arkouda_extension_array import ArkoudaExtensionArray
from ._arkouda_string_array import ArkoudaStringArray
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

    @overload
    def astype(self, dtype: np.dtype[Any], copy: bool = True) -> NDArray[Any]: ...

    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = True) -> ExtensionArray: ...

    @overload
    def astype(self, dtype: Any, copy: bool = True) -> Union[ExtensionArray, NDArray[Any]]: ...

    def astype(
        self,
        dtype: Any,
        copy: bool = False,
    ) -> Union[ExtensionArray, NDArray[Any]]:
        """
        Cast to a specified dtype.

        * If ``dtype`` is categorical (pandas ``category`` / ``CategoricalDtype`` /
          ``ArkoudaCategoricalDtype``), returns an Arkouda-backed
          ``ArkoudaCategoricalArray`` (optionally copied).
        * If ``dtype`` requests ``object``, returns a NumPy ``ndarray`` of dtype object
          containing the category labels (materialized to the client).
        * If ``dtype`` requests a string dtype, returns an Arkouda-backed
          ``ArkoudaStringArray`` containing the labels as strings.
        * Otherwise, casts the labels (as strings) to the requested dtype and returns an
          Arkouda-backed ExtensionArray.

        Parameters
        ----------
        dtype : Any
            Target dtype.
        copy : bool
            Whether to force a copy when possible. If categorical-to-categorical and
            ``copy=True``, attempts to copy the underlying Arkouda ``Categorical`` (if
            supported). Default is False.

        Returns
        -------
        Union[ExtensionArray, NDArray[Any]]
            The cast result. Returns a NumPy array only when casting to ``object``;
            otherwise returns an Arkouda-backed ExtensionArray.

        Examples
        --------
        Casting to ``category`` returns an Arkouda-backed categorical array:

        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaCategoricalArray
        >>> c = ArkoudaCategoricalArray(ak.Categorical(ak.array(["x", "y", "x"])))
        >>> out = c.astype("category")
        >>> out is c
        True

        Forcing a copy when casting to the same categorical dtype returns a new array:

        >>> out2 = c.astype("category", copy=True)
        >>> out2 is c
        False
        >>> out2.to_ndarray()
        array(['x', 'y', 'x'], dtype='<U...')

        Casting to ``object`` materializes the category labels to a NumPy object array:

        >>> c.astype(object)
        array(['x', 'y', 'x'], dtype=object)

        Casting to a string dtype returns an Arkouda-backed string array of labels:

        >>> s = c.astype("string")
        >>> s.to_ndarray()
        array(['x', 'y', 'x'], dtype='<U1')

        Casting to another dtype casts the labels-as-strings and returns an Arkouda-backed array:

        >>> c_num = ArkoudaCategoricalArray(ak.Categorical(ak.array(["1", "2", "3"])))
        >>> a = c_num.astype("int64")
        >>> a.to_ndarray()
        array([1, 2, 3])
        """
        from arkouda.numpy._typing._typing import is_string_dtype_hint

        # --- 1) ExtensionDtype branch first: proves overload #2 returns ExtensionArray ---
        if isinstance(dtype, ExtensionDtype):
            if hasattr(dtype, "numpy_dtype"):
                dtype = dtype.numpy_dtype

            if isinstance(dtype, (ArkoudaCategoricalDtype, pd_CategoricalDtype)) or dtype in (
                "category",
            ):
                if not copy:
                    return self
                data = self._data.copy() if hasattr(self._data, "copy") else self._data
                return cast(ExtensionArray, type(self)(data))

            data = self._data.to_strings()

            if is_string_dtype_hint(dtype):
                return cast(ExtensionArray, ArkoudaStringArray._from_sequence(data))

            casted = data.astype(dtype)
            return cast(ExtensionArray, ArkoudaExtensionArray._from_sequence(casted))

        # --- 2) object -> numpy ---
        if dtype in (object, np.object_, "object", np.dtype("O")):
            return self.to_ndarray().astype(object, copy=copy)

        if isinstance(dtype, (ArkoudaCategoricalDtype, pd_CategoricalDtype)) or dtype in ("category",):
            if not copy:
                return self
            data = self._data.copy() if hasattr(self._data, "copy") else self._data
            return type(self)(data)

        data = self._data.to_strings()

        if is_string_dtype_hint(dtype):
            return ArkoudaStringArray._from_sequence(data)

        casted = data.astype(dtype)
        return ArkoudaExtensionArray._from_sequence(casted)

    def isna(self):
        return ak.zeros(self._data.size, dtype=ak.bool)

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

    def _not_implemented(self, name: str):
        raise NotImplementedError(f"`{name}` is not implemented for ArkoudaCategoricalArray yet.")

    def add_categories(self, *args, **kwargs):
        self._not_implemented("add_categories")

    def as_ordered(self, *args, **kwargs):
        self._not_implemented("as_ordered")

    def as_unordered(self, *args, **kwargs):
        self._not_implemented("as_unordered")

    def check_for_ordered(self, *args, **kwargs):
        self._not_implemented("check_for_ordered")

    def describe(self, *args, **kwargs):
        self._not_implemented("describe")

    @classmethod
    def from_codes(cls, *args, **kwargs):
        raise NotImplementedError("`from_codes` is not implemented for ArkoudaCategoricalArray yet.")

    def isnull(self, *args, **kwargs):
        self._not_implemented("isnull")

    def max(self, *args, **kwargs):
        self._not_implemented("max")

    def memory_usage(self, *args, **kwargs):
        self._not_implemented("memory_usage")

    def min(self, *args, **kwargs):
        self._not_implemented("min")

    def notna(self, *args, **kwargs):
        self._not_implemented("notna")

    def notnull(self, *args, **kwargs):
        self._not_implemented("notnull")

    def remove_categories(self, *args, **kwargs):
        self._not_implemented("remove_categories")

    def remove_unused_categories(self, *args, **kwargs):
        self._not_implemented("remove_unused_categories")

    def rename_categories(self, *args, **kwargs):
        self._not_implemented("rename_categories")

    def reorder_categories(self, *args, **kwargs):
        self._not_implemented("reorder_categories")

    def set_categories(self, *args, **kwargs):
        self._not_implemented("set_categories")

    def set_ordered(self, *args, **kwargs):
        self._not_implemented("set_ordered")

    def sort_values(self, *args, **kwargs):
        self._not_implemented("sort_values")

    def swapaxes(self, *args, **kwargs):
        self._not_implemented("swapaxes")

    def to_list(self, *args, **kwargs):
        self._not_implemented("to_list")

    def value_counts(self, *args, **kwargs):
        self._not_implemented("value_counts")
