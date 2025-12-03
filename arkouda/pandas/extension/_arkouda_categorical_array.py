from typing import TYPE_CHECKING, Any, TypeVar

from pandas.api.extensions import ExtensionArray

from arkouda.numpy.dtypes import bool_
from arkouda.numpy.pdarrayclass import pdarray
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.pdarraycreation import zeros

from ._arkouda_extension_array import ArkoudaExtensionArray
from ._dtypes import ArkoudaCategoricalDtype


if TYPE_CHECKING:
    from arkouda.categorical import Categorical
else:
    Categorical = TypeVar("Categorical")


__all__ = ["ArkoudaCategoricalArray"]


class ArkoudaCategoricalArray(ArkoudaExtensionArray, ExtensionArray):
    default_fill_value: str = ""

    def __init__(self, data):
        from arkouda.pandas.categorical import Categorical

        if not isinstance(data, Categorical):
            data = Categorical(data)
            # raise TypeError("Expected arkouda Categorical")
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
        return zeros(self._data.size, dtype=bool_)

    @property
    def dtype(self):
        return ArkoudaCategoricalDtype()

    def __eq__(self, other):
        return self._data == (other._data if isinstance(other, ArkoudaCategoricalArray) else other)

    def __repr__(self):
        return f"ArkoudaCategoricalArray({self._data})"
