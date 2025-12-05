from typing import TYPE_CHECKING, TypeVar

import numpy as np  # new

from pandas.api.extensions import ExtensionArray

import arkouda as ak

from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.pdarraycreation import full as ak_full
from arkouda.numpy.pdarraycreation import pdarray

from ._arkouda_array import ArkoudaArray
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
        if not isinstance(data, ak.Categorical):
            raise TypeError("Expected arkouda Categorical")
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
        """Elementwise equality for ArkoudaCategoricalArray."""
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
            if other_ak.size != len(self):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other_ak)

        # Case 5: unsupported type → all False
        return ArkoudaArray(ak_full(len(self), False, dtype=bool))

    def __repr__(self):
        return f"ArkoudaCategoricalArray({self._data})"
