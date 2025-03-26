from typing import TYPE_CHECKING, TypeVar

from pandas.api.extensions import ExtensionArray

import arkouda as ak

from ._arkouda_base_array import ArkoudaBaseArray
from ._dtypes import ArkoudaCategoricalDtype


if TYPE_CHECKING:
    from arkouda.categorical import Categorical
else:
    Categorical = TypeVar("Categorical")


__all__ = ["ArkoudaCategoricalArray"]


class ArkoudaCategoricalArray(ArkoudaBaseArray, ExtensionArray):
    default_fill_value: str = ""

    def __init__(self, data):
        if not isinstance(data, ak.Categorical):
            raise TypeError("Expected arkouda Categorical")
        self._data = data

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        import arkouda as ak

        # if 'scalars' are raw labels (strings), build ak.Categorical
        if not isinstance(scalars, ak.Categorical):
            scalars = ak.Categorical(ak.array(scalars))
        return cls(scalars)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._data[idx]
        return ArkoudaCategoricalArray(self._data[idx])

    def __len__(self):
        return int(self._data.size)

    def isna(self):
        return ak.zeros(self._data.size, dtype=ak.bool)

    def copy(self):
        return ArkoudaCategoricalArray(self._data[:])

    @property
    def dtype(self):
        return ArkoudaCategoricalDtype()

    def to_numpy(self, dtype=None, copy=False, na_value=None):
        return self._data.to_ndarray()

    def __eq__(self, other):
        return self._data == (other._data if isinstance(other, ArkoudaCategoricalArray) else other)

    def __repr__(self):
        return f"ArkoudaCategoricalArray({self._data})"

    def factorize(self, *, sort=False, use_na_sentinel=True, **kwargs):
        import numpy as np
        import pandas as pd

        codes, uniques = pd.factorize(
            np.asarray(self.to_numpy()),
            sort=sort,
            use_na_sentinel=use_na_sentinel,
        )
        return codes, uniques
