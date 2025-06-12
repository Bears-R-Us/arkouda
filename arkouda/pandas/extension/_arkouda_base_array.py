import numpy as np
from pandas.api.extensions import ExtensionArray

import arkouda as ak
from arkouda.numpy.pdarraysetops import concatenate as ak_concat

__all__ = ["_ensure_numpy", "ArkoudaBaseArray"]


def _ensure_numpy(x):
    if hasattr(x, "to_ndarray"):
        return x.to_ndarray()
    return np.asarray(x)


class ArkoudaBaseArray(ExtensionArray):
    def __init__(self, data):
        self._data = data  # Subclasses should ensure this is the correct ak object

    def __len__(self):
        return self._data.sizes

    def take(self, indexer, fill_value=None, allow_fill=False):
        indexer = indexer.to_ndarray() if hasattr(indexer, "to_ndarray") else np.asarray(indexer)

        if allow_fill:
            mask = indexer == -1
            valid_indexer = indexer.copy()
            valid_indexer[mask] = 0  # valid placeholder index

            arr = self.to_ndarray()
            taken = arr[valid_indexer]

            # Determine default fill value
            if fill_value is None:
                if self.dtype.name == "string":
                    fill_value = ""
                elif self.dtype.name == "int64":
                    fill_value = -1
                else:
                    raise ValueError("Specify fill_value explicitly for non-string/int arrays")

            taken[mask] = fill_value
        else:
            arr = self.to_ndarray()
            taken = arr[indexer]

        from arkouda.pandas.extension._arkouda_categorical_array import (
            ArkoudaCategoricalArray,
        )

        if isinstance(self, ArkoudaCategoricalArray):
            return ArkoudaCategoricalArray(ak.Categorical(ak.array(taken)))
        else:
            return type(self)(ak.array(taken))

    def _fill_missing(self, mask, fill_value):
        raise NotImplementedError("Subclasses must implement _fill_missing")

    def to_numpy(self):
        return self._data.to_ndarray()

    def to_ndarray(self):
        return self._data.to_ndarray()


def _concat_same_type(cls, to_concat):
    return cls(ak_concat([x._data for x in to_concat]))
