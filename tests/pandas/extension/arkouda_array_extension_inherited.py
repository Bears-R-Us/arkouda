import numpy as np
import pandas as pd
import pytest

from pandas.api.extensions import ExtensionArray as PandasExtensionArray

import arkouda as ak

from arkouda import Categorical
from arkouda.pandas.extension import (
    ArkoudaArray,
    ArkoudaCategoricalArray,
    ArkoudaStringArray,
)


class TestArkoudaArrayExplodeInherited:
    def test_explode_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should *not* override `_explode`; it should inherit the
        default implementation from pandas.api.extensions.ExtensionArray.

        This locks in the fact that explode semantics are coming from pandas,
        not Arkouda-specific code.
        """
        # Not defined directly on ArkoudaArray
        assert "_explode" not in ArkoudaArray.__dict__

        # Method object is exactly the base ExtensionArray implementation
        assert ArkoudaArray._explode is PandasExtensionArray._explode

    @pytest.mark.parametrize(
        "ea_cls, values",
        [
            # Numeric ArkoudaArray examples
            (ArkoudaArray, [1, 2, 3]),
            (ArkoudaArray, [10.5, 20.5]),
            # String-backed example
            (ArkoudaStringArray, ["a", "b", "c"]),
            # Categorical-backed example
            (ArkoudaCategoricalArray, ["red", "blue", "red"]),
        ],
    )
    def test_explode_scalar_data_roundtrip(self, ea_cls, values):
        """
        For arrays that do NOT contain list-like elements, the default
        ExtensionArray._explode implementation should:

        - return a copy of the array (same values, same type),
        - return lengths as an ndarray of ones with shape (len(arr),).

        This should hold for ArkoudaArray, ArkoudaStringsArray, and
        ArkoudaCategoricalArray when they wrap scalar (non-list-like) data.
        """
        # Construct the underlying Arkouda data depending on EA type
        if ea_cls is ArkoudaCategoricalArray:
            # Categorical typically wraps an Arkouda Categorical built from values
            ak_data = Categorical(ak.array(values))
        else:
            # Numeric or string cases: ak.array should produce the right Arkouda type
            ak_data = ak.array(values)

        arr = ea_cls(ak_data)

        exploded, lengths = arr._explode()

        # 1) Same EA type
        assert isinstance(exploded, type(arr))

        # 2) Values unchanged
        orig_np = arr.to_numpy()
        exploded_np = exploded.to_numpy()
        np.testing.assert_array_equal(exploded_np, orig_np)

        # 3) lengths: ndarray of ones, one entry per original element
        assert isinstance(lengths, np.ndarray)
        assert lengths.shape == (len(arr),)
        assert np.issubdtype(lengths.dtype, np.integer)
        assert np.all(lengths == 1)

    @pytest.mark.parametrize(
        "ea_cls, values",
        [
            (ArkoudaArray, [1, 2, 3]),
            (ArkoudaStringArray, ["x", "y", "z"]),
            (ArkoudaCategoricalArray, ["low", "med", "high"]),
        ],
    )
    def test_explode_does_not_change_series_values(self, ea_cls, values):
        """
        Series.explode for non-list-like, non-object extension dtypes is
        effectively a no-op: it returns a Series copy with identical data
        and index. Arkouda-backed Series of all three EA types should
        follow that behavior.
        """
        if ea_cls is ArkoudaCategoricalArray:
            ak_data = Categorical(ak.array(values))
        else:
            ak_data = ak.array(values)

        arr = ea_cls(ak_data)
        s = pd.Series(arr)

        result = s.explode()

        # Expect a value-wise identical Series.
        # (dtype may differ depending on pandas internals, so we focus on
        # values and index equality.)
        assert list(result.index) == list(s.index)
        np.testing.assert_array_equal(result.to_numpy(), s.to_numpy())
