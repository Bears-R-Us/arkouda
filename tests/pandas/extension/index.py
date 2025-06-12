import pandas as pd

import arkouda as ak
from arkouda.pandas.extension._arkouda_array import ArkoudaArray
from arkouda.pandas.extension._arkouda_string_array import ArkoudaStringArray


class TestIndexExtension:
    def test_index_basic_ops(self):
        idx = pd.Index(ArkoudaArray(ak.array([10, 20, 30])), name="foo")
        assert len(idx) == 3
        assert idx.name == "foo"
        assert idx[0] == 10
        assert list(idx[:2].to_numpy()) == [10, 20]

    def test_index_equality(self):
        a = pd.Index(ArkoudaArray(ak.array([1, 2, 3])))
        b = pd.Index(ArkoudaArray(ak.array([1, 2, 3])))
        c = pd.Index(ArkoudaArray(ak.array([3, 2, 1])))

        assert a.equals(b)
        assert not a.equals(c)

    def test_multiindex_construction_and_get_level_values(self):
        idx1 = ArkoudaArray(ak.array([1, 2, 3]))
        idx2 = ArkoudaStringArray(ak.array(["a", "b", "c"]))

        mi = pd.MultiIndex.from_arrays([idx1, idx2], names=["num", "letter"])

        assert mi.nlevels == 2
        assert mi.names == ["num", "letter"]

        lv0 = mi.get_level_values(0)
        lv1 = mi.get_level_values(1)

        assert lv0.tolist() == [1, 2, 3]
        assert lv1.tolist() == ["a", "b", "c"]

    def test_multiindex_equals_and_length(self):
        idx1a = ArkoudaArray(ak.array([1, 2]))
        idx2a = ArkoudaStringArray(ak.array(["x", "y"]))
        idx1b = ArkoudaArray(ak.array([1, 2]))
        idx2b = ArkoudaStringArray(ak.array(["x", "y"]))

        mi1 = pd.MultiIndex.from_arrays([idx1a, idx2a])
        mi2 = pd.MultiIndex.from_arrays([idx1b, idx2b])

        assert mi1.equals(mi2)
        assert len(mi1) == 2
