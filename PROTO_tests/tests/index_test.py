import pandas as pd
import pytest

import arkouda as ak
from arkouda.dtypes import dtype
from arkouda.pdarrayclass import pdarray


class TestIndex:
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_index_creation(self, size):
        idx = ak.Index(ak.arange(size))

        assert isinstance(idx, ak.Index)
        assert idx.size == size
        assert idx.to_list() == list(range(size))

    def test_index_creation_lists(self):
        i = ak.Index([1, 2, 3])
        assert isinstance(i.values, pdarray)

        i2 = ak.Index([1, 2, 3], allow_list=True)
        assert isinstance(i2.values, list)
        assert i2.dtype == dtype("int64")

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        assert isinstance(i3.values, list)
        assert i3.dtype == dtype("<U")

        with pytest.raises(ValueError):
            i4 = ak.Index([1, 2, 3], allow_list=True, max_list_size=2)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_multiindex_creation(self, size):
        # test list generation
        idx = ak.MultiIndex([ak.arange(size), ak.arange(size)])
        assert isinstance(idx, ak.MultiIndex)
        assert idx.levels == 2
        assert idx.size == size

        # test tuple generation
        idx = ak.MultiIndex((ak.arange(size), ak.arange(size)))
        assert isinstance(idx, ak.MultiIndex)
        assert idx.levels == 2
        assert idx.size == size

        with pytest.raises(TypeError):
            idx = ak.MultiIndex(ak.arange(size))

        with pytest.raises(ValueError):
            idx = ak.MultiIndex([ak.arange(size), ak.arange(size - 1)])

    def test_is_unique(self):
        i = ak.Index(ak.array([0, 1, 2]))
        assert i.is_unique

        i = ak.Index(ak.array([0, 1, 1]))
        assert not i.is_unique

        i = ak.Index([0, 1, 2], allow_list=True)
        assert i.is_unique

        i = ak.Index([0, 1, 1], allow_list=True)
        assert not i.is_unique

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_factory(self, size):
        idx = ak.Index.factory(ak.arange(size))
        assert isinstance(idx, ak.Index)

        idx = ak.Index.factory([ak.arange(size), ak.arange(size)])
        assert isinstance(idx, ak.MultiIndex)

    def test_argsort(self):
        idx = ak.Index.factory(ak.arange(5))
        i = idx.argsort(False)
        assert i.to_list() == [4, 3, 2, 1, 0]

        idx = ak.Index(ak.array([1, 0, 4, 2, 5, 3]))
        i = idx.argsort()
        # values should be the indexes in the array of idx
        assert i.to_list() == [1, 0, 3, 5, 2, 4]

        i = ak.Index([1, 2, 3])
        assert i.argsort(ascending=True).to_list() == [0, 1, 2]
        assert i.argsort(ascending=False).to_list() == [2, 1, 0]

        i2 = ak.Index([1, 2, 3], allow_list=True)
        assert i2.argsort(ascending=True) == [0, 1, 2]
        assert i2.argsort(ascending=False) == [2, 1, 0]

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        assert i3.argsort(ascending=True) == [0, 1, 2]
        assert i3.argsort(ascending=False) == [2, 1, 0]

        i4 = ak.Index(ak.array(["a", "b", "c"]))
        assert i4.argsort(ascending=True).to_list() == [0, 1, 2]
        assert i4.argsort(ascending=False).to_list() == [2, 1, 0]

    def test_concat(self):
        idx_1 = ak.Index.factory(ak.arange(5))

        idx_2 = ak.Index(ak.array([2, 4, 1, 3, 0]))

        idx_full = idx_1.concat(idx_2)
        assert idx_full.to_list() == [0, 1, 2, 3, 4, 2, 4, 1, 3, 0]

        i = ak.Index([1, 2, 3], allow_list=True)
        i2 = ak.Index(["a", "b", "c"], allow_list=True)
        assert i.concat(i2).to_list() == ["1", "2", "3", "a", "b", "c"]

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_lookup(self, size):
        idx = ak.Index.factory(ak.arange(size))
        lk = idx.lookup(ak.array([0, size - 1]))

        assert lk.to_list() == [i in [0, size - 1] for i in range(size)]

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_multi_argsort(self, size):
        idx = ak.Index.factory([ak.arange(size), ak.arange(size)])
        s = idx.argsort(False)
        assert s.to_list() == list(reversed(range(size)))

        s = idx.argsort()
        assert s.to_list() == list(range(size))

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_multi_concat(self, size):
        idx = ak.Index.factory([ak.arange(size), ak.arange(size)])
        idx_2 = ak.Index.factory(ak.arange(size) + 0.1)
        with pytest.raises(TypeError):
            idx.concat(idx_2)

        idx_2 = ak.Index.factory([ak.arange(size), ak.arange(size)])
        idx_full = idx.concat(idx_2)
        assert idx_full.to_pandas().tolist() == [(i, i) for i in range(size)] * 2

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_multi_lookup(self, size):
        idx = ak.Index.factory([ak.arange(size), ak.arange(size)])
        truth = [0, 3, 2]
        lk = ak.array(truth)
        result = idx.lookup([lk, lk])

        assert result.to_list() == [i in truth for i in range(size)]

    def test_to_pandas(self):
        i = ak.Index([1, 2, 3])
        assert i.to_pandas().equals(pd.Index([1, 2, 3]))

        i2 = ak.Index([1, 2, 3], allow_list=True)
        assert i2.to_pandas().equals(pd.Index([1, 2, 3]))

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        assert i3.to_pandas().equals(pd.Index(["a", "b", "c"]))

        i4 = ak.Index(ak.array(["a", "b", "c"]))
        assert i4.to_pandas().equals(pd.Index(["a", "b", "c"]))

    def test_to_ndarray(self):
        from numpy import array as ndarray
        from numpy import array_equal

        i = ak.Index([1, 2, 3])
        assert array_equal(i.to_ndarray(), ndarray([1, 2, 3]))

        i2 = ak.Index([1, 2, 3], allow_list=True)
        assert array_equal(i2.to_ndarray(), ndarray([1, 2, 3]))

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        assert array_equal(i3.to_ndarray(), ndarray(["a", "b", "c"]))

        i4 = ak.Index(ak.array(["a", "b", "c"]))
        assert array_equal(i4.to_ndarray(), ndarray(["a", "b", "c"]))

    def test_to_list(self):
        i = ak.Index([1, 2, 3])
        assert i.to_list() == [1, 2, 3]

        i2 = ak.Index([1, 2, 3], allow_list=True)
        assert i2.to_list() == [1, 2, 3]

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        assert i3.to_list() == ["a", "b", "c"]

        i4 = ak.Index(ak.array(["a", "b", "c"]))
        assert i4.to_list() == ["a", "b", "c"]

    def test_register_list_values(self):
        i = ak.Index([1, 2, 3], allow_list=True)
        with pytest.raises(TypeError):
            i.register("test")
