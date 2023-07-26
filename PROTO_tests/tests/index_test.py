import pytest

import arkouda as ak


class TestIndex:
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_index_creation(self, size):
        idx = ak.Index(ak.arange(size))

        assert isinstance(idx, ak.Index)
        assert idx.size == size
        assert idx.to_list() == list(range(size))

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

    def test_concat(self):
        idx_1 = ak.Index.factory(ak.arange(5))

        idx_2 = ak.Index(ak.array([2, 4, 1, 3, 0]))

        idx_full = idx_1.concat(idx_2)
        assert idx_full.to_list() == [0, 1, 2, 3, 4, 2, 4, 1, 3, 0]

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
