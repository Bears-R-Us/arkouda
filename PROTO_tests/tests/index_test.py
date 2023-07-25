import pytest

import arkouda as ak


class TestIndex:
    def test_index_creation(self):
        idx = ak.Index(ak.arange(5))

        assert isinstance(idx, ak.Index)
        assert idx.size == 5
        assert idx.to_list() == list(range(5))

    def test_multiindex_creation(self):
        # test list generation
        idx = ak.MultiIndex([ak.arange(5), ak.arange(5)])
        assert isinstance(idx, ak.MultiIndex)
        assert idx.levels == 2
        assert idx.size == 5

        # test tuple generation
        idx = ak.MultiIndex((ak.arange(5), ak.arange(5)))
        assert isinstance(idx, ak.MultiIndex)
        assert idx.levels == 2
        assert idx.size == 5

        with pytest.raises(TypeError):
            idx = ak.MultiIndex(ak.arange(5))

        with pytest.raises(ValueError):
            idx = ak.MultiIndex([ak.arange(5), ak.arange(3)])

    def test_is_unique(self):
        i = ak.Index(ak.array([0, 1, 2]))
        assert i.is_unique

        i = ak.Index(ak.array([0, 1, 1]))
        assert not i.is_unique

    def test_factory(self):
        idx = ak.Index.factory(ak.arange(5))
        assert isinstance(idx, ak.Index)

        idx = ak.Index.factory([ak.arange(5), ak.arange(5)])
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

    def test_lookup(self):
        idx = ak.Index.factory(ak.arange(5))
        lk = idx.lookup(ak.array([0, 4]))
        assert lk.to_list() == [True, False, False, False, True]

    def test_multi_argsort(self):
        idx = ak.Index.factory([ak.arange(5), ak.arange(5)])
        s = idx.argsort(False)
        assert s.to_list() == list(reversed(range(5)))

        s = idx.argsort()
        assert s.to_list() == list(range(5))

    def test_multi_concat(self):
        idx = ak.Index.factory([ak.arange(5), ak.arange(5)])
        idx_2 = ak.Index.factory(ak.array([0.1, 1.1, 2.2, 3.3, 4.4]))
        with pytest.raises(TypeError):
            idx.concat(idx_2)

        idx_2 = ak.Index.factory([ak.arange(5), ak.arange(5)])
        idx_full = idx.concat(idx_2)
        ans = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        assert idx_full.to_pandas().tolist() == ans

    def test_multi_lookup(self):
        idx = ak.Index.factory([ak.arange(5), ak.arange(5)])

        lk = ak.array([0, 3, 2])

        result = idx.lookup([lk, lk])
        assert result.to_list() == [True, False, True, True, False]
