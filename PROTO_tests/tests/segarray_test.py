import os
import tempfile
import math
import pytest
from arkouda import io_util
import arkouda as ak
import numpy as np


DTYPES = [ak.int64, ak.uint64, ak.bigint, ak.float64, ak.bool, ak.str_]
NO_BOOL = [ak.int64, ak.uint64, ak.bigint, ak.float64, ak.str_]
NO_STR = [ak.int64, ak.uint64, ak.bigint, ak.float64, ak.bool]

class TestSegArray:
    @staticmethod
    def make_segarray(size, dtype):
        segs = np.arange(0, size, 5)
        if dtype in [ak.int64, ak.uint64]:
            vals = np.random.randint(0, math.floor(size / 2), size, dtype=dtype)
        elif dtype == ak.bigint:
            vals = np.array([2**200 + i for i in range(size)])
        elif dtype == ak.float64:
            vals = np.linspace(-(size/2), (size/2), size)
        elif dtype == ak.str_:
            alpha_num = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            np_codes = np.random.choice(alpha_num, size=[size, 2])
            vals = np.array([''.join(code) for code in np_codes])
        elif dtype == ak.bool:
            vals = np.random.randint(0, 2, size, dtype=dtype)
        else:
            vals = None

        return segs, vals

    @staticmethod
    def make_segarray_edge(dtype):
        """
        Small specific examples to test handling of empty segments
        """
        segs = np.array([0, 0, 3, 5, 5, 9, 10])
        if dtype in [ak.int64, ak.uint64]:
            vals = np.random.randint(0, 5, 10, dtype=dtype)
        elif dtype == ak.bigint:
            vals = np.array([2**200 + i for i in range(10)])
        elif dtype == ak.float64:
            vals = np.linspace(-5, 5, 10)
        elif dtype == ak.str_:
            alpha_num = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            np_codes = np.random.choice(alpha_num, size=[10, 2])
            vals = np.array([''.join(code) for code in np_codes])
        elif dtype == ak.bool:
            vals = np.random.randint(0, 2, 10, dtype=dtype)
        else:
            vals = None

        return segs, vals

    @staticmethod
    def make_concat_segarray(dtype):
        segs = np.arange(0, 10, 2)
        if dtype in [ak.int64, ak.uint64]:
            vals = np.random.randint(0, 5, 10, dtype=dtype)
        elif dtype == ak.bigint:
            vals = np.array([2**200 + i for i in range(10)])
        elif dtype == ak.float64:
            vals = np.linspace(-5, 5, 10)
        elif dtype == ak.str_:
            alpha_num = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            np_codes = np.random.choice(alpha_num, size=[10, 2])
            vals = np.array([''.join(code) for code in np_codes])
        elif dtype == ak.bool:
            vals = np.random.randint(0, 2, 10, dtype=dtype)
        else:
            vals = None

        return segs, vals

    def make_append_error_checks(self, dtype):
        if dtype in [ak.int64, ak.uint64, ak.bigint, ak.bool]:
            return self.make_concat_segarray(ak.float64)
        else:
            return self.make_concat_segarray(ak.int64)

    def build_single_append_array(self, size, dtype):
        if dtype in [ak.int64, ak.uint64]:
            vals = np.random.randint(0, 5, size, dtype=dtype)
        elif dtype == ak.bigint:
            vals = np.array([2**200 + i for i in range(size)])
        elif dtype == ak.float64:
            vals = np.linspace(-5, 5, size)
        elif dtype == ak.str_:
            alpha_num = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            np_codes = np.random.choice(alpha_num, size=[size, 2])
            vals = np.array([''.join(code) for code in np_codes])
        elif dtype == ak.bool:
            vals = np.random.randint(0, 2, size, dtype=dtype)
        else:
            vals = None

        return vals

    def get_append_scalar(self, dtype):
        if dtype in [ak.int64, ak.uint64]:
            return 99
        elif dtype == ak.bigint:
            return 99 + 2**200
        elif dtype == ak.bool:
            return False
        elif dtype == ak.float64:
            return -3.14

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_creation(self, size, dtype):
        segs_np, vals_np = self.make_segarray(size, dtype)
        sa = ak.SegArray(ak.array(segs_np), ak.array(vals_np))

        assert isinstance(sa, ak.SegArray)
        assert segs_np.tolist() == sa.segments.to_list()
        assert vals_np.tolist() == sa.values.to_list()
        assert sa.size == len(segs_np)
        assert sa.dtype == dtype

        expected_lens = np.concatenate((segs_np[1:], np.array([size]))) - segs_np
        assert expected_lens.tolist() == sa.lengths.to_list()

        with pytest.raises(TypeError):
            ak.SegArray(segs_np, ak.array(vals_np))

        with pytest.raises(TypeError):
            ak.SegArray(ak.array(segs_np), vals_np)

    def test_empty_creation(self):
        sa = ak.SegArray(ak.array([], dtype=ak.int64), ak.array([]))

        assert isinstance(sa, ak.SegArray)
        assert sa.size == 0
        assert [] == sa.lengths.to_list()

    def test_generic_error_handling(self):
        with pytest.raises(TypeError):
            ak.SegArray([0, 5, 6], ak.arange(10))

        with pytest.raises(TypeError):
            ak.SegArray(ak.arange(0, 10, 2), [i for i in range(10)])

        with pytest.raises(ValueError):
            ak.SegArray(ak.array([0, 1, 4, 3]), ak.arange(10))

        with pytest.raises(ValueError):
            ak.SegArray(ak.array([0, 1, 4, 3]), ak.arange(10))

        with pytest.raises(ValueError):
            ak.SegArray(ak.array([1, 4, 3]), ak.arange(10))

        with pytest.raises(ValueError):
            ak.SegArray(ak.array([], dtype=ak.int64), ak.arange(10))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_creation_edge_case(self, dtype):
        segs_np, vals_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(segs_np), ak.array(vals_np))

        assert isinstance(sa, ak.SegArray)
        assert segs_np.tolist() == sa.segments.to_list()
        assert vals_np.tolist() == sa.values.to_list()
        assert sa.size == len(segs_np)
        assert sa.dtype == dtype

        expected_lens = np.concatenate((segs_np[1:], np.array([10]))) - segs_np
        assert expected_lens.tolist() == sa.lengths.to_list()

        with pytest.raises(TypeError):
            ak.SegArray(segs_np, ak.array(vals_np))

        with pytest.raises(TypeError):
            ak.SegArray(ak.array(segs_np), vals_np)

    def test_multi_array_creation(self):
        ma = [
            ak.array([0, 1, 2, 3]),
            ak.array([], dtype=ak.int64),
            ak.array([4, 5]),
            ak.array([], dtype=ak.int64)
        ]
        sa = ak.SegArray.from_multi_array(ma)

        assert isinstance(sa, ak.SegArray)
        assert [0, 4, 4, 6] == sa.segments.to_list()
        assert [i for i in range(6)] == sa.values.to_list()
        assert sa.size == 4

        expected_lens = [4, 0, 2, 0]
        assert expected_lens == sa.lengths.to_list()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_concat(self, size, dtype):
        seg_np, val_np = self.make_segarray(size, dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        c_seg, c_val = self.make_concat_segarray(dtype)
        c_sa = ak.SegArray(ak.array(c_seg), ak.array(c_val))

        result = ak.SegArray.concat([sa, c_sa])
        assert isinstance(result, ak.SegArray)
        assert result.size == (sa.size + c_sa.size)
        assert result.lengths.to_list() == (sa.lengths.to_list() + c_sa.lengths.to_list())
        assert result.segments.to_list() == np.concatenate([seg_np, c_seg + val_np.size]).tolist()
        assert result.values.to_list() == sa.values.to_list() + c_sa.values.to_list()
        assert result.to_list() == (sa.to_list() + c_sa.to_list())

        # test concat with empty segments
        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
        c_seg, c_val = self.make_segarray_edge(dtype)
        c_sa = ak.SegArray(ak.array(c_seg), ak.array(c_val))
        result = ak.SegArray.concat([sa, c_sa])
        assert isinstance(result, ak.SegArray)
        assert result.size == (sa.size + c_sa.size)
        assert result.lengths.to_list() == (sa.lengths.to_list() + c_sa.lengths.to_list())
        assert result.segments.to_list() == np.concatenate([seg_np, c_seg + val_np.size]).tolist()
        assert result.values.to_list() == sa.values.to_list() + c_sa.values.to_list()
        assert result.to_list() == (sa.to_list() + c_sa.to_list())

        # test axis=1
        if dtype != ak.str_:  # TODO - updated to run on strings once #2646 is complete
            seg_np, val_np = self.make_segarray_edge(dtype)
            sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
            c_seg, c_val = self.make_segarray_edge(dtype)
            c_sa = ak.SegArray(ak.array(c_seg), ak.array(c_val))
            result = ak.SegArray.concat([sa, c_sa], axis=1)
            assert isinstance(result, ak.SegArray)
            assert result.size == sa.size
            assert result.lengths.to_list() == [x+y for (x, y) in zip(sa.lengths.to_list(), c_sa.lengths.to_list())]
            assert result.to_list() == [x + y for (x, y) in zip(sa.to_list(), c_sa.to_list())]

    def test_concat_error_handling(self):
        sa_1 = ak.SegArray(ak.arange(0, 10, 2), ak.arange(10))
        sa_2 = ak.SegArray(ak.arange(0, 10, 2), ak.arange(10))
        with pytest.raises(ValueError):
            ak.SegArray.concat([sa_1, sa_2], ordered=False)

        with pytest.raises(ValueError):
            ak.SegArray.concat([])

        assert ak.SegArray.concat([ak.array([1, 2])]) == NotImplemented

        sa_1 = ak.SegArray(ak.arange(0, 10, 2), ak.arange(10))
        sa_2 = ak.SegArray(ak.arange(0, 20, 2), ak.arange(20))
        with pytest.raises(ValueError):
            ak.SegArray.concat([sa_1, sa_2], axis=1)

        with pytest.raises(ValueError):
            ak.SegArray.concat([sa_1, sa_2], axis=5)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_suffix(self, size, dtype):
        seg_np, val_np = self.make_segarray(size, dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        suffix, origin = sa.get_suffixes(1)
        assert origin.all()
        assert suffix[0].to_list() == [x[-1] for x in sa.to_list()]

        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
        suffix, origin = sa.get_suffixes(2)
        assert origin.to_list() == [False, True, False, False, True, False, False]
        sa_list = sa.to_list()
        expected = [[s[(-2+i)] for s in sa_list if len(s) > 2] for i in range(2)]
        assert [x.to_list() for x in suffix] == expected

        suffix, origin = sa.get_suffixes(2, proper=False)
        assert origin.to_list() == [False, True, True, False, True, False, False]
        sa_list = sa.to_list()
        expected = [[s[(-2 + i)] for s in sa_list if len(s) > 1] for i in range(2)]
        assert [x.to_list() for x in suffix] == expected

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_prefixes(self, size, dtype):
        seg_np, val_np = self.make_segarray(size, dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        prefix, origin = sa.get_prefixes(1)
        assert origin.all()
        assert prefix[0].to_list() == [x[0] for x in sa.to_list()]

        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        prefix, origin = sa.get_prefixes(2)
        assert origin.to_list() == [False, True, False, False, True, False, False]
        sa_list = sa.to_list()
        expected = [[s[(0 + i)] for s in sa_list if len(s) > 2] for i in range(2)]
        assert [x.to_list() for x in prefix] == expected

        prefix, origin = sa.get_prefixes(2, proper=False)
        assert origin.to_list() == [False, True, True, False, True, False, False]
        sa_list = sa.to_list()
        expected = [[s[(0 + i)] for s in sa_list if len(s) > 1] for i in range(2)]
        assert [x.to_list() for x in prefix] == expected

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_ngram(self, dtype):
        # TODO - come back to this
        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
        sa_list = sa.to_list()

        ngram, origin = sa.get_ngrams(2)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NO_BOOL)  # TODO add bool processing once issue #2647 is complete
    def test_get_jth(self, size, dtype):
        seg_np, val_np = self.make_segarray(size, dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        res, origins = sa.get_jth(1)
        assert res.to_list() == [x[1] for x in sa.to_list() if 1 < len(x)]

        res, origins = sa.get_jth(4)
        if dtype != ak.str_:
            assert res.to_list() == [x[4] if 4 < len(x) else 0 for x in sa.to_list()]
        else:
            assert res.to_list() == [x[4] for x in sa.to_list() if 4 < len(x)]

        res, origins = sa.get_jth(4, compressed=True)
        assert res.to_list() == [x[4] for x in sa.to_list() if 4 < len(x)]

        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        res, origins = sa.get_jth(2)
        if dtype != ak.str_:
            assert res.to_list() == [x[2] if 2 < len(x) else 0 for x in sa.to_list()]
        else:
            assert res.to_list() == [x[2] for x in sa.to_list() if 2 < len(x)]

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NO_STR)  # Strings arrays are immutable
    def test_set_jth(self, size, dtype):
        seg_np, val_np = self.make_segarray(size, dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        sa.set_jth(0, 1, 99)
        val_np[seg_np[0]+1] = 99
        test_sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
        assert val_np.tolist() == sa.values.to_list()
        assert test_sa.to_list() == sa.to_list()

        sa.set_jth(ak.array([0, 1, 2]), 3, 17)
        val_np[seg_np[0]+3] = 17
        val_np[seg_np[1]+3] = 17
        val_np[seg_np[2]+3] = 17
        test_sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
        assert val_np.tolist() == sa.values.to_list()
        assert test_sa.to_list() == sa.to_list()

        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        sa.set_jth(1, 1, 5)
        val_np[seg_np[1] + 1] = 5
        test_sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
        assert val_np.tolist() == sa.values.to_list()
        assert test_sa.to_list() == sa.to_list()

        sa.set_jth(ak.array([1, 4]), 1, 11)
        val_np[seg_np[1] + 1] = 11
        val_np[seg_np[4] + 1] = 11
        test_sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
        assert val_np.tolist() == sa.values.to_list()
        assert test_sa.to_list() == sa.to_list()

        with pytest.raises(ValueError):
            sa.set_jth(4, 4, 999)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_get_length_n(self, dtype):
        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        elem, origin = sa.get_length_n(2)
        assert [x.to_list() for x in elem] == [[sa[2][i]] for i in range(2)]
        assert origin.to_list() == [True if sa[i].size == 2 else False for i in range(sa.size)]

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_append(self, size, dtype):
        seg_np, val_np = self.make_segarray(size, dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        err_seg, err_val = self.make_append_error_checks(dtype)
        assert sa.append(ak.array(err_val)) == NotImplemented
        err_sa = ak.SegArray(ak.array(err_seg), ak.array(err_val))
        with pytest.raises(TypeError):
            sa.append(err_sa)

        edge_seg, edge_val = self.make_segarray_edge(dtype)
        edge_sa = ak.SegArray(ak.array(edge_seg), ak.array(edge_val))

        result = sa.append(edge_sa)

        assert isinstance(result, ak.SegArray)
        assert result.size == (sa.size + edge_sa.size)
        assert result.lengths.to_list() == (sa.lengths.to_list() + edge_sa.lengths.to_list())
        assert result.segments.to_list() == np.concatenate([seg_np, edge_seg + val_np.size]).tolist()
        assert result.values.to_list() == sa.values.to_list() + edge_sa.values.to_list()
        assert result.to_list() == (sa.to_list() + edge_sa.to_list())

        result = edge_sa.append(sa)
        assert isinstance(result, ak.SegArray)
        assert result.size == (edge_sa.size + sa.size)
        assert result.lengths.to_list() == (edge_sa.lengths.to_list() + sa.lengths.to_list())
        assert result.segments.to_list() == np.concatenate([edge_seg, seg_np + edge_val.size]).tolist()
        assert result.values.to_list() == edge_sa.values.to_list() + sa.values.to_list()
        assert result.to_list() == (edge_sa.to_list() + sa.to_list())

        # test axis=1
        if dtype != ak.str_:  # TODO - updated to run on strings once #2646 is complete
            seg_np, val_np = self.make_segarray(size, dtype)
            sa2 = ak.SegArray(ak.array(seg_np), ak.array(val_np))

            result = sa.append(sa2, axis=1)
            assert isinstance(result, ak.SegArray)
            assert result.size == sa.size
            assert result.lengths.to_list() == [x + y for (x, y) in zip(sa.lengths.to_list(), sa2.lengths.to_list())]
            assert result.to_list() == [x + y for (x, y) in zip(sa.to_list(), sa2.to_list())]

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NO_STR)
    def test_append_single(self, size, dtype):
        # TODO - add testing for empty segments with issue #2650
        seg_np, val_np = self.make_segarray(size, dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
        to_append = self.build_single_append_array(sa.size, dtype)
        result = sa.append_single(ak.array(to_append))

        assert isinstance(result, ak.SegArray)
        assert result.size == sa.size
        assert result.lengths.to_list() == (sa.lengths + 1).to_list()
        sa_list = sa.to_list()
        for i, s in zip(range(len(sa_list)), sa_list):
            s.append(to_append[i])
        assert result.to_list() == sa_list

        # test single value
        to_append = self.get_append_scalar(dtype)
        result = sa.append_single(to_append)
        assert result.size == sa.size
        assert result.lengths.to_list() == (sa.lengths + 1).to_list()
        sa_list = sa.to_list()
        for i, s in zip(range(len(sa_list)), sa_list):
            s.append(to_append)
        assert result.to_list() == sa_list

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NO_STR)
    def test_prepend_single(self, size, dtype):
        # TODO - add testing for empty segments with issue #2650
        seg_np, val_np = self.make_segarray(size, dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        to_prepend = self.build_single_append_array(sa.size, dtype)
        result = sa.prepend_single(ak.array(to_prepend))

        assert isinstance(result, ak.SegArray)
        assert result.size == sa.size
        assert result.lengths.to_list() == (sa.lengths + 1).to_list()
        sa_list = sa.to_list()
        for i, s in zip(range(len(sa_list)), sa_list):
            s.insert(0, to_prepend[i])
        assert result.to_list() == sa_list

        # test single value
        to_prepend = self.get_append_scalar(dtype)
        result = sa.prepend_single(to_prepend)
        assert result.size == sa.size
        assert result.lengths.to_list() == (sa.lengths + 1).to_list()
        sa_list = sa.to_list()
        for i, s in zip(range(len(sa_list)), sa_list):
            s.insert(0, to_prepend)
        assert result.to_list() == sa_list

    def test_remove_repeats(self):
        # Testing with small example to ensure that we are getting duplicates
        return

