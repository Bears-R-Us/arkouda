import math
import os
from string import ascii_letters, digits
import tempfile

import numpy as np
import pytest

import arkouda as ak
from arkouda.pandas import io_util


DTYPES = [ak.int64, ak.uint64, ak.bigint, ak.float64, ak.bool_, ak.str_]
NO_BOOL = [ak.int64, ak.uint64, ak.bigint, ak.float64, ak.str_]
NO_STR = [ak.int64, ak.uint64, ak.bigint, ak.float64, ak.bool_]
NO_FLOAT = [ak.int64, ak.uint64, ak.bigint, ak.str_, ak.bool_]
NO_FLOAT_STR = [ak.int64, ak.uint64, ak.bigint, ak.bool_]
NUMERIC_DTYPES = [ak.int64, ak.uint64, ak.float64]
SETOPS = ["intersect", "union", "setdiff", "setxor"]


@pytest.fixture
def seg_test_base_tmp(request):
    seg_test_base_tmp = "{}/.seg_test".format(os.getcwd())
    io_util.get_directory(seg_test_base_tmp)

    # Define a finalizer function for teardown
    def finalizer():
        # Clean up any resources if needed
        io_util.delete_directory(seg_test_base_tmp)

    # Register the finalizer to ensure cleanup
    request.addfinalizer(finalizer)
    return seg_test_base_tmp


class TestSegArray:
    def test_segarray_docstrings(self):
        import doctest

        from arkouda.numpy import segarray

        result = doctest.testmod(segarray, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @staticmethod
    def make_segarray(size, dtype):
        segs = np.arange(0, size, 5)
        if dtype in [ak.int64, ak.uint64]:
            vals = np.random.randint(0, math.floor(size / 2), size, dtype=dtype)
        elif dtype == ak.bigint:
            vals = np.array([2**200 + i for i in range(size)])
        elif dtype == ak.float64:
            vals = np.linspace(-(size / 2), (size / 2), size)
        elif dtype == ak.str_:
            alpha_num = list(ascii_letters + digits)
            np_codes = np.random.choice(alpha_num, size=[size, 2])
            vals = np.array(["".join(code) for code in np_codes])
        elif dtype == ak.bool_:
            vals = np.random.randint(0, 2, size, dtype=dtype)
        else:
            vals = None

        return segs, vals

    @staticmethod
    def make_segarray_edge(dtype):
        """Small specific examples to test handling of empty segments."""
        segs = np.array([0, 0, 3, 5, 5, 9, 10])
        if dtype in [ak.int64, ak.uint64]:
            vals = np.random.randint(0, 3, 10, dtype=dtype)
        elif dtype == ak.bigint:
            vals = np.array([2**200 + i for i in range(10)])
        elif dtype == ak.float64:
            vals = np.linspace(-2.5, 2.5, 10)
        elif dtype == ak.str_:
            alpha_num = list(ascii_letters + digits)
            np_codes = np.random.choice(alpha_num, size=[10, 2])
            vals = np.array(["".join(code) for code in np_codes])
        elif dtype == ak.bool_:
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
            alpha_num = list(ascii_letters + digits)
            np_codes = np.random.choice(alpha_num, size=[10, 2])
            vals = np.array(["".join(code) for code in np_codes])
        elif dtype == ak.bool_:
            vals = np.random.randint(0, 2, 10, dtype=dtype)
        else:
            vals = None

        return segs, vals

    def make_append_error_checks(self, dtype):
        if dtype in [ak.int64, ak.uint64, ak.bigint, ak.bool_]:
            return self.make_concat_segarray(ak.float64)
        else:
            return self.make_concat_segarray(ak.int64)

    @staticmethod
    def build_single_append_array(size, dtype):
        if dtype in [ak.int64, ak.uint64]:
            vals = np.random.randint(0, 5, size, dtype=dtype)
        elif dtype == ak.bigint:
            vals = np.array([2**200 + i for i in range(size)])
        elif dtype == ak.float64:
            vals = np.linspace(-5, 5, size)
        elif dtype == ak.str_:
            alpha_num = list(ascii_letters + digits)
            np_codes = np.random.choice(alpha_num, size=[size, 2])
            vals = np.array(["".join(code) for code in np_codes])
        elif dtype == ak.bool_:
            vals = np.random.randint(0, 2, size, dtype=dtype)
        else:
            vals = None

        return vals

    def get_append_scalar(self, dtype):
        if dtype in [ak.int64, ak.uint64]:
            return 99
        elif dtype == ak.bigint:
            return 99 + 2**200
        elif dtype == ak.bool_:
            return False
        elif dtype == ak.float64:
            return -3.14

    @staticmethod
    def build_repeat_filter_data(dtype):
        """
        Small set to validate repeats and filters are used properly
        Only returns values as segments will change throughout test.
        """
        if dtype in [ak.int64, ak.uint64]:
            a = [1, 2, 1, 1, 3, 3, 5, 4, 6, 2]
            b = [10, 11, 11, 12, 13, 10, 4, 6, 1, 12]
        elif dtype == ak.bigint:
            a = [2**200 + i for i in [1, 2, 1, 1, 3, 3, 5, 4, 6, 2]]
            b = [2**200 + i for i in [10, 11, 11, 12, 13, 10, 4, 6, 1, 12]]
        elif dtype == ak.bool_:
            a = [True, False, True, False, False]
            b = [False, False, True, False, True]
        elif dtype == ak.float64:
            a = [3.14, 2.23, 1.01, 3.14, 3.14, 5.7, 1.01]
            b = [2.23, 1.01, 3.14, 3.14, 5.7, 1.01]
        elif dtype == ak.str_:
            a = ["abc", "123", "abc", "abc", "a"]
            b = ["a", "a", "b", "a"]

        return a, b

    @staticmethod
    def get_setops_segments(dtype):
        if dtype in [ak.int64, ak.uint64, ak.bigint]:
            a = [1, 2, 3, 1, 4]
            b = [3, 1, 4, 5]
            c = [1, 3, 3, 5]
            d = [2, 2, 4]
            if dtype == ak.bigint:
                a = [2**200 + x for x in a]
                b = [2**200 + x for x in b]
                c = [2**200 + x for x in c]
                d = [2**200 + x for x in d]
        elif dtype == ak.bool_:
            a = [True, False, True, True]
            b = [False, False, False]
            c = [True, True]
            d = [False, True, False, True, False]
        elif dtype == ak.float64:
            a = [3.14, 2.01, 5.77, 9.31]
            b = [5.1, 7.6, 3.14]
            c = [3.14, 5.77, 9.00, 6.43]
            d = [0.13, 7.6, 3.14, 7.77]
        elif dtype == ak.str_:
            a = ["a", "abc", "123", "b", "c"]
            b = ["a", "abc", "b", "123"]
            c = ["c", "abc", "x", "y"]
            d = ["abc", "b", "z"]

        return a, b, c, d

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_creation(self, size, dtype):
        segs_np, vals_np = self.make_segarray(size, dtype)
        sa = ak.SegArray(ak.array(segs_np), ak.array(vals_np))

        assert isinstance(sa, ak.SegArray)
        assert segs_np.tolist() == sa.segments.tolist()
        assert vals_np.tolist() == sa.values.tolist()
        assert sa.size == len(segs_np)
        assert sa.dtype == dtype

        expected_lens = np.concatenate((segs_np[1:], np.array([size]))) - segs_np
        assert expected_lens.tolist() == sa.lengths.tolist()

        with pytest.raises(TypeError):
            ak.SegArray(segs_np, ak.array(vals_np))

        with pytest.raises(TypeError):
            ak.SegArray(ak.array(segs_np), vals_np)

    def test_creation_empty_segment(self):
        a = [10, 11]
        b = [20, 21, 22]
        c = [30]

        # test empty as first elements
        flat = ak.array(b + c)
        segs = ak.array([0, 0, len(b)])
        segarr = ak.SegArray(segs, flat)
        assert isinstance(segarr, ak.SegArray)
        assert segarr.lengths.tolist() == [0, 3, 1]

        # test empty as middle element
        flat = ak.array(a + c)
        segs = ak.array([0, len(a), len(a)])
        segarr = ak.SegArray(segs, flat)
        assert isinstance(segarr, ak.SegArray)
        assert segarr.lengths.tolist() == [2, 0, 1]

        # test empty as last
        flat = ak.array(a + b + c)
        segs = ak.array([0, len(a), len(a) + len(b), len(a) + len(b) + len(c)])
        segarr = ak.SegArray(segs, flat)
        assert isinstance(segarr, ak.SegArray)
        assert segarr.lengths.tolist() == [2, 3, 1, 0]

    def test_empty_creation(self):
        sa = ak.SegArray(ak.array([], dtype=ak.int64), ak.array([]))

        assert isinstance(sa, ak.SegArray)
        assert sa.size == 0
        assert [] == sa.lengths.tolist()

    def test_generic_error_handling(self):
        with pytest.raises(TypeError):
            ak.SegArray([0, 5, 6], ak.arange(10))

        with pytest.raises(TypeError):
            ak.SegArray(ak.arange(0, 10, 2), [i for i in range(10)])

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
        assert segs_np.tolist() == sa.segments.tolist()
        assert vals_np.tolist() == sa.values.tolist()
        assert sa.size == len(segs_np)
        assert sa.dtype == dtype

        expected_lens = np.concatenate((segs_np[1:], np.array([10]))) - segs_np
        assert expected_lens.tolist() == sa.lengths.tolist()

        with pytest.raises(TypeError):
            ak.SegArray(segs_np, ak.array(vals_np))

        with pytest.raises(TypeError):
            ak.SegArray(ak.array(segs_np), vals_np)

    def test_multi_array_creation(self):
        ma = [
            ak.array([0, 1, 2, 3]),
            ak.array([], dtype=ak.int64),
            ak.array([4, 5]),
            ak.array([], dtype=ak.int64),
        ]
        sa = ak.SegArray.from_multi_array(ma)

        assert isinstance(sa, ak.SegArray)
        assert [0, 4, 4, 6] == sa.segments.tolist()
        assert list(range(6)) == sa.values.tolist()
        assert sa.size == 4

        expected_lens = [4, 0, 2, 0]
        assert expected_lens == sa.lengths.tolist()

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
        assert result.lengths.tolist() == (sa.lengths.tolist() + c_sa.lengths.tolist())
        assert result.segments.tolist() == np.concatenate([seg_np, c_seg + val_np.size]).tolist()
        assert result.values.tolist() == sa.values.tolist() + c_sa.values.tolist()
        assert result.tolist() == (sa.tolist() + c_sa.tolist())

        # test concat with empty segments
        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
        c_seg, c_val = self.make_segarray_edge(dtype)
        c_sa = ak.SegArray(ak.array(c_seg), ak.array(c_val))
        result = ak.SegArray.concat([sa, c_sa])
        assert isinstance(result, ak.SegArray)
        assert result.size == (sa.size + c_sa.size)
        assert result.lengths.tolist() == (sa.lengths.tolist() + c_sa.lengths.tolist())
        assert result.segments.tolist() == np.concatenate([seg_np, c_seg + val_np.size]).tolist()
        assert result.values.tolist() == sa.values.tolist() + c_sa.values.tolist()
        assert result.tolist() == (sa.tolist() + c_sa.tolist())

        # test axis=1
        if dtype != ak.str_:  # TODO - updated to run on strings once #2646 is complete
            seg_np, val_np = self.make_segarray_edge(dtype)
            sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
            c_seg, c_val = self.make_segarray_edge(dtype)
            c_sa = ak.SegArray(ak.array(c_seg), ak.array(c_val))
            result = ak.SegArray.concat([sa, c_sa], axis=1)
            assert isinstance(result, ak.SegArray)
            assert result.size == sa.size
            assert result.lengths.tolist() == (sa.lengths + c_sa.lengths).tolist()
            assert result.tolist() == [x + y for (x, y) in zip(sa.tolist(), c_sa.tolist())]

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
        assert suffix[0].tolist() == [x[-1] for x in sa.tolist()]

        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
        suffix, origin = sa.get_suffixes(2)
        assert origin.tolist() == [False, True, False, False, True, False, False]
        expected = [[s[(-2 + i)] for s in sa.tolist() if len(s) > 2] for i in range(2)]
        assert [x.tolist() for x in suffix] == expected

        suffix, origin = sa.get_suffixes(2, proper=False)
        assert origin.tolist() == [False, True, True, False, True, False, False]
        expected = [[s[(-2 + i)] for s in sa.tolist() if len(s) > 1] for i in range(2)]
        assert [x.tolist() for x in suffix] == expected

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_prefixes(self, size, dtype):
        seg_np, val_np = self.make_segarray(size, dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        prefix, origin = sa.get_prefixes(1)
        assert origin.all()
        assert prefix[0].tolist() == [x[0] for x in sa.tolist()]

        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        prefix, origin = sa.get_prefixes(2)
        assert origin.tolist() == [False, True, False, False, True, False, False]
        expected = [[s[(i)] for s in sa.tolist() if len(s) > 2] for i in range(2)]
        assert [x.tolist() for x in prefix] == expected

        prefix, origin = sa.get_prefixes(2, proper=False)
        assert origin.tolist() == [False, True, True, False, True, False, False]
        expected = [[s[(i)] for s in sa.tolist() if len(s) > 1] for i in range(2)]
        assert [x.tolist() for x in prefix] == expected

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_ngram(self, dtype):
        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        ngram, origin = sa.get_ngrams(2)
        ng_list = [x.tolist() for x in ngram]
        ng_tuple = list(zip(ng_list[0], ng_list[1]))
        exp_list = []
        exp_origin = []
        for i in range(sa.size):
            seg = sa[i]
            if len(seg) > 1:
                for j in range(len(seg) - 1):
                    exp_list.append((seg[j], seg[j + 1]))
                    exp_origin.append(i)
        assert ng_tuple == exp_list
        assert origin.tolist() == exp_origin

        with pytest.raises(ValueError):
            sa.get_ngrams(7)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NO_BOOL)  # TODO add bool processing once issue #2647 is complete
    def test_get_jth(self, size, dtype):
        seg_np, val_np = self.make_segarray(size, dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        res, origins = sa.get_jth(1)
        assert res.tolist() == [x[1] for x in sa.tolist() if 1 < len(x)]
        assert origins.tolist() == [4 < len(x) for x in sa.tolist()]

        res, origins = sa.get_jth(4)
        if dtype != ak.str_:
            assert res.tolist() == [x[4] if 4 < len(x) else 0 for x in sa.tolist()]
        else:
            assert res.tolist() == [x[4] for x in sa.tolist() if 4 < len(x)]

        res, origins = sa.get_jth(4, compressed=True)
        assert res.tolist() == [x[4] for x in sa.tolist() if 4 < len(x)]

        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        res, origins = sa.get_jth(2)
        if dtype != ak.str_:
            assert res.tolist() == [x[2] if 2 < len(x) else 0 for x in sa.tolist()]
        else:
            assert res.tolist() == [x[2] for x in sa.tolist() if 2 < len(x)]

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NO_STR)  # Strings arrays are immutable
    def test_set_jth(self, size, dtype):
        seg_np, val_np = self.make_segarray(size, dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        sa.set_jth(0, 1, 99)
        val_np[seg_np[0] + 1] = 99
        test_sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
        assert val_np.tolist() == sa.values.tolist()
        assert test_sa.tolist() == sa.tolist()

        if len(sa) > 2:
            sa.set_jth(ak.array([0, 1, 2]), 3, 17)
            val_np[seg_np[0] + 3] = 17
            val_np[seg_np[1] + 3] = 17
            val_np[seg_np[2] + 3] = 17
            test_sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
            assert val_np.tolist() == sa.values.tolist()
            assert test_sa.tolist() == sa.tolist()

        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        sa.set_jth(1, 1, 5)
        val_np[seg_np[1] + 1] = 5
        test_sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
        assert val_np.tolist() == sa.values.tolist()
        assert test_sa.tolist() == sa.tolist()

        sa.set_jth(ak.array([1, 4]), 1, 11)
        val_np[seg_np[1] + 1] = 11
        val_np[seg_np[4] + 1] = 11
        test_sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))
        assert val_np.tolist() == sa.values.tolist()
        assert test_sa.tolist() == sa.tolist()

        with pytest.raises(ValueError):
            sa.set_jth(4, 4, 999)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_get_length_n(self, dtype):
        seg_np, val_np = self.make_segarray_edge(dtype)
        sa = ak.SegArray(ak.array(seg_np), ak.array(val_np))

        elem, origin = sa.get_length_n(2)
        assert [x.tolist() for x in elem] == [[sa[2][i]] for i in range(2)]
        assert origin.tolist() == [True if sa[i].size == 2 else False for i in range(sa.size)]

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
        assert result.lengths.tolist() == (sa.lengths.tolist() + edge_sa.lengths.tolist())
        assert result.segments.tolist() == np.concatenate([seg_np, edge_seg + val_np.size]).tolist()
        assert result.values.tolist() == sa.values.tolist() + edge_sa.values.tolist()
        assert result.tolist() == (sa.tolist() + edge_sa.tolist())

        result = edge_sa.append(sa)
        assert isinstance(result, ak.SegArray)
        assert result.size == (edge_sa.size + sa.size)
        assert result.lengths.tolist() == (edge_sa.lengths.tolist() + sa.lengths.tolist())
        assert result.segments.tolist() == np.concatenate([edge_seg, seg_np + edge_val.size]).tolist()
        assert result.values.tolist() == edge_sa.values.tolist() + sa.values.tolist()
        assert result.tolist() == (edge_sa.tolist() + sa.tolist())

        # test axis=1
        if dtype != ak.str_:  # TODO - updated to run on strings once #2646 is complete
            seg_np, val_np = self.make_segarray(size, dtype)
            sa2 = ak.SegArray(ak.array(seg_np), ak.array(val_np))

            result = sa.append(sa2, axis=1)
            assert isinstance(result, ak.SegArray)
            assert result.size == sa.size
            assert result.lengths.tolist() == (sa.lengths + sa2.lengths).tolist()
            assert result.tolist() == [x + y for (x, y) in zip(sa.tolist(), sa2.tolist())]

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
        assert result.lengths.tolist() == (sa.lengths + 1).tolist()
        sa_list = sa.tolist()
        for i, s in enumerate(sa_list):
            s.append(to_append[i])
        assert result.tolist() == sa_list

        # test single value
        to_append = self.get_append_scalar(dtype)
        result = sa.append_single(to_append)
        assert result.size == sa.size
        assert result.lengths.tolist() == (sa.lengths + 1).tolist()
        sa_list = sa.tolist()
        for s in sa_list:
            s.append(to_append)
        assert result.tolist() == sa_list

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
        assert result.lengths.tolist() == (sa.lengths + 1).tolist()
        sa_list = sa.tolist()
        for i, s in enumerate(sa_list):
            s.insert(0, to_prepend[i])
        assert result.tolist() == sa_list

        # test single value
        to_prepend = self.get_append_scalar(dtype)
        result = sa.prepend_single(to_prepend)
        assert result.size == sa.size
        assert result.lengths.tolist() == (sa.lengths + 1).tolist()
        sa_list = sa.tolist()
        for s in sa_list:
            s.insert(0, to_prepend)
        assert result.tolist() == sa_list

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_remove_repeats(self, dtype):
        # Testing with small example to ensure that we are getting duplicates
        a, b = self.build_repeat_filter_data(dtype)

        exp_idx_a = np.array(a[:-1]) == np.array(a[1:])
        exp_idx_b = np.array(b[:-1]) == np.array(b[1:])
        exp_a = np.concatenate([np.array([a[0]]), np.array(a[1:])[~exp_idx_a]])
        exp_b = np.concatenate([np.array([b[0]]), np.array(b[1:])[~exp_idx_b]])

        # Test with no empty segments
        segments = ak.array([0, len(a)])
        flat = ak.array(a + b)
        sa = ak.SegArray(segments, flat)
        result = sa.remove_repeats()
        assert [0, len(exp_a)] == result.segments.tolist()
        assert np.concatenate([exp_a, exp_b]).tolist() == result.values.tolist()

        # test empty segments
        # TODO - update line below to segments = ak.array([0, 0, len(a), len(a), len(a), len(a)+len(b)])
        #  when issue #2661 is corrected. Also, needed in the first assert below
        segments = ak.array([0, len(a), len(a), len(a), len(a) + len(b)])
        flat = ak.array(a + b)
        sa = ak.SegArray(segments, flat)
        result = sa.remove_repeats()
        assert [
            0,
            len(exp_a),
            len(exp_a),
            len(exp_a),
            len(exp_a) + len(exp_b),
        ] == result.segments.tolist()
        assert np.concatenate([exp_a, exp_b]).tolist() == result.values.tolist()

    @pytest.mark.parametrize("dtype", NO_FLOAT_STR)
    @pytest.mark.parametrize("op", SETOPS)
    def test_setops(self, dtype, op):
        # TODO - string results not properly ordered. Need to add ak.str_ testing
        #  back once #2665 is worked.
        a, b, c, d = self.get_setops_segments(dtype)

        # test with no empty segments
        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + b))
        segarr_2 = ak.SegArray(ak.array([0, len(c)]), ak.array(c + d))
        sa_op = getattr(segarr, op)
        result = sa_op(segarr_2)

        np_func = getattr(np, f"{op}1d")
        exp_1 = np_func(np.array(a), np.array(c))
        exp_2 = np_func(np.array(b), np.array(d))
        assert result.segments.tolist() == [0, len(exp_1)]
        assert result.values.tolist() == np.concatenate([exp_1, exp_2]).tolist()
        exp_sa = ak.SegArray(ak.array([0, len(exp_1)]), ak.array(np.concatenate([exp_1, exp_2])))
        assert result.tolist() == exp_sa.tolist()

        # TODO - empty segments testing

    def test_segarray_load(self, seg_test_base_tmp):
        segarr = ak.SegArray(ak.array([0, 9, 14]), ak.arange(20))
        with tempfile.TemporaryDirectory(dir=seg_test_base_tmp) as tmp_dirname:
            segarr.to_hdf(f"{tmp_dirname}/seg_test.h5")

            seg_load = ak.SegArray.read_hdf(f"{tmp_dirname}/seg_test*").popitem()[1]
            assert ak.all(segarr == seg_load)

    def test_bigint(self):
        a = [2**80, 2**81]
        b = [2**82, 2**83]
        c = [2**84]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])
        segarr = ak.SegArray(segments, akflat)

        assert isinstance(segarr, ak.SegArray)
        assert segarr.lengths.tolist() == [2, 2, 1]
        assert segarr[0].tolist() == a
        assert segarr[1].tolist() == b
        assert segarr[2].tolist() == c
        assert segarr[ak.array([1, 2])].values.tolist() == b + c
        assert segarr.__eq__(ak.array([1])) == NotImplemented
        assert segarr.__eq__(segarr).all()
        assert segarr._non_empty_count == 3

    @staticmethod
    def get_filter(dtype):
        if dtype in [ak.int64, ak.uint64]:
            return 3
        elif dtype == ak.bigint:
            return 3 + 2**200
        elif dtype == ak.float64:
            return 3.14
        elif dtype == ak.bool_:
            return False
        elif dtype == ak.str_:
            return "a"

    @staticmethod
    def get_filter_list(dtype):
        if dtype in [ak.int64, ak.uint64]:
            return [1, 3]
        elif dtype == ak.bigint:
            return [1 + 2**200, 3 + 2**200]
        elif dtype == ak.float64:
            return [3.14, 1.01]
        elif dtype == ak.bool_:
            return [False]
        elif dtype == ak.str_:
            return ["a", "abc"]

    @pytest.mark.parametrize("dtype", [ak.int64])
    def test_filter(self, dtype):
        # TODO - once #2666 is resolved, this test will need to be updated for the SegArray
        #  being filtered containing empty segments prior to filter
        a, b = self.build_repeat_filter_data(dtype)
        sa = ak.SegArray(ak.array([0, len(a)]), ak.array(a + b))

        # test filtering single value retain empties
        f = self.get_filter(dtype)
        filter_result = sa.filter(f, discard_empty=False)
        assert sa.size == filter_result.size
        # ensure 2 does not exist in return values
        assert (filter_result.values != f).all()
        for i in range(sa.size):
            assert sa[i][(sa[i] != f)].tolist() == filter_result[i].tolist()

        # test list filter
        fl = self.get_filter_list(dtype)
        filter_result = sa.filter(fl, discard_empty=False)
        assert sa.size == filter_result.size
        # ensure 1 & 2 do not exist in return values
        assert (filter_result.values != fl[0]).all()
        assert (filter_result.values != fl[1]).all()
        for i in range(sa.size):
            x = ak.in1d(ak.array(sa[i]), ak.array(fl), invert=True)
            v = ak.array(sa[i])[x]
            assert v.tolist() == filter_result[i].tolist()

        # test pdarray filter
        filter_result = sa.filter(ak.array(fl), discard_empty=False)
        assert sa.size == filter_result.size
        # ensure 1 & 2 do not exist in return values
        assert (filter_result.values != fl[0]).all()
        assert (filter_result.values != fl[1]).all()
        for i in range(sa.size):
            x = ak.in1d(ak.array(sa[i]), ak.array(fl), invert=True)
            v = ak.array(sa[i])[x]
            assert v.tolist() == filter_result[i].tolist()

        # test dropping empty segments
        fl = list(set(a))
        filter_result = sa.filter(ak.array(fl), discard_empty=True)
        # ensure 1 & 2 do not exist in return values
        assert (filter_result.values != fl[0]).all()
        assert (filter_result.values != fl[1]).all()
        offset = 0
        for i in range(sa.size):
            x = ak.in1d(ak.array(sa[i]), ak.array(fl), invert=True)
            v = ak.array(sa[i])[x]
            if v.size != 0:
                assert v.tolist() == filter_result[i - offset].tolist()
            else:
                offset += 1

    def test_equality(self):
        # reproducer for issue #2617
        # verify equality no matter position of empty seg
        for has_empty_seg in (
            [0, 0, 9, 14],
            [0, 9, 9, 14, 14],
            [0, 0, 7, 9, 14, 14, 17, 20],
        ):
            sa = ak.SegArray(ak.array(has_empty_seg), ak.arange(-10, 10))
            assert (sa == sa).all()

        s1 = ak.SegArray(ak.array([0, 4, 14, 14]), ak.arange(-10, 10))
        s2 = ak.SegArray(ak.array([0, 9, 14, 14]), ak.arange(-10, 10))
        assert (s1 == s2).tolist() == [False, False, True, True]

        # test segarrays with empty segments, multiple types, and edge cases
        df = ak.DataFrame(
            {
                "c_1": ak.SegArray(ak.array([0, 0, 9, 14]), ak.arange(-10, 10)),
                "c_2": ak.SegArray(
                    ak.array([0, 5, 10, 10]),
                    ak.arange(2**63, 2**63 + 15, dtype=ak.uint64),
                ),
                "c_3": ak.SegArray(ak.array([0, 0, 5, 10]), ak.randint(0, 1, 15, dtype=ak.bool_)),
                "c_4": ak.SegArray(
                    ak.array([0, 9, 14, 14]),
                    ak.array(
                        [
                            np.nan,
                            np.finfo(np.float64).min,
                            -np.inf,
                            -7.0,
                            -3.14,
                            -0.0,
                            0.0,
                            3.14,
                            7.0,
                            np.finfo(np.float64).max,
                            np.inf,
                            np.nan,
                            np.nan,
                            np.nan,
                        ]
                    ),
                ),
                "c_5": ak.SegArray(
                    ak.array([0, 2, 5, 5]),
                    ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
                ),
                "c_6": ak.SegArray(
                    ak.array([0, 2, 2, 2]),
                    ak.array(["a", "b", "", "c", "d", "e", "f", "g", "h", "i"]),
                ),
                "c_7": ak.SegArray(
                    ak.array([0, 0, 2, 2]),
                    ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
                ),
                "c_8": ak.SegArray(
                    ak.array([0, 2, 3, 3]),
                    ak.array(["", "'", " ", "test", "", "'", "", " ", ""]),
                ),
                "c_9": ak.SegArray(
                    ak.array([0, 5, 5, 8]),
                    ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
                ),
                "c_10": ak.SegArray(
                    ak.array([0, 5, 8, 8]),
                    ak.array(["abc", "123", "xyz", "l", "m", "n", "o", "p", "arkouda"]),
                ),
            }
        )

        for col in df.columns:
            a = df[col]
            if a.dtype == ak.float64:
                a = a.to_ndarray()
                if isinstance(a[0], np.ndarray):
                    assert all(np.allclose(a1, b1, equal_nan=True) for a1, b1 in zip(a, a))
                else:
                    assert np.allclose(a, a, equal_nan=True)
            else:
                assert (a == a).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_mean(self, size, dtype):
        c_segs, c_vals = self.make_segarray(size, dtype)
        c_sa = ak.SegArray(ak.array(c_segs), ak.array(c_vals))
        for i in range(len(c_sa)):
            assert math.isclose(c_sa.mean()[i], c_sa[i].mean())
