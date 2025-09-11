import pytest

import arkouda as ak


DATA_TYPES = [ak.int64, ak.uint64, ak.float64]


class TestAlignment:
    def test_alignment_docstrings(self):
        import doctest

        from arkouda import alignment

        result = doctest.testmod(alignment, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @staticmethod
    def get_interval_info(lower_bound, upper_bound, vals, dtype):
        lb = ak.array(lower_bound, dtype)
        ub = ak.array(upper_bound, dtype)
        v = ak.array(vals, dtype)

        if dtype == ak.uint64:
            lb += 2**63
            ub += 2**63
            v += 2**63
        elif dtype == ak.float64:
            lb += 0.5
            ub += 0.5
            v += 0.5
        return lb, ub, v

    @pytest.mark.parametrize("dtype", DATA_TYPES)
    def test_search_interval(self, dtype):
        expected_result = [2, 5, 4, 0, 3, 1, 4, -1, -1]
        lb = [0, 10, 20, 30, 40, 50]
        ub = [9, 19, 29, 39, 49, 59]
        v = [22, 51, 44, 1, 38, 19, 40, 60, 100]

        lower_bound, upper_bound, vals = self.get_interval_info(lb, ub, v, dtype)
        interval_idxs = ak.search_intervals(vals, (lower_bound, upper_bound))
        assert expected_result == interval_idxs.tolist()

    @pytest.mark.parametrize("dtype", DATA_TYPES)
    def test_multi_array_search_interval(self, dtype):
        # Added for Issue #1548
        starts = (ak.array([0, 10, 20], dtype), ak.array([0, 10, 20], dtype))
        ends = (ak.array([4, 14, 24], dtype), ak.array([4, 14, 24], dtype))
        vals = (ak.array([3, 13, 23], dtype), ak.array([23, 13, 3], dtype))
        ans = [-1, 1, -1]
        assert ans == ak.search_intervals(vals, (starts, ends), hierarchical=False).tolist()
        assert ans == ak.interval_lookup((starts, ends), ak.arange(3), vals).tolist()

        vals = (ak.array([23, 13, 3], dtype), ak.array([23, 13, 3], dtype))
        ans = [2, 1, 0]
        assert ans == ak.search_intervals(vals, (starts, ends), hierarchical=False).tolist()
        assert ans == ak.interval_lookup((starts, ends), ak.arange(3), vals).tolist()

        vals = (ak.array([23, 13, 33], dtype), ak.array([23, 13, 3], dtype))
        ans = [2, 1, -1]
        assert ans == ak.search_intervals(vals, (starts, ends), hierarchical=False).tolist()
        assert ans == ak.interval_lookup((starts, ends), ak.arange(3), vals).tolist()

        # test hierarchical flag
        starts = (ak.array([0, 5], dtype), ak.array([0, 11], dtype))
        ends = (ak.array([5, 9], dtype), ak.array([10, 20], dtype))
        vals = (
            ak.array([0, 0, 2, 5, 5, 6, 6, 9], dtype),
            ak.array([0, 20, 1, 5, 15, 0, 12, 30], dtype),
        )

        search_intervals = ak.search_intervals(vals, (starts, ends), hierarchical=False).tolist()
        assert search_intervals == [0, -1, 0, 0, 1, -1, 1, -1]

        search_intervals_hierarchical = ak.search_intervals(vals, (starts, ends)).tolist()
        assert search_intervals_hierarchical == [0, 0, 0, 0, 1, 1, 1, -1]

        # bigint is equivalent to hierarchical=True case
        bi_starts = ak.bigint_from_uint_arrays([ak.cast(a, ak.uint64) for a in starts])
        bi_ends = ak.bigint_from_uint_arrays([ak.cast(a, ak.uint64) for a in ends])
        bi_vals = ak.bigint_from_uint_arrays([ak.cast(a, ak.uint64) for a in vals])
        assert (
            ak.search_intervals(bi_vals, (bi_starts, bi_ends)).tolist() == search_intervals_hierarchical
        )

    @pytest.mark.parametrize("dtype", DATA_TYPES)
    def test_search_interval_nonunique(self, dtype):
        expected_result = [2, 5, 2, 1, 3, 1, 4, -1, -1]
        lb = [0, 10, 20, 30, 40, 50]
        ub = [9, 19, 29, 39, 49, 59]
        v = [22, 51, 22, 19, 38, 19, 40, 60, 100]

        lower_bound, upper_bound, vals = self.get_interval_info(lb, ub, v, dtype)
        interval_idxs = ak.search_intervals(vals, (lower_bound, upper_bound))
        assert expected_result == interval_idxs.tolist()

    def test_error_handling(self):
        lb = [0, 10, 20, 30, 40, 50]
        ub = [9, 19, 29, 39, 49, 59]
        v = [22, 51, 22, 19, 38, 19, 40, 60, 100]

        lower_bound = ak.array(lb, dtype=ak.int64)
        upper_bound = ak.array(ub, dtype=ak.float64)
        vals = ak.array(v, dtype=ak.int64)

        with pytest.raises(TypeError):
            ak.search_intervals(vals, (lower_bound, upper_bound))

        lower_bound = ak.array(lb, dtype=ak.int64)
        upper_bound = ak.array(ub, dtype=ak.int64)
        vals = ak.array(v, dtype=ak.int64)

        with pytest.raises(ValueError):
            ak.search_intervals(vals, (lower_bound, upper_bound, upper_bound))

        t = ak.array(["a", "b", "c", "d", "e", "f"])
        with pytest.raises(TypeError):
            ak.search_intervals(t, (lower_bound, upper_bound))

        with pytest.raises(ValueError):
            ak.search_intervals(vals, (ak.array([0, 10, 20]), upper_bound))

        with pytest.raises(ValueError):
            ak.search_intervals(vals, (upper_bound, lower_bound))

    def test_representative_cases(self):
        # Create 4 rectangles (2-d intervals) which demonstrate three classes of
        # relationships between multi-dimensional intervals (hyperslabs):
        #    1. Nested (B is a proper subset of A)
        #    2. Intersecting (A and C overlap but neither is a subset of the other)
        #    3. Disjoint (A and D do not intersect)
        # Then create points that explore each region of this diagram.

        A = [(2, 3), (5, 6)]
        B = [(2, 4), (3, 5)]
        C = [(4, 5), (6, 6)]
        D = [(7, 1), (8, 3)]
        lowerleft, upperright = tuple(zip(A, B, C, D))
        x0, y0 = tuple(zip(*lowerleft))
        x1, y1 = tuple(zip(*upperright))
        x0 = ak.array(x0)
        y0 = ak.array(y0)
        x1 = ak.array(x1)
        y1 = ak.array(y1)
        intervals = ((x0, y0), (x1, y1))

        testpoints = [
            (7, 8),
            (4, 7),
            (2, 6),
            (5, 6),
            (1, 5),
            (4, 5),
            (6, 5),
            (3, 4),
            (6, 4),
            (2, 3),
            (5, 3),
            (8, 2),
            (3, 1),
        ]
        x_test, y_test = tuple(zip(*testpoints))
        values = (ak.array(x_test), ak.array(y_test))
        tiebreak_smallest = (y1 - y0) * (x1 - x0)
        first_answer = [-1, -1, 0, 0, -1, 0, 2, 0, -1, 0, 0, 3, -1]
        smallest_answer = [-1, -1, 0, 2, -1, 2, 2, 1, -1, 0, 0, 3, -1]
        first_result = ak.search_intervals(values, intervals, hierarchical=False)
        assert first_result.tolist() == first_answer
        smallest_result = ak.search_intervals(
            values, intervals, tiebreak=tiebreak_smallest, hierarchical=False
        )
        assert smallest_result.tolist() == smallest_answer
