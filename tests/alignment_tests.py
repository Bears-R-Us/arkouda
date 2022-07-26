from base_test import ArkoudaTest
from context import arkouda as ak


class DataFrameTest(ArkoudaTest):
    def test_search_interval(self):
        expected_result = [2, 5, 4, 0, 3, 1, 4, -1, -1]
        lb = [0, 10, 20, 30, 40, 50]
        ub = [10, 20, 30, 40, 50, 60]
        v = [22, 51, 44, 1, 38, 19, 40, 60, 100]

        # test int64
        lower_bound = ak.array(lb)
        upper_bound = ak.array(ub)
        vals = ak.array(v)
        interval_idxs = ak.search_intervals(vals, (lower_bound, upper_bound))
        self.assertListEqual(expected_result, interval_idxs.to_ndarray().tolist())

        # test uint64
        lower_bound = ak.array(lb, dtype=ak.uint64)
        upper_bound = ak.array(ub, dtype=ak.uint64)
        vals = ak.array(v, dtype=ak.uint64)
        interval_idxs = ak.search_intervals(vals, (lower_bound, upper_bound))
        self.assertListEqual(expected_result, interval_idxs.to_ndarray().tolist())

        # test uint64
        lower_bound = ak.array(lb, dtype=ak.float64)
        upper_bound = ak.array(ub, dtype=ak.float64)
        vals = ak.array(v, dtype=ak.float64)
        interval_idxs = ak.search_intervals(vals, (lower_bound, upper_bound))
        self.assertListEqual(expected_result, interval_idxs.to_ndarray().tolist())

    def test_multi_array_search_interval(self):
        # Added for Issue #1548
        starts = (ak.array([0, 10, 20]), ak.array([0, 10, 20]))
        ends = (ak.array([5, 15, 25]), ak.array([5, 15, 25]))
        vals = (ak.array([3, 13, 23]), ak.array([23, 13, 3]))
        ans = [-1, 1, -1]
        self.assertListEqual(ans, ak.search_intervals(vals, (starts, ends)).to_ndarray().tolist())
        self.assertListEqual(
            ans, ak.interval_lookup((starts, ends), ak.arange(3), vals).to_ndarray().tolist()
        )

        vals = (ak.array([23, 13, 3]), ak.array([23, 13, 3]))
        ans = [2, 1, 0]
        self.assertListEqual(ans, ak.search_intervals(vals, (starts, ends)).to_ndarray().tolist())
        self.assertListEqual(
            ans, ak.interval_lookup((starts, ends), ak.arange(3), vals).to_ndarray().tolist()
        )

        vals = (ak.array([23, 13, 33]), ak.array([23, 13, 3]))
        ans = [2, 1, -1]
        self.assertListEqual(ans, ak.search_intervals(vals, (starts, ends)).to_ndarray().tolist())
        self.assertListEqual(
            ans, ak.interval_lookup((starts, ends), ak.arange(3), vals).to_ndarray().tolist()
        )

    def test_search_interval_nonunique(self):
        expected_result = [2, 5, 2, 1, 3, 1, 4, -1, -1]
        lb = [0, 10, 20, 30, 40, 50]
        ub = [10, 20, 30, 40, 50, 60]
        v = [22, 51, 22, 19, 38, 19, 40, 60, 100]

        # test int64
        lower_bound = ak.array(lb)
        upper_bound = ak.array(ub)
        vals = ak.array(v)
        interval_idxs = ak.search_intervals(vals, (lower_bound, upper_bound))
        self.assertListEqual(expected_result, interval_idxs.to_ndarray().tolist())

        # test uint64
        lower_bound = ak.array(lb, dtype=ak.uint64)
        upper_bound = ak.array(ub, dtype=ak.uint64)
        vals = ak.array(v, dtype=ak.uint64)
        interval_idxs = ak.search_intervals(vals, (lower_bound, upper_bound))
        self.assertListEqual(expected_result, interval_idxs.to_ndarray().tolist())

        # test uint64
        lower_bound = ak.array(lb, dtype=ak.float64)
        upper_bound = ak.array(ub, dtype=ak.float64)
        vals = ak.array(v, dtype=ak.float64)
        interval_idxs = ak.search_intervals(vals, (lower_bound, upper_bound))
        self.assertListEqual(expected_result, interval_idxs.to_ndarray().tolist())

    def test_error_handling(self):
        lb = [0, 10, 20, 30, 40, 50]
        ub = [10, 20, 30, 40, 50, 60]
        v = [22, 51, 22, 19, 38, 19, 40, 60, 100]

        lower_bound = ak.array(lb, dtype=ak.int64)
        upper_bound = ak.array(ub, dtype=ak.float64)
        vals = ak.array(v, dtype=ak.int64)

        with self.assertRaises(TypeError):
            ak.search_intervals(vals, (lower_bound, upper_bound))

        lower_bound = ak.array(lb, dtype=ak.int64)
        upper_bound = ak.array(ub, dtype=ak.int64)
        vals = ak.array(v, dtype=ak.int64)

        with self.assertRaises(ValueError):
            ak.search_intervals(vals, (lower_bound, upper_bound, upper_bound))

        t = ak.array(["a", "b", "c", "d", "e", "f"])
        with self.assertRaises(TypeError):
            ak.search_intervals(t, (lower_bound, upper_bound))

        with self.assertRaises(ValueError):
            ak.search_intervals(vals, (ak.array([0, 10, 20]), upper_bound))

        with self.assertRaises(ValueError):
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
        x1 = ak.array(x1) + 1  # convert to half-open
        y1 = ak.array(y1) + 1  # convert to half-open
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
        first_result = ak.search_intervals(values, intervals)
        self.assertListEqual(first_result.to_ndarray().tolist(), first_answer)
        smallest_result = ak.search_intervals(values, intervals, tiebreak=tiebreak_smallest)
        self.assertListEqual(smallest_result.to_ndarray().tolist(), smallest_answer)
