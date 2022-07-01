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

        with self.assertRaises(ValueError):
            ak.search_intervals(vals, (lower_bound[::-1], upper_bound[::-1]))

        with self.assertRaises(ValueError):
            ak.search_intervals(
                vals, (ak.array([0, 10, 20, 30, 40, 50]), ak.array([10, 20, 35, 40, 50, 60]))
            )
