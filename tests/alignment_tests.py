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
