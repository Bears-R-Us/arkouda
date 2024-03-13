import arkouda as ak


class TestUtil:
    def test_sparse_sum_helper(self):
        cfg = ak.get_config()
        N = (10**4) * cfg["numLocales"]
        select_from = ak.arange(N)
        inds1 = select_from[ak.randint(0, 10, N) % 3 == 0]
        inds2 = select_from[ak.randint(0, 10, N) % 3 == 0]
        vals1 = ak.randint(-(2**32), 2**32, N)[inds1]
        vals2 = ak.randint(-(2**32), 2**32, N)[inds2]

        merge_idx, merge_vals = ak.util.sparse_sum_help(inds1, inds2, vals1, vals2, merge=True)
        sort_idx, sort_vals = ak.util.sparse_sum_help(inds1, inds2, vals1, vals2, merge=False)
        gb_idx, gb_vals = ak.GroupBy(ak.concatenate([inds1, inds2], ordered=False)).sum(
            ak.concatenate((vals1, vals2), ordered=False)
        )

        assert (merge_idx == sort_idx).all()
        assert (merge_idx == gb_idx).all()
        assert (merge_vals == sort_vals).all()
