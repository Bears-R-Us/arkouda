import pytest

import arkouda as ak


def setup_agg(t="int", N=None):
    cfg = ak.get_config()
    N = N or (pytest.prob_size * cfg["numLocales"])

    keys = ak.sort(ak.randint(0, 2**32, N, seed=pytest.seed if pytest.seed is not None else None))
    intvals = ak.randint(0, 2**16, N, seed=(pytest.seed + 1 if pytest.seed is not None else None))
    g = ak.GroupBy(keys, assume_sorted=True)

    if t == "int":
        return g, intvals, keys
    else:
        boolvals = (intvals % 2) == 0
        return g, boolvals, keys


def run_agg(g, vals, op):
    g.aggregate(vals, op)
    return vals.size + vals.itemsize


@pytest.mark.benchmark(group="GroupBy.aggregate")
@pytest.mark.parametrize("op", ak.GroupBy.Reductions)
def bench_aggregate(benchmark, op):
    N = 10**4 if pytest.correctness_only else pytest.prob_size * ak.get_config()["numLocales"]

    if op in ["any", "all"]:
        g, vals, keys = setup_agg("bool", N)
    else:
        g, vals, keys = setup_agg("int", N)

    if pytest.numpy:
        keys_np = keys.to_ndarray()
        vals_np = vals.to_ndarray()

        def numpy_agg():
            import pandas as pd

            df = pd.DataFrame({"key": keys_np, "val": vals_np})
            if op == "sum":
                df.groupby("key")["val"].sum()
            elif op == "prod":
                df.groupby("key")["val"].prod()
            elif op == "min":
                df.groupby("key")["val"].min()
            elif op == "max":
                df.groupby("key")["val"].max()
            elif op == "mean":
                df.groupby("key")["val"].mean()
            elif op == "any":
                df.groupby("key")["val"].any()
            elif op == "all":
                df.groupby("key")["val"].all()
            elif op == "count":
                df.groupby("key")["val"].count()

            return vals_np.size * vals_np.itemsize

        numBytes = benchmark.pedantic(numpy_agg, rounds=pytest.trials)
    else:
        numBytes = benchmark.pedantic(run_agg, args=(g, vals, op), rounds=pytest.trials)

        if pytest.correctness_only:
            # Validate against pandas
            keys_np = keys.to_ndarray()
            vals_np = vals.to_ndarray()
            import pandas as pd

            df = pd.DataFrame({"key": keys_np, "val": vals_np})
            if hasattr(df.groupby("key")["val"], op):
                if op == "prod":  # avoid numeric instability for large ints
                    result_np = df.groupby("key")["val"].prod()
                else:
                    result_np = getattr(df.groupby("key")["val"], op)()

                ark_keys, ark_vals = g.aggregate(vals, op)
                result_df = pd.Series(data=ark_vals.to_ndarray().flatten(), index=ark_keys.to_ndarray())
                pd.testing.assert_series_equal(
                    result_np.sort_index(),
                    result_df.sort_index(),
                    check_names=False,
                    check_dtype=False,
                    rtol=1e-10,
                )

    benchmark.extra_info["description"] = (
        f"Measures performance of GroupBy.aggregate using the {op} operator."
    )
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (numBytes / benchmark.stats["mean"]) / 2**30
    )
