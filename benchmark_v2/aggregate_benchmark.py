import pytest

from benchmark_utils import calc_num_bytes

import arkouda as ak


def setup_agg(N, t="int"):
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
    N = pytest.N

    if op in ["any", "all"]:
        g, vals, keys = setup_agg(N, "bool")
    else:
        g, vals, keys = setup_agg(N, "int")

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

        benchmark.pedantic(numpy_agg, rounds=pytest.trials)
        num_bytes = calc_num_bytes(vals_np)
    else:
        benchmark.pedantic(run_agg, args=(g, vals, op), rounds=pytest.trials)
        num_bytes = calc_num_bytes(vals)

    benchmark.extra_info["description"] = (
        f"Measures performance of GroupBy.aggregate using the {op} operator."
    )
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["num_bytes"] = num_bytes
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
