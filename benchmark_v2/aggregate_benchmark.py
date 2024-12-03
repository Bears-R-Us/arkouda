import arkouda as ak
import pytest


def setup_agg(t="int"):
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]

    # Sort keys so that aggregations will not have to permute values
    # We just want to measure aggregation time, not gather
    keys = ak.sort(ak.randint(0, 2**32, N, seed=pytest.seed))
    intvals = ak.randint(0, 2**16, N, seed=(pytest.seed + 1 if pytest.seed is not None else None))
    g = ak.GroupBy(keys, assume_sorted=True)

    if t == "int":
        return g, intvals
    else:
        boolvals = (intvals % 2) == 0
        return g, boolvals


def run_agg(g, vals, op):
    g.aggregate(vals, op)

    return vals.size + vals.itemsize

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="GroupBy.aggregate")
@pytest.mark.parametrize("op", ak.GroupBy.Reductions)
def bench_aggregate(benchmark, op):
    if op in ["any", "all"]:
        g, vals = setup_agg("bool")
    else:
        g, vals = setup_agg()

    numBytes = benchmark.pedantic(run_agg, args=(g, vals, op), rounds=pytest.trials)
    benchmark.extra_info[
        "description"
    ] = f"Measures performance of GroupBy.aggregate using the {op} operator."
    benchmark.extra_info["problem_size"] = pytest.prob_size
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (numBytes / benchmark.stats["mean"]) / 2**30
    )
