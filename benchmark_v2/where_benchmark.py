import numpy as np
import pytest

import arkouda as ak

TYPES = ("int64", "uint64", "float64")
STYLES = ("vv", "vs", "sv", "ss")


def alternate(L, R, n):
    v = np.full(n, R)
    v[::2] = L
    return v


@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Numpy_Scan")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("v_or_s", STYLES)
def bench_where(benchmark, dtype, v_or_s):
    """
    Measures the performance of where
    """
    if dtype in pytest.dtype:
        cfg = ak.get_config()
        N = pytest.prob_size * cfg["numLocales"]
        seed = pytest.seed

        a = ak.randint(0, 2**32, N, seed=seed, dtype=dtype)
        b = ak.randint(0, 2**32, N, seed=seed, dtype=dtype)
        c = ak.array(alternate(True, False, N))

        fxn = getattr(ak, "where")
        if v_or_s == "vv":
            left = a
            right = b
        elif v_or_s == "vs":
            left = a
            right = b[0]
        elif v_or_s == "sv":
            left = a[0]
            right = b
        else:
            left = a[0]
            right = b[0]

        benchmark.pedantic(fxn, args=[c, left, right], rounds=pytest.trials)

        benchmark.extra_info["description"] = "Measures the performance of where" + " " + v_or_s
        benchmark.extra_info["problem_size"] = pytest.prob_size
