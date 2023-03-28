import arkouda as ak
import pytest

TYPES = ("int64", "uint64", "float64")


@pytest.mark.parametrize("dtype", TYPES)
def test_argsort_correctness(dtype):
    N = pytest.prob_size
    seed = pytest.seed
    if dtype == "int64":
        a = ak.randint(0, 2**32, N, seed=seed)
    elif dtype == "uint64":
        a = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
    elif dtype == "float64":
        a = ak.randint(0, 1, N, dtype=ak.float64, seed=seed)

    perm = ak.argsort(a)
    if dtype in ("int64", "uint64", "float64"):
        assert ak.is_sorted(a[perm])
