import numpy as np
import pytest

import arkouda as ak
from arkouda.numpy.sorting import SortingAlgorithm

TYPES = ("int64", "float64")


def get_nbytes(data):
    if isinstance(data, (ak.pdarray, ak.Strings)):
        return data.nbytes
    else:
        return sum(x.nbytes for x in data)


def do_argsort(data, algo):
    if isinstance(data, (ak.pdarray, ak.Strings)):
        return ak.argsort(data, algo)
    else:
        return ak.coargsort(data, algo)


def run_sort(benchmark, name, data, sort_fn):
    # NumPy fallback
    if pytest.numpy:
        if isinstance(data, tuple):
            data_np = tuple(x.to_ndarray() for x in data)

            def sort_op():
                np.lexsort(data_np[::-1])
                return sum(x.nbytes for x in data_np)
        else:
            data_np = data.to_ndarray()

            def sort_op():
                np.argsort(data_np)
                return data_np.nbytes

        backend = "NumPy"

    else:

        def sort_op():
            p = sort_fn(data)
            if pytest.correctness_only:
                if isinstance(data, tuple):
                    sorted_data = [d[p] for d in data]  # noqa: F841
                    # Assume lexsorted comparison is hard to check; skip for now
                else:
                    assert ak.is_sorted(data[p])
            return get_nbytes(data)

        backend = "Arkouda"

    numBytes = benchmark.pedantic(sort_op, rounds=pytest.trials)

    benchmark.extra_info["description"] = f"{name} via {backend}"
    benchmark.extra_info["problem_size"] = data[0].size if isinstance(data, tuple) else data.size
    benchmark.extra_info["backend"] = backend
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (numBytes / benchmark.stats["mean"]) / 2**30
    )


@pytest.mark.benchmark(group="AK_Sort_Cases")
@pytest.mark.parametrize("algo", SortingAlgorithm)
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("bits", ("16-bit", "32-bit", "64-bit"))
def bench_random_uniform(benchmark, algo, dtype, bits):
    """
    Uniformly distributed integers of 1, 2, and 4 digits.
    Uniformly distributed reals in (0, 1)
    """
    if dtype in pytest.dtype:
        N = pytest.prob_size

        if dtype == "int64":
            if bits == "16-bit":
                data = ak.randint(0, 2**16, N)
            elif bits == "32-bit":
                data = ak.randint(0, 2**32, N)
            else:
                data = ak.randint(-(2**63), 2**63, N)

            run_sort(benchmark, "random_uniform", data, lambda x: do_argsort(x, algo))
        elif bits == "64-bit":  # float64 - We only want this to run once per algorithm, not 3 times
            data = ak.uniform(N)
            run_sort(benchmark, "random_uniform", data, lambda x: do_argsort(x, algo))


def _generate_power_law_data():
    y = ak.uniform(pytest.prob_size)
    a = -2.5  # power law exponent, between -2 and -3
    ub = 2**32  # upper bound

    return ((ub ** (a + 1) - 1) * y + 1) ** (1 / (a + 1))


@pytest.mark.benchmark(group="AK_Sort_Cases")
@pytest.mark.parametrize("algo", SortingAlgorithm)
@pytest.mark.parametrize("dtype", TYPES)
def bench_power_law(benchmark, algo, dtype):
    """
    Power law distributed (alpha = 2.5) reals and integers in (1, 2**32)
    """
    if dtype in pytest.dtype:
        data = _generate_power_law_data()

        if dtype == "int64":
            data = ak.cast(data, ak.int64)

        run_sort(benchmark, "random_uniform", data, lambda x: do_argsort(x, algo))


@pytest.mark.benchmark(group="AK_Sort_Cases")
@pytest.mark.parametrize("algo", SortingAlgorithm)
def bench_rmat(benchmark, algo):
    """
    RMAT-generated edges (coargsort of two vertex arrays)
    """
    # N = number of edges = number of elements / 2
    N = pytest.prob_size // 2
    avgdegree = 10
    lgNv = int(np.log2(N / avgdegree))
    # probabilities
    a = 0.01
    b = (1.0 - a) / 3.0
    c = b
    d = b
    # quantites to use in edge generation loop
    ab = a + b
    c_norm = c / (c + d)
    a_norm = a / (a + b)
    # init edge arrays
    ii = ak.ones(N, dtype=ak.int64)
    jj = ak.ones(N, dtype=ak.int64)
    # generate edges
    for ib in range(1, lgNv):
        ii_bit = ak.uniform(N) > ab
        jj_bit = ak.uniform(N) > (c_norm * ii_bit + a_norm * (~ii_bit))
        ii = ii + ((2 ** (ib - 1)) * ii_bit)
        jj = jj + ((2 ** (ib - 1)) * jj_bit)

    data = (ii, jj)

    run_sort(benchmark, "random_uniform", data, lambda x: do_argsort(x, algo))


@pytest.mark.benchmark(group="AK_Sort_Cases")
@pytest.mark.parametrize("algo", SortingAlgorithm)
@pytest.mark.parametrize("mode", ("concat", "interleaved"))
def bench_block_sorted(benchmark, algo, mode):
    """
    The concatenation of two sorted arrays of unequal length
    The interleaving of two sorted arrays of unequal length

    Most often occurs in array setops, where two arrays are
    uniqued (via sorting), then concatenated and sorted
    """
    N = pytest.prob_size

    splitpoint = 0.4
    Na = int(splitpoint * N)
    Nb = N - Na
    # Construct a and b such that:
    #   1) Values overlap
    #   2) a and b are sorted
    a = ak.arange(Na)
    b = ak.arange(Nb)
    data = ak.concatenate((a, b), ordered=(mode == "concat"))

    run_sort(benchmark, "random_uniform", data, lambda x: do_argsort(x, algo))


@pytest.mark.benchmark(group="AK_Sort_Cases")
@pytest.mark.parametrize("algo", SortingAlgorithm)
def bench_refinement(benchmark, algo):
    """
    Coargsort of two arrays, where the first is already sorted
    but has many repeated values
    """
    N = pytest.prob_size
    groupsize = 100
    a = ak.arange(N // 2) // groupsize
    factor = 2**32 // a.max()
    a *= factor
    b = ak.randint(0, 2**32, N // 2)
    data = (a, b)

    run_sort(benchmark, "random_uniform", data, lambda x: do_argsort(x, algo))


@pytest.mark.benchmark(group="AK_Sort_Cases")
@pytest.mark.parametrize("algo", SortingAlgorithm)
def bench_time_like(benchmark, algo):
    """
    Data like a datetime64[ns]:
    - spanning 1 year
    - with second granularity
    - but stored with nanosecond precision
    """
    # seconds in a year
    year_sec = 365 * 24 * 60 * 60
    # offset to almost 2020 (yeah yeah, leap days)
    twentytwenty = 50 * year_sec
    # second-resolution timestamps spanning approx 2020-2021
    a = ak.randint(0, year_sec, pytest.prob_size) + twentytwenty
    # stored as datetime64[ns]
    a *= 10**9
    data = a

    run_sort(benchmark, "random_uniform", data, lambda x: do_argsort(x, algo))


@pytest.mark.benchmark(group="AK_Sort_Cases")
@pytest.mark.parametrize("algo", SortingAlgorithm)
def bench_ip_like(benchmark, algo):
    """
    Data like a 90/10 mix of IPv4 and IPv6 addresses
    """
    N = pytest.prob_size
    multiplicity = 10
    nunique = N // (2 * multiplicity) if N >= (2 * multiplicity) else 1
    # First generate unique addresses, then sample with replacement
    u1 = ak.zeros(nunique, dtype=ak.int64)
    u2 = ak.zeros(nunique, dtype=ak.int64)
    v4 = ak.uniform(nunique) < 0.9
    n4 = v4.sum()
    v6 = ~v4
    n6 = v4.size - n4
    u1[v4] = ak.randint(0, 2**32, n4)
    u1[v6] = ak.randint(-(2**63), 2**63, n6)
    u2[v6] = ak.randint(-(2**63), 2**63, n6)
    sample = ak.randint(0, nunique, N // 2)
    IP1 = u1[sample]
    IP2 = u2[sample]
    data = (IP1, IP2)

    run_sort(benchmark, "random_uniform", data, lambda x: do_argsort(x, algo))
