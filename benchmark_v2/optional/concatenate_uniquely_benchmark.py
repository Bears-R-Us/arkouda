import pytest

import arkouda as ak

from arkouda import Strings
from benchmark_v2.benchmark_utils import calc_num_bytes


# Fraction of each (post-overlap) array that is made up of shared overlap strings
OVERLAP = (0.0, 0.25, 0.5, 0.75)

# 1 => ak.unique(ak.concatenate([a, b], ordered=False))
# 2 => Strings.concatenate_uniquely([a, b])
METHOD = (1, 2)


def _unique_concat(a: Strings, b: Strings) -> Strings:
    return ak.unique(ak.concatenate([a, b], ordered=False))


def _concatenate_uniquely(a: Strings, b: Strings) -> Strings:
    return Strings.concatenate_uniquely([a, b])


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="AK_strings_concat_unique")
@pytest.mark.parametrize("overlap", OVERLAP)
@pytest.mark.parametrize("method", METHOD)
def bench_strings_concat_unique(benchmark, overlap, method):
    """
    Benchmark concatenating two Strings arrays and removing duplicates, with a controlled overlap.

    Setup:
      - Build two disjoint base arrays:
          a_base: ~12-char strings
          b_base: ~13-char strings
      - Build an overlap array:
          ov: ~11-char strings
      - Append ov into both a_base and b_base using ak.concatenate(..., ordered=False)

    Bench:
      method=1: ak.unique(ak.concatenate([a, b], ordered=False))
      method=2: Strings.concatenate_uniquely([a, b])
    """
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]
    seed = pytest.seed or 0

    # Choose an element count so bytes are N.
    # Use a conservative avg length estimate (printable chars).
    avg_chars = 13.0
    final_n = max(1, int(N / avg_chars))

    # overlap is fraction of each *final* array that comes from the shared overlap array
    ov_n = int(final_n * overlap)
    base_n = max(1, final_n - ov_n)  # ensure we always have some non-overlap content

    # Random string generators:
    # NOTE: random_strings_uniform requires maxlen > minlen, so fixed length L uses (L, L+1).
    a_base = ak.random_strings_uniform(
        minlen=12, maxlen=13, size=base_n, characters="printable", seed=seed
    )
    b_base = ak.random_strings_uniform(
        minlen=13, maxlen=14, size=base_n, characters="printable", seed=seed + 1
    )
    ov = ak.random_strings_uniform(
        minlen=11, maxlen=12, size=ov_n, characters="printable", seed=seed + 2
    )

    # Append overlap into both (unordered concat to avoid extra overhead / guarantees)
    a = ak.concatenate([a_base, ov], ordered=False)
    b = ak.concatenate([b_base, ov], ordered=False)

    # Bytes involved (inputs; output size varies with overlap + method)
    bytes_a = calc_num_bytes(a)
    bytes_b = calc_num_bytes(b)
    num_bytes = bytes_a + bytes_b

    fns = [
        _unique_concat,
        _concatenate_uniquely,
    ]
    fn = fns[method - 1]

    benchmark.pedantic(
        fn,
        args=(a, b),
        rounds=pytest.trials,
    )

    benchmark.extra_info["description"] = (
        "Concats two Strings arrays and removes duplicates; "
        f"overlap={overlap}, method={method} "
        "(1=unique(concat), 2=Strings.concatenate_uniquely)."
    )
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["num_bytes"] = num_bytes
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (num_bytes / benchmark.stats["mean"]) / 2**30
    )
