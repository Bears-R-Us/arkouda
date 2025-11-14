import pytest

from benchmark_utils import calc_num_bytes

import arkouda as ak


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Parquet_Fixed_Strings")
@pytest.mark.parametrize("scaling", [False, True])
@pytest.mark.parametrize("fixed_len", [1, 8])
@pytest.mark.parametrize("nfiles", [1, 5, 10])
def bench_parquet_fixed_strings(benchmark, tmp_path, scaling, fixed_len, nfiles):
    N = pytest.N if scaling else pytest.prob_size
    base_path = tmp_path / f"parq_{'scaled' if scaling else 'flat'}_{nfiles}"
    base_path.mkdir(parents=True, exist_ok=True)

    strings = ak.random_strings_uniform(fixed_len, fixed_len + 1, N, seed=pytest.seed)
    num_bytes = calc_num_bytes(strings)

    strings.to_parquet(base_path / "part_0.parquet")

    def run():
        ak.read_parquet(str(base_path / "part_0_LOCALE0000.parquet"), fixed_len=fixed_len)

    benchmark.pedantic(run, rounds=pytest.trials)

    benchmark.extra_info["description"] = (
        f"Parquet read performance for fixed-length strings of size {fixed_len}, "
        f"{'scaling' if scaling else 'non-scaling'} mode from {nfiles} files."
    )
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["fixed_len"] = fixed_len
    benchmark.extra_info["scaling"] = scaling
    benchmark.extra_info["nfiles"] = nfiles
    benchmark.extra_info["backend"] = "Arkouda"
    benchmark.extra_info["num_bytes"] = num_bytes
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
