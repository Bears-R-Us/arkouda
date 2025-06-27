from pathlib import Path

import pytest

import arkouda as ak


@pytest.mark.skip_numpy(True)
@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Parquet_Fixed_Strings")
@pytest.mark.parametrize("scaling", [False, True])
@pytest.mark.parametrize("fixed_size", [1, 8])
@pytest.mark.parametrize("nfiles", [1])
def bench_parquet_fixed_strings(benchmark, tmp_path, scaling, fixed_size, nfiles):
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"] if scaling else pytest.prob_size
    base_path = Path(pytest.io_path) / f"parq_{'scaled' if scaling else 'flat'}_{nfiles}"
    base_path.mkdir(parents=True, exist_ok=True)

    strings = ak.random_strings_uniform(fixed_size, fixed_size + 1, N, seed=pytest.seed)

    strings.to_parquet(base_path / "part_0.parquet")

    def run():
        ak.read_parquet(str(base_path / "part_0_LOCALE0000.parquet"), fixed_len=fixed_size)

    benchmark.pedantic(run, rounds=pytest.trials)

    benchmark.extra_info["description"] = (
        f"Parquet read performance for fixed-length strings of size {fixed_size}, "
        f"{'scaling' if scaling else 'non-scaling'} mode."
    )
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["fixed_size"] = fixed_size
    benchmark.extra_info["scaling"] = scaling
    benchmark.extra_info["nfiles"] = nfiles
