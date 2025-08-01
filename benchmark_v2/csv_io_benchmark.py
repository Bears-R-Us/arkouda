from glob import glob
import os

import pytest

import arkouda as ak

TYPES = ("str", "int64", "float64", "uint64")


def generate_array(dtype: str, size: int, seed: int, fixed_size: int = -1) -> ak.pdarray:
    if dtype == "int64":
        return ak.randint(0, 2**32, size, seed=seed)
    elif dtype == "float64":
        return ak.randint(0, 1, size, dtype=ak.float64, seed=seed)
    elif dtype == "uint64":
        return ak.randint(0, 2**32, size, dtype=ak.uint64, seed=seed)
    elif dtype == "str":
        if fixed_size > 0:
            return ak.random_strings_uniform(fixed_size, fixed_size + 1, size, seed=seed)
        else:
            return ak.random_strings_uniform(1, 16, size, seed=seed)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def remove_files(path):
    for f in glob(path + "*"):
        os.remove(f)


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="CSV_IO")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("op", ["read", "write"])
def bench_csv_io(benchmark, tmp_path, dtype, op):
    N = pytest.N
    path = tmp_path / f"csv_io_{dtype}"
    trials = pytest.trials
    nfiles = 1

    if op == "write":
        a = generate_array(dtype, N, seed=pytest.seed)

        def run():
            for j in range(nfiles):
                a.to_csv(f"{path}{j:04}", overwrite=True)

        benchmark.pedantic(run, rounds=trials)
    else:
        a = generate_array(dtype, N, seed=pytest.seed)
        for j in range(nfiles):
            a.to_csv(f"{path}{j:04}", overwrite=True)

        def run():
            ak.read_csv(str(path) + "*").popitem()[1]

        benchmark.pedantic(run, rounds=trials)
        remove_files(str(path))

    num_bytes = a.nbytes

    benchmark.extra_info["description"] = f"CSV {op} benchmark for dtype={dtype}"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["operation"] = op
    benchmark.extra_info["backend"] = "Arkouda"
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (num_bytes / benchmark.stats["mean"]) / 2**30
    )
