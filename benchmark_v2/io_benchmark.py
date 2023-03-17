import arkouda as ak
import os
import pytest
from glob import glob

TYPES = (
    "int64",
    "float64",
    "uint64",
)

FILETYPES = ("HDF5", "PARQUET")


def write_files(a, ftype):
    for i in range(pytest.io_files):
        a.to_hdf(f"{pytest.io_path}_hdf_{i:04}") if ftype == "HDF5" else a.to_parquet(
            f"{pytest.io_path}_par_{i:04}", compression="snappy" if pytest.io_compressed else None
        )


@pytest.mark.benchmark(group="Arkouda_IO")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("ftype", FILETYPES)
def bench_ak_write(benchmark, dtype, ftype):
    if pytest.io_write or (not pytest.io_read and not pytest.io_delete) and dtype in pytest.dtype:
        cfg = ak.get_config()
        N = pytest.prob_size * cfg["numLocales"]

        if dtype == "int64":
            a = ak.randint(0, 2**32, N, seed=pytest.seed)
        elif dtype == "float64":
            a = ak.randint(0, 1, N, dtype=ak.float64, seed=pytest.seed)
        elif dtype == "uint64":
            a = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)

        benchmark.pedantic(write_files, args=(a, ftype), rounds=pytest.trials)

        nbytes = a.size * a.itemsize * pytest.io_files

        benchmark.extra_info["description"] = f"Measures the performance of IO write {dtype} to {ftype} file"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2 ** 30)


@pytest.mark.benchmark(group="Arkouda_IO")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("ftype", FILETYPES)
def bench_ak_read(benchmark, dtype, ftype):
    if pytest.io_read or (not pytest.io_write and not pytest.io_delete) and dtype in pytest.dtype:
        cfg = ak.get_config()
        N = pytest.prob_size * cfg["numLocales"]
        a = ak.array([])

        if ftype == "HDF5":
            a = benchmark.pedantic(ak.read_hdf, args=[pytest.io_path + "_hdf_*"], rounds=pytest.trials)
        else:
            a = benchmark.pedantic(ak.read_parquet, args=[pytest.io_path + "_par_*"], rounds=pytest.trials)

        nbytes = a.size * a.itemsize

        benchmark.extra_info["description"] = f"Measures the performance of IO read {dtype} from {ftype} file"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2 ** 30)


@pytest.mark.benchmark(group="Arkouda_IO")
def bench_ak_delete(benchmark):
    if pytest.io_delete or (not pytest.io_write and not pytest.io_read):
        benchmark.pedantic(remove_files, rounds=1)

        cfg = ak.get_config()
        benchmark.extra_info["description"] = f"Measures the performance of IO delete files from system"
        benchmark.extra_info["problem_size"] = pytest.io_files * cfg["numLocales"]
        benchmark.extra_info["transfer_rate"] = "N/A"


def remove_files():
    for f in glob(pytest.io_path + "*"):
        os.remove(f)