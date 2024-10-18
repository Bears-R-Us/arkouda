import os
from glob import glob

import pytest

import arkouda as ak
from arkouda.io import to_parquet

TYPES = ("int64", "float64", "uint64", "str")
FILETYPES = ("HDF5", "PARQUET")
COMPRESSIONS = (None, "snappy", "gzip", "brotli", "zstd", "lz4")


def _write_files(a, ftype, dtype, compression=None):
    for i in range(pytest.io_files):
        a.to_hdf(f"{pytest.io_path}_hdf_{dtype}_{i:04}") if ftype == "HDF5" else to_parquet(
            [a],
            f"{pytest.io_path}_par_{compression}_{dtype}_{i:04}",
            compression=compression,
        )


def _write_multi(a, dtype, compression=None):
    data = a._prep_data()
    for i in range(pytest.io_files):
        to_parquet(
            data,
            f"{pytest.io_path}_par_multi_{compression}_{dtype}_{i:04}",
            compression=compression,
        )


def _append_files(a, dtype, compression):
    _remove_append_test_files(compression, dtype)
    for i in range(pytest.io_files):
        for key in a:
            val = a[key]
            to_parquet(
                [val],
                f"{pytest.io_path}_par_multi_{compression}_{dtype}_app_{i:04}",
                names=[key],
                mode="append",
                compression=compression,
            )


def _generate_array(N, dtype):
    if dtype == "int64":
        return ak.randint(0, 2**32, N, seed=pytest.seed)
    elif dtype == "float64":
        return ak.randint(0, 1, N, dtype=ak.float64, seed=pytest.seed)
    elif dtype == "uint64":
        return ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
    elif dtype == "str":
        return ak.random_strings_uniform(1, 16, N, seed=pytest.seed)


def _generate_df(N, dtype, returnDict=False):
    df_dict = {
        "c_1": _generate_array(N, dtype),
        "c_2": _generate_array(N, dtype),
        "c_3": _generate_array(N, dtype),
        "c_4": _generate_array(N, dtype),
        "c_5": _generate_array(N, dtype),
    }
    return df_dict if returnDict else ak.DataFrame(df_dict)

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_IO_Write_HDF5")
@pytest.mark.parametrize("dtype", TYPES)
def bench_ak_write_hdf(benchmark, dtype):
    if pytest.io_write or (not pytest.io_read and not pytest.io_delete) and dtype in pytest.dtype:
        cfg = ak.get_config()
        N = pytest.prob_size * cfg["numLocales"]

        a = _generate_array(N, dtype)

        benchmark.pedantic(_write_files, args=(a, "HDF5", dtype), rounds=pytest.trials)

        if dtype in ["int64", "float64", "uint64"]:
            nbytes = a.size * a.itemsize * pytest.io_files
        else:
            nbytes = a.nbytes * a.entry.itemsize * pytest.io_files

        benchmark.extra_info[
            "description"
        ] = f"Measures the performance of IO write {dtype} to HDF5 file"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_IO_Write_Parquet")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("comp", COMPRESSIONS)
def bench_ak_write_parquet(benchmark, dtype, comp):
    if pytest.io_write or (not pytest.io_read and not pytest.io_delete) \
            and dtype in pytest.dtype and comp in pytest.io_compression:
        cfg = ak.get_config()
        N = pytest.prob_size * cfg["numLocales"]

        a = _generate_array(N, dtype)

        benchmark.pedantic(_write_files, args=(a, "PARQUET", dtype, comp), rounds=pytest.trials)

        if dtype in ["int64", "float64", "uint64"]:
            nbytes = a.size * a.itemsize * pytest.io_files
        else:
            nbytes = a.nbytes * a.entry.itemsize * pytest.io_files

        benchmark.extra_info[
            "description"
        ] = f"Measures the performance of IO write {dtype} to Parquet file using {comp} compression"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_IO_Write_Parquet")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("comp", COMPRESSIONS)
def bench_ak_write_parquet_multi(benchmark, dtype, comp):
    if pytest.io_write or (not pytest.io_read and not pytest.io_delete) \
            and dtype in pytest.dtype and comp in pytest.io_compression:
        cfg = ak.get_config()
        N = pytest.prob_size * cfg["numLocales"]

        a = _generate_df(N, dtype)

        benchmark.pedantic(_write_multi, args=(a, dtype, comp), rounds=pytest.trials)

        nbytes = 0
        for c in a.columns:
            col = a[c]
            if dtype in ["int64", "float64", "uint64"]:
                nbytes = col.size * col.itemsize * pytest.io_files
            else:
                nbytes = col.nbytes * col.entry.itemsize * pytest.io_files

        benchmark.extra_info[
            "description"
        ] = f"Measures the performance of IO write {dtype} to Parquet file using {comp} compression"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_IO_Write_Parquet")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("comp", COMPRESSIONS)
def bench_ak_write_parquet_append(benchmark, dtype, comp):
    if pytest.io_write or (not pytest.io_read and not pytest.io_delete) \
            and dtype in pytest.dtype and comp in pytest.io_compression:
        cfg = ak.get_config()
        N = pytest.prob_size * cfg["numLocales"]

        a = _generate_df(N, dtype, True)

        benchmark.pedantic(_append_files, args=(a, dtype, comp), rounds=pytest.trials)

        nbytes = 0
        for col in a.values():
            if dtype in ["int64", "float64", "uint64"]:
                nbytes = col.size * col.itemsize * pytest.io_files
            else:
                nbytes = col.nbytes * col.entry.itemsize * pytest.io_files

        benchmark.extra_info[
            "description"
        ] = f"Measures the performance of IO write {dtype} to Parquet file using {comp} compression"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_IO_Read_HDF5")
@pytest.mark.parametrize("dtype", TYPES)
def bench_ak_read_hdf(benchmark, dtype):
    if pytest.io_read or (not pytest.io_write and not pytest.io_delete) and dtype in pytest.dtype:
        dataset = "strings_array" if dtype == "str" else "array"
        a = benchmark.pedantic(
            ak.read_hdf, args=[pytest.io_path + f"_hdf_{dtype}*", dataset], rounds=pytest.trials
        )

        nbytes = 0
        if isinstance(a, ak.pdarray):
            nbytes += a.size * a.itemsize
        elif isinstance(a, ak.Strings):
            nbytes += a.nbytes * a.entry.itemsize

        benchmark.extra_info["description"] = "Measures the performance of IO read from HDF5 files"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_IO_Read_Parquet")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("comp", COMPRESSIONS)
def bench_ak_read_parquet(benchmark, dtype, comp):
    if pytest.io_read or (not pytest.io_write and not pytest.io_delete) \
            and comp in pytest.io_compression and dtype in pytest.dtype:
        a = benchmark.pedantic(
            ak.read_parquet, args=[pytest.io_path + f"_par_{comp}_{dtype}_*"], rounds=pytest.trials
        )

        nbytes = 0
        if isinstance(a, ak.pdarray):
            nbytes += a.size * a.itemsize
        elif isinstance(a, ak.Strings):
            nbytes += a.nbytes * a.entry.itemsize

        benchmark.extra_info["description"] = "Measures the performance of IO read from Parquet files"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_IO_Read_Parquet")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("comp", COMPRESSIONS)
def bench_ak_read_parquet_multi_column(benchmark, dtype, comp):
    """
    Read files written by parquet multicolumn and parquet append modes
    """
    if pytest.io_read or (not pytest.io_write and not pytest.io_delete) \
            and comp in pytest.io_compression and dtype in pytest.dtype:
        a = benchmark.pedantic(
            ak.read_parquet, args=[pytest.io_path + f"_par_multi_{comp}_{dtype}_*"], rounds=pytest.trials
        )

        nbytes = 0
        for col in a:
            if isinstance(col, ak.pdarray):
                nbytes += col.size * col.itemsize
            elif isinstance(col, ak.Strings):
                nbytes += col.nbytes * col.entry.itemsize

        benchmark.extra_info["description"] = "Measures the performance of IO read from Parquet files"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_IO_Delete")
def bench_ak_delete(benchmark):
    if pytest.io_delete or (not pytest.io_write and not pytest.io_read):
        benchmark.pedantic(_remove_files, rounds=1)

        cfg = ak.get_config()
        benchmark.extra_info["description"] = "Measures the performance of IO delete files from system"
        benchmark.extra_info["problem_size"] = pytest.io_files * cfg["numLocales"]
        benchmark.extra_info["transfer_rate"] = "N/A"


def _remove_files():
    for f in glob(pytest.io_path + "*"):
        os.remove(f)


def _remove_append_test_files(compression, dtype):
    for f in glob(f"{pytest.io_path}_par_multi_{compression}_{dtype}_app_" + "*"):
        os.remove(f)
