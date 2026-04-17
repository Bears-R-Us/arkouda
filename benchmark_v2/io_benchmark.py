import os
import shutil

from glob import glob

import pytest

from benchmark_utils import calc_num_bytes

import arkouda as ak

from arkouda.pandas.io import to_parquet


TYPES = ("int64", "float64", "uint64", "str")
FILETYPES = ("HDF5", "PARQUET")
COMPRESSIONS = (None, "snappy", "gzip", "brotli", "zstd", "lz4")


def _write_files(a, ftype, dtype, compression=None):
    for i in range(pytest.io_files):
        fname = f"{_build_prefix(ftype, dtype, compression)}_{i:04}"
        (
            a.to_hdf(fname)
            if ftype == "HDF5"
            else to_parquet(
                [a],
                fname,
                compression=compression,
            )
        )


def _write_multi(a, dtype, compression=None):
    data = a._prep_data()
    for i in range(pytest.io_files):
        to_parquet(
            data,
            f"{pytest.io_path}_par_multi_{compression}_{dtype}_{i:04}",
            compression=compression,
        )


def _write_multi_hdf(df, dtype):
    # Write each column into its own HDF5 file across pytest.io_files,
    # mirroring the "multi" strategy used for Parquet (many files, multiple columns).
    for i in range(pytest.io_files):
        for c in df.columns:
            arr = df[c]
            fname = f"{pytest.io_path}_hdf_multi_{dtype}_{c}_{i:04}"
            arr.to_hdf(fname)


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


def _build_prefix(ftype: str, dtype: str, compression=None, multi=False, append=False):
    base = f"{pytest.io_path}_"
    if ftype == "HDF5":
        return base + f"hdf_{dtype}"
    if multi:
        path = base + f"par_multi_{compression}_{dtype}"
        if append:
            path += "_app"
        return path
    return base + f"par_{compression}_{dtype}"


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_IO_Write_HDF5")
@pytest.mark.parametrize("dtype", TYPES)
def bench_write_hdf(benchmark, dtype):
    if pytest.io_write or (not pytest.io_read and not pytest.io_delete) and dtype in pytest.dtype:
        N = pytest.N

        a = _generate_array(N, dtype)

        benchmark.pedantic(_write_files, args=(a, "HDF5", dtype), rounds=pytest.trials)

        if dtype in ["int64", "float64", "uint64"]:
            num_bytes = calc_num_bytes(a) * pytest.io_files
        else:
            num_bytes = calc_num_bytes(a) * pytest.io_files

        benchmark.extra_info["description"] = (
            f"Measures the performance of IO write {dtype} to HDF5 file"
        )
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_IO_Write_Parquet")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("comp", COMPRESSIONS)
def bench_write_parquet(benchmark, dtype, comp):
    if (
        pytest.io_write
        or (not pytest.io_read and not pytest.io_delete)
        and dtype in pytest.dtype
        and comp in pytest.io_compression
    ):
        N = pytest.N

        a = _generate_array(N, dtype)

        benchmark.pedantic(_write_files, args=(a, "PARQUET", dtype, comp), rounds=pytest.trials)

        if dtype in ["int64", "float64", "uint64"]:
            num_bytes = calc_num_bytes(a) * pytest.io_files
        else:
            num_bytes = calc_num_bytes(a) * pytest.io_files

        benchmark.extra_info["description"] = (
            f"Measures the performance of IO write {dtype} to Parquet file using {comp} compression"
        )
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_IO_Write_Parquet")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("comp", COMPRESSIONS)
def bench_write_parquet_multi(benchmark, dtype, comp):
    if (
        pytest.io_write
        or (not pytest.io_read and not pytest.io_delete)
        and dtype in pytest.dtype
        and comp in pytest.io_compression
    ):
        N = pytest.N

        a = _generate_df(N, dtype)

        benchmark.pedantic(_write_multi, args=(a, dtype, comp), rounds=pytest.trials)

        num_bytes = 0
        for c in a.columns:
            col = a[c]
            if dtype in ["int64", "float64", "uint64"]:
                num_bytes += calc_num_bytes(col) * pytest.io_files
            else:
                num_bytes += calc_num_bytes(col) * pytest.io_files

        benchmark.extra_info["description"] = (
            f"Measures the performance of IO write {dtype} to Parquet file using {comp} compression"
        )
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_IO_Write_Parquet")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("comp", COMPRESSIONS)
def bench_write_parquet_append(benchmark, dtype, comp):
    if (
        pytest.io_write
        or (not pytest.io_read and not pytest.io_delete)
        and dtype in pytest.dtype
        and comp in pytest.io_compression
    ):
        N = pytest.N

        a = _generate_df(N, dtype, True)

        benchmark.pedantic(_append_files, args=(a, dtype, comp), rounds=pytest.trials)

        num_bytes = 0
        for col in a.values():
            if dtype in ["int64", "float64", "uint64"]:
                num_bytes += calc_num_bytes(col) * pytest.io_files
            else:
                num_bytes += calc_num_bytes(col) * pytest.io_files

        benchmark.extra_info["description"] = (
            f"Measures the performance of IO write {dtype} to Parquet file using {comp} compression"
        )
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_IO_Read_HDF5")
@pytest.mark.parametrize("dtype", TYPES)
def bench_read_hdf(benchmark, dtype):
    if pytest.io_read or (not pytest.io_write and not pytest.io_delete) and dtype in pytest.dtype:
        dataset = "strings_array" if dtype == "str" else "array"
        a = benchmark.pedantic(
            ak.read_hdf,
            args=[pytest.io_path + f"_hdf_{dtype}*", dataset],
            rounds=pytest.trials,
        )

        num_bytes = calc_num_bytes(a)

        benchmark.extra_info["description"] = "Measures the performance of IO read from HDF5 files"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_IO_Read_Parquet")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("comp", COMPRESSIONS)
def bench_read_parquet(benchmark, dtype, comp):
    if (
        pytest.io_read
        or (not pytest.io_write and not pytest.io_delete)
        and comp in pytest.io_compression
        and dtype in pytest.dtype
    ):
        a = benchmark.pedantic(
            ak.read_parquet,
            args=[pytest.io_path + f"_par_{comp}_{dtype}_*"],
            rounds=pytest.trials,
        )

        num_bytes = calc_num_bytes(a)

        benchmark.extra_info["description"] = "Measures the performance of IO read from Parquet files"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_IO_Read_Parquet")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("comp", COMPRESSIONS)
def bench_read_parquet_multi(benchmark, dtype, comp):
    """Read files written by parquet multicolumn and parquet append modes."""
    if (
        pytest.io_read
        or (not pytest.io_write and not pytest.io_delete)
        and comp in pytest.io_compression
        and dtype in pytest.dtype
    ):
        a = benchmark.pedantic(
            ak.read_parquet,
            args=[pytest.io_path + f"_par_multi_{comp}_{dtype}_*"],
            rounds=pytest.trials,
        )

        num_bytes = calc_num_bytes(a)

        benchmark.extra_info["description"] = "Measures the performance of IO read from Parquet files"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_IO_Write_HDF5_Multi")
@pytest.mark.parametrize("dtype", TYPES)
def bench_write_hdf_multi(benchmark, dtype):
    if pytest.io_write or (not pytest.io_read and not pytest.io_delete) and dtype in pytest.dtype:
        N = pytest.N
        df = _generate_df(N, dtype)
        benchmark.pedantic(_write_multi_hdf, args=(df, dtype), rounds=pytest.trials)

        # compute num_bytes like in parquet multi
        num_bytes = calc_num_bytes(df)

        benchmark.extra_info["description"] = f"Measures IO write (multi) {dtype} to HDF5"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["num_bytes"] = num_bytes
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_IO_Read_HDF5_Multi")
@pytest.mark.parametrize("dtype", TYPES)
def bench_read_hdf_multi(benchmark, dtype):
    if pytest.io_read or (not pytest.io_write and not pytest.io_delete) and dtype in pytest.dtype:
        dataset = "strings_array" if dtype == "str" else "array"
        a = benchmark.pedantic(
            ak.read_hdf,
            args=[pytest.io_path + f"_hdf_multi_{dtype}_*", dataset],
            rounds=pytest.trials,
        )

        num_bytes = calc_num_bytes(a)

        benchmark.extra_info["description"] = "Measures IO read (multi) from HDF5 files"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["num_bytes"] = num_bytes
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_IO_Delete")
def bench_delete(benchmark):
    if pytest.io_delete or (not pytest.io_write and not pytest.io_read):
        benchmark.pedantic(_remove_files, rounds=1)

        benchmark.extra_info["description"] = "Measures the performance of IO delete files from system"
        benchmark.extra_info["problem_size"] = pytest.io_files * pytest.cfg["numNodes"]
        benchmark.extra_info["transfer_rate"] = "N/A"


def _remove_files():
    for f in glob(pytest.io_path + "*"):
        try:
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)
        except Exception as e:
            raise RuntimeWarning(f"Warning: Could not delete {f}: {e}")


def _remove_append_test_files(compression, dtype):
    for f in glob(f"{pytest.io_path}_par_multi_{compression}_{dtype}_app_" + "*"):
        os.remove(f)
