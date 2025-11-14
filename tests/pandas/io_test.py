import copy
import glob
import os
import shutil
import tempfile

from typing import List, Mapping, Union

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pandas.testing import assert_series_equal

import arkouda as ak

from arkouda import read_zarr, to_zarr
from arkouda.pandas import io_util
from arkouda.testing import assert_frame_equal


NUMERIC_TYPES = ["int64", "float64", "bool", "uint64"]
NUMERIC_AND_STR_TYPES = NUMERIC_TYPES + ["str"]

seed = pytest.seed


@pytest.fixture
def par_test_base_tmp(request):
    par_test_base_tmp = "{}/.par_io_test".format(pytest.temp_directory)
    io_util.get_directory(par_test_base_tmp)

    # Define a finalizer function for teardown
    def finalizer():
        # Clean up any resources if needed
        io_util.delete_directory(par_test_base_tmp)

    # Register the finalizer to ensure cleanup
    request.addfinalizer(finalizer)
    return par_test_base_tmp


@pytest.fixture
def hdf_test_base_tmp(request):
    hdf_test_base_tmp = "{}/.hdf_test".format(pytest.temp_directory)
    io_util.get_directory(hdf_test_base_tmp)

    with open("{}/not-a-file_LOCALE0000".format(hdf_test_base_tmp), "w"):
        pass

    # Define a finalizer function for teardown
    def finalizer():
        # Clean up any resources if needed
        io_util.delete_directory(hdf_test_base_tmp)

    # Register the finalizer to ensure cleanup
    request.addfinalizer(finalizer)
    return hdf_test_base_tmp


@pytest.fixture
def csv_test_base_tmp(request):
    csv_test_base_tmp = "{}/.csv_test".format(pytest.temp_directory)
    io_util.get_directory(csv_test_base_tmp)

    # Define a finalizer function for teardown
    def finalizer():
        # Clean up any resources if needed
        io_util.delete_directory(csv_test_base_tmp)

    # Register the finalizer to ensure cleanup
    request.addfinalizer(finalizer)
    return csv_test_base_tmp


@pytest.fixture
def zarr_test_base_tmp(request):
    zarr_test_base_tmp = "{}/.zarr_test".format(pytest.temp_directory)
    io_util.get_directory(zarr_test_base_tmp)

    # Define a finalizer function for teardown
    def finalizer():
        # Clean up any resources if needed
        io_util.delete_directory(zarr_test_base_tmp)

    # Register the finalizer to ensure cleanup
    request.addfinalizer(finalizer)
    return zarr_test_base_tmp


@pytest.fixture
def import_export_base_tmp(request):
    import_export_base_tmp = "{}/import_export_test".format(pytest.temp_directory)
    io_util.get_directory(import_export_base_tmp)

    # Define a finalizer function for teardown
    def finalizer():
        # Clean up any resources if needed
        io_util.delete_directory(import_export_base_tmp)

    # Register the finalizer to ensure cleanup
    request.addfinalizer(finalizer)
    return import_export_base_tmp


def make_ak_arrays(size, dtype):
    if dtype in ["int64", "float64"]:
        # randint for float is equivalent to uniform
        return ak.randint(-(2**32), 2**32, size=size, dtype=dtype)
    elif dtype == "uint64":
        return ak.cast(ak.randint(-(2**32), 2**32, size=size), dtype)
    elif dtype == "bool":
        return ak.randint(0, 1, size=size, dtype=dtype)
    elif dtype == "str":
        return ak.random_strings_uniform(1, 16, size=size)
    return None


def make_edge_case_arrays(dtype):
    if dtype == "int64":
        return np.array([np.iinfo(np.int64).min, -1, 0, 3, np.iinfo(np.int64).max], dtype=dtype)
    elif dtype == "uint64":
        return np.array([0, 1, 2**63 + 3, np.iinfo(np.uint64).max], dtype=dtype)
    elif dtype == "float64":
        return np.array(
            [
                np.nan,
                np.finfo(np.float64).min,
                -np.inf,
                -7.0,
                -3.14,
                -0.0,
                0.0,
                3.14,
                7.0,
                np.finfo(np.float64).max,
                np.inf,
                np.nan,
                np.nan,
                np.nan,
            ]
        )
    elif dtype == "bool":
        return np.array([True, False, False, True])
    elif dtype == "str":
        return np.array(['"', " ", ""])
    return None


def segarray_setup(dtype):
    if dtype in ["int64", "uint64"]:
        return [0, 1, 2], [1], [15, 21]
    elif dtype == "float64":
        return [1.1, 1.1, 2.7], [1.99], [15.2, 21.0]
    elif dtype == "bool":
        return [0, 1, 1], [0], [1, 0]
    elif dtype == "str":
        return ["one", "two", "three"], ["un", "deux", "trois"], ["uno", "dos", "tres"]
    return None


def edge_case_segarray_setup(dtype):
    if dtype == "int64":
        return [np.iinfo(np.int64).min, -1, 0], [1], [15, np.iinfo(np.int64).max]
    if dtype == "uint64":
        return [0, 1, 2**63 + 3], [0], [np.iinfo(np.uint64).max, 17]
    elif dtype == "float64":
        return [-0.0, np.finfo(np.float64).min, np.nan, 2.7], [1.99], [np.inf, np.nan]
    elif dtype == "bool":
        return [0, 1, 1], [0], [1, 0]
    elif dtype == "str":
        return ['"', " ", ""], ["test"], ["'", ""]
    return None


def make_multi_dtype_dict():
    return {
        "c_1": ak.array([np.iinfo(np.int64).min, -1, 0, np.iinfo(np.int64).max]),
        "c_2": ak.SegArray(ak.array([0, 0, 9, 14]), ak.arange(-10, 10)),
        "c_3": ak.arange(2**63 + 3, 2**63 + 7, dtype=ak.uint64),
        "c_4": ak.SegArray(ak.array([0, 5, 10, 10]), ak.arange(2**63, 2**63 + 15, dtype=ak.uint64)),
        "c_5": ak.array([False, True, False, False]),
        "c_6": ak.SegArray(ak.array([0, 0, 5, 10]), ak.randint(0, 1, 15, dtype=ak.bool_)),
        "c_7": ak.array([-0.0, np.finfo(np.float64).min, np.nan, np.inf]),
        "c_8": ak.SegArray(
            ak.array([0, 9, 14, 14]),
            ak.array(
                [
                    np.nan,
                    np.finfo(np.float64).min,
                    -np.inf,
                    -7.0,
                    -3.14,
                    -0.0,
                    0.0,
                    3.14,
                    7.0,
                    np.finfo(np.float64).max,
                    np.inf,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ),
        ),
        "c_9": ak.array(["abc", " ", "xyz", ""]),
        "c_10": ak.SegArray(
            ak.array([0, 2, 5, 5]),
            ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
        ),
        "c_11": ak.SegArray(
            ak.array([0, 2, 2, 2]),
            ak.array(["a", "b", "", "c", "d", "e", "f", "g", "h", "i"]),
        ),
        "c_12": ak.SegArray(
            ak.array([0, 0, 2, 2]),
            ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
        ),
        "c_13": ak.SegArray(
            ak.array([0, 2, 3, 3]),
            ak.array(["", "'", " ", "test", "", "'", "", " ", ""]),
        ),
        "c_14": ak.SegArray(
            ak.array([0, 5, 5, 8]),
            ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
        ),
        "c_15": ak.SegArray(
            ak.array([0, 5, 8, 8]),
            ak.array(["abc", "123", "xyz", "l", "m", "n", "o", "p", "arkouda"]),
        ),
    }


class TestParquet:
    COMPRESSIONS = [None, "snappy", "gzip", "brotli", "zstd", "lz4"]

    def test_io_docstrings(self, par_test_base_tmp):
        import doctest

        from arkouda import io

        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp_dirname)  # Change to temp directory
                result = doctest.testmod(io, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
            finally:
                os.chdir(old_cwd)  # Always return to original directory

            assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_read_and_write(self, par_test_base_tmp, prob_size, dtype, comp):
        ak_arr = make_ak_arrays(prob_size * pytest.nl, dtype)
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/pq_test_correct"
            ak_arr.to_parquet(file_name, "my-dset", compression=comp)
            pq_arr = ak.read_parquet(f"{file_name}*", "my-dset")["my-dset"]
            assert (ak_arr == pq_arr).all()

            # verify generic read works
            gen_arr = ak.read(f"{file_name}*", "my-dset")["my-dset"]
            assert (ak_arr == gen_arr).all()

            # verify generic load works
            gen_arr = ak.load(path_prefix=file_name, dataset="my-dset")["my-dset"]
            assert (ak_arr == gen_arr).all()

            # verify generic load works with file_format parameter
            gen_arr = ak.load(path_prefix=file_name, dataset="my-dset", file_format="Parquet")["my-dset"]
            assert (ak_arr == gen_arr).all()

            # verify load_all works
            gen_arr = ak.load_all(path_prefix=file_name)
            assert (ak_arr == gen_arr["my-dset"]).all()

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    def test_multi_file(self, par_test_base_tmp, prob_size, dtype):
        is_multi_loc = pytest.nl != 1
        NUM_FILES = pytest.nl if is_multi_loc else 2
        adjusted_size = int(prob_size / NUM_FILES) * NUM_FILES
        ak_arr = make_ak_arrays(adjusted_size, dtype)

        per_arr = int(adjusted_size / NUM_FILES)
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/pq_test"
            if is_multi_loc:
                # when multi locale multiple created automatically
                ak_arr.to_parquet(file_name, "test-dset")
            else:
                # when single locale artifically create multiple files
                for i in range(NUM_FILES):
                    arr_in_file_i = ak_arr[(i * per_arr) : (i * per_arr) + per_arr]
                    arr_in_file_i.to_parquet(f"{file_name}{i:04d}", "test-dset")

            assert len(glob.glob(f"{file_name}*")) == NUM_FILES
            pq_arr = ak.read_parquet(f"{file_name}*", "test-dset")["test-dset"]
            assert (ak_arr == pq_arr).all()

    def test_wrong_dset_name(self, par_test_base_tmp):
        ak_arr = ak.randint(0, 2**32, 100)
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/pq_test"
            ak_arr.to_parquet(file_name, "test-dset-name")

            with pytest.raises(RuntimeError):
                ak.read_parquet(f"{file_name}*", "wrong-dset-name")

            with pytest.raises(ValueError):
                ak.read_parquet(f"{file_name}*", ["test-dset-name", "wrong-dset-name"])

    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_edge_case_read_write(self, par_test_base_tmp, dtype, comp):
        np_edge_case = make_edge_case_arrays(dtype)
        ak_edge_case = ak.array(np_edge_case)
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            ak_edge_case.to_parquet(f"{tmp_dirname}/pq_test_edge_case", "my-dset", compression=comp)
            pq_arr = ak.read_parquet(f"{tmp_dirname}/pq_test_edge_case*", "my-dset")["my-dset"]
            if dtype == "float64":
                assert np.allclose(np_edge_case, pq_arr.to_ndarray(), equal_nan=True)
            else:
                assert (np_edge_case == pq_arr.to_ndarray()).all()

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("use_str", [True, False])
    @pytest.mark.parametrize("null_handling", ["none", "only floats", "all"])
    def test_large_parquet_io(self, par_test_base_tmp, prob_size, use_str, null_handling):
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            filename = f"{tmp_dirname}/pq_test_large_parquet"
            size = max(
                prob_size, 2**21 + 8
            )  # A problem had been detected with parquet files of > 2**21 entries
            bool_array = np.array((size // 2) * [True, False]).tolist()
            flt_array = np.arange(size).astype(np.float64).tolist()
            int_array = np.arange(size).astype(np.int64).tolist()
            arrays = [bool_array, int_array, flt_array]
            names = ["first", "second", "third"]
            if use_str:
                arrays.append(np.array(["a" + str(i) for i in np.arange(size)]).tolist())
                names.append("fourth")
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=names)
            s = pd.Series(np.random.randn(size), index=index)
            df = s.to_frame()
            df.to_parquet(filename)
            ak_df = ak.DataFrame(ak.read_parquet(filename, null_handling=null_handling))
            #  This check is on all of the random numbers generated in s
            assert np.all(ak_df.to_pandas().values[:, 0] == s.values)
            #  This check is on all of the elements of the MultiIndex
            for i in range(len(names)):
                assert np.all(
                    df.index.get_level_values(names[i]).to_numpy() == ak_df[names[i]].to_ndarray()
                )

    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    def test_get_datasets(self, par_test_base_tmp, dtype):
        ak_arr = make_ak_arrays(10, dtype)
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            ak_arr.to_parquet(f"{tmp_dirname}/pq_test", "TEST_DSET")
            dsets = ak.get_datasets(f"{tmp_dirname}/pq_test*")
            assert ["TEST_DSET"] == dsets

    def test_append(self, par_test_base_tmp):
        # use small size to cut down on execution time
        append_size = 32

        base_dset = ak.randint(0, 2**32, append_size)
        ak_dict = {dt: make_ak_arrays(append_size, dt) for dt in NUMERIC_AND_STR_TYPES}

        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/pq_test"
            base_dset.to_parquet(file_name, "base-dset")

            for key in ak_dict.keys():
                ak_dict[key].to_parquet(file_name, key, mode="append")

            ak_vals = ak.read_parquet(f"{file_name}*")

            for key in ak_dict:
                assert ak_vals[key].tolist() == ak_dict[key].tolist()

    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    def test_append_empty(self, par_test_base_tmp, dtype):
        # use small size to cut down on execution time
        ak_arr = make_ak_arrays(32, dtype)
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            ak_arr.to_parquet(f"{tmp_dirname}/pq_test_correct", "my-dset", mode="append")
            pq_arr = ak.read_parquet(f"{tmp_dirname}/pq_test_correct*", "my-dset")["my-dset"]

            assert ak_arr.tolist() == pq_arr.tolist()

    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_null_strings(self, par_test_base_tmp, comp):
        null_strings = ak.array(["first-string", "", "string2", "", "third", "", ""])
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/null_strings"
            null_strings.to_parquet(file_name, compression=comp)

            ak_data = ak.read_parquet(f"{file_name}*").popitem()[1]
            assert (null_strings == ak_data).all()

            # datasets must be specified for get_null_indices
            res = ak.get_null_indices(f"{file_name}*", datasets="strings_array").popitem()[1]
            assert [0, 1, 0, 1, 0, 1, 1] == res.tolist()

    def test_null_indices(self):
        datadir = "resources/parquet-testing"
        basename = "null-strings.parquet"

        filename = os.path.join(datadir, basename)
        res = ak.get_null_indices(filename, datasets="col1")["col1"]

        assert [0, 1, 0, 1, 0, 1, 1] == res.tolist()

    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_compression(self, par_test_base_tmp, comp):
        a = ak.arange(150)

        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            # write with the selected compression
            a.to_parquet(f"{tmp_dirname}/compress_test", compression=comp)

            # ensure read functions
            rd_arr = ak.read_parquet(f"{tmp_dirname}/compress_test*", "array")["array"]

            # validate the list read out matches the array used to write
            assert rd_arr.tolist() == a.tolist()

        b = ak.randint(0, 2, 150, dtype=ak.bool_)

        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            # write with the selected compression
            b.to_parquet(f"{tmp_dirname}/compress_test", compression=comp)

            # ensure read functions
            rd_arr = ak.read_parquet(f"{tmp_dirname}/compress_test*", "array")["array"]

            # validate the list read out matches the array used to write
            assert rd_arr.tolist() == b.tolist()

    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_nan_compressions(self, par_test_base_tmp, comp):
        # Reproducer for issue #2005 specifically for gzip
        pdf = pd.DataFrame(
            {
                "all_nan": np.array([np.nan, np.nan, np.nan, np.nan]),
                "some_nan": np.array([3.14, np.nan, 7.12, 4.44]),
            }
        )

        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            pdf.to_parquet(f"{tmp_dirname}/nan_compressed_pq", engine="pyarrow", compression=comp)

            ak_data = ak.read_parquet(f"{tmp_dirname}/nan_compressed_pq")
            rd_df = ak.DataFrame(ak_data)
            pd.testing.assert_frame_equal(rd_df.to_pandas(), pdf)

    def test_gzip_nan_rd(self, par_test_base_tmp):
        # create pandas dataframe
        pdf = pd.DataFrame(
            {
                "all_nan": np.array([np.nan, np.nan, np.nan, np.nan]),
                "some_nan": np.array([3.14, np.nan, 7.12, 4.44]),
            }
        )

        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            pdf.to_parquet(f"{tmp_dirname}/gzip_pq", engine="pyarrow", compression="gzip")

            ak_data = ak.read_parquet(f"{tmp_dirname}/gzip_pq")
            rd_df = ak.DataFrame(ak_data)
            assert pdf.equals(rd_df.to_pandas())

    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_segarray_read(self, par_test_base_tmp, comp):
        df = pd.DataFrame(
            {
                "IntList": [
                    [np.iinfo(np.int64).max],
                    [0, 1, -2],
                    [],
                    [3, -4, np.iinfo(np.int64).min, 6],
                    [-1, 2, 3],
                ],
                "BoolList": [[False], [True, False], [False, False, False], [True], []],
                "FloatList": [
                    [np.finfo(np.float64).max],
                    [3.14, np.nan, np.finfo(np.float64).min, 2.23, 3.08],
                    [],
                    [np.inf, 6.8],
                    [-0.0, np.nan, np.nan, np.nan],
                ],
                "UintList": [
                    np.array([], np.uint64),
                    np.array([1, 2**64 - 2], np.uint64),
                    np.array([2**63 + 1], np.uint64),
                    np.array([2, 2, 0], np.uint64),
                    np.array([11], np.uint64),
                ],
                "StringsList": [['"', " ", ""], [], ["test"], [], ["'", ""]],
                "EmptySegList": [[], [0, 1], [], [3, 4, 5, 6], []],
            }
        )
        table = pa.Table.from_pandas(df)
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/segarray_parquet"
            pq.write_table(table, file_name, compression=comp)

            # verify full file read with various object types
            ak_data = ak.read_parquet(f"{file_name}*")
            for k, v in ak_data.items():
                assert isinstance(v, ak.SegArray)
                for x, y in zip(df[k].tolist(), v.tolist()):
                    if isinstance(x, np.ndarray):
                        x = x.tolist()
                    assert x == y if k != "FloatList" else np.allclose(x, y, equal_nan=True)

            # verify individual column selection
            for k, v in df.items():
                ak_data = ak.read_parquet(f"{file_name}*", datasets=k)[k]
                assert isinstance(ak_data, ak.SegArray)
                for x, y in zip(v.tolist(), ak_data.tolist()):
                    if isinstance(x, np.ndarray):
                        x = x.tolist()
                    assert x == y if k != "FloatList" else np.allclose(x, y, equal_nan=True)

        # test for handling empty segments only reading single segarray
        df = pd.DataFrame({"ListCol": [[8], [0, 1], [], [3, 4, 5, 6], []]})
        table = pa.Table.from_pandas(df)
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/empty_segments", compression=comp)

            ak_data = ak.read_parquet(f"{tmp_dirname}/empty_segments*")["ListCol"]
            assert isinstance(ak_data, ak.SegArray)
            assert ak_data.size == 5
            for i in range(5):
                assert df["ListCol"][i] == ak_data[i].tolist()

        df = pd.DataFrame(
            {
                "IntCol": [0, 1, 2, 3],
                "ListCol": [[0, 1, 2], [0, 1], [3, 4, 5, 6], [1, 2, 3]],
            }
        )
        table = pa.Table.from_pandas(df)
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/segarray_varied_parquet"
            pq.write_table(table, file_name, compression=comp)

            # read full file
            ak_data = ak.read_parquet(f"{file_name}*")
            for k, v in ak_data.items():
                assert df[k].tolist() == v.tolist()

            # read individual datasets
            ak_data = ak.read_parquet(f"{file_name}*", datasets="IntCol")["IntCol"]
            assert isinstance(ak_data, ak.pdarray)
            assert df["IntCol"].tolist() == ak_data.tolist()
            ak_data = ak.read_parquet(f"{file_name}*", datasets="ListCol")["ListCol"]
            assert isinstance(ak_data, ak.SegArray)
            assert df["ListCol"].tolist() == ak_data.tolist()

        # test for multi-file with and without empty segs
        is_multi_loc = pytest.nl != 1
        NUM_FILES = pytest.nl if is_multi_loc else 2
        regular = (
            [[0, 1, 2], [0, 1], [3, 4, 5, 6], [1, 2, 3]],
            [[0, 1, 11], [0, 1], [3, 4, 5, 6], [1]],
        )
        first_empty = (
            [[], [0, 1], [], [3, 4, 5, 6], []],
            [[0, 1], [], [3, 4, 5, 6], [], [1, 2, 3]],
        )
        # there are two empty segs tests with only difference being the first segment being [8] not []
        # including to avoid loss of coverage
        # use deepcopy to avoid changing first_empty
        first_non_empty = copy.deepcopy(first_empty)
        first_non_empty[0][0] = [8]
        for args in [regular, first_empty, first_non_empty]:
            lists = [args[i % 2] for i in range(NUM_FILES)]
            dataframes = [pd.DataFrame({"ListCol": li}) for li in lists]
            tables = [pa.Table.from_pandas(df) for df in dataframes]
            combo = pd.concat(dataframes, ignore_index=True)
            with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
                file_name = f"{tmp_dirname}/segarray_varied_parquet"
                if is_multi_loc:
                    # when multi locale multiple created automatically
                    # so create concatenated segarray and write using arkouda
                    # have to set dtype to avoid empty list being created as float type
                    concat_segarr = ak.SegArray.concat(
                        [
                            ak.SegArray.from_multi_array([ak.array(a, dtype=ak.int64) for a in li])
                            for li in lists
                        ]
                    )
                    concat_segarr.to_parquet(file_name, "ListCol", compression=comp)
                else:
                    # when single locale artifically create multiple files
                    for i in range(NUM_FILES):
                        pq.write_table(tables[i], f"{file_name}_LOCALE{i:04d}", compression=comp)
                ak_data = ak.read_parquet(f"{file_name}*")["ListCol"]
                assert isinstance(ak_data, ak.SegArray)
                assert ak_data.size == len(lists[0]) * NUM_FILES
                for i in range(ak_data.size):
                    assert combo["ListCol"][i] == ak_data[i].tolist()

    def test_segarray_string(self, par_test_base_tmp):
        words = ak.array(["one,two,three", "uno,dos,tres"])
        strs, segs = words.regex_split(",", return_segments=True)
        x = ak.SegArray(segs, strs)

        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            x.to_parquet(f"{tmp_dirname}/segarr_str")

            rd = ak.read_parquet(f"{tmp_dirname}/segarr_str_*").popitem()[1]
            assert isinstance(rd, ak.SegArray)
            assert x.segments.tolist() == rd.segments.tolist()
            assert x.values.tolist() == rd.values.tolist()
            assert x.tolist() == rd.tolist()

        # additional testing for empty segments. See Issue #2560
        a, c = (
            ["one", "two", "three"],
            ["uno", "dos", "tres"],
        )
        s = ak.SegArray(ak.array([0, 0, len(a), len(a), len(a), len(a) + len(c)]), ak.array(a + c))
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/segarray_test_empty")
            rd_data = ak.read_parquet(f"{tmp_dirname}/segarray_test_empty_*").popitem()[1]
            assert s.tolist() == rd_data.tolist()

    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    @pytest.mark.parametrize("segarray_create", [segarray_setup, edge_case_segarray_setup])
    def test_segarray_write(self, par_test_base_tmp, dtype, segarray_create):
        a, b, c = segarray_create(dtype)
        s = ak.SegArray(ak.array([0, len(a), len(a) + len(b)]), ak.array(a + b + c))
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/segarray_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/segarray_test*").popitem()[1]
            for i in range(3):
                x, y = s[i].tolist(), rd_data[i].tolist()
                assert x == y if dtype != "float64" else np.allclose(x, y, equal_nan=True)

        s = ak.SegArray(ak.array([0, 0, len(a), len(a), len(a), len(a) + len(c)]), ak.array(a + c))
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/segarray_test_empty")

            rd_data = ak.read_parquet(f"{tmp_dirname}/segarray_test_empty*").popitem()[1]
            for i in range(6):
                x, y = s[i].tolist(), rd_data[i].tolist()
                assert x == y if dtype != "float64" else np.allclose(x, y, equal_nan=True)

    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_multi_col_write(self, par_test_base_tmp, comp):
        df_dict = make_multi_dtype_dict()
        akdf = ak.DataFrame(df_dict)
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            # use multi-column write to generate parquet file
            akdf.to_parquet(f"{tmp_dirname}/multi_col_parquet", compression=comp)
            # read files and ensure that all resulting fields are as expected
            rd_data = ak.read_parquet(f"{tmp_dirname}/multi_col_parquet*")
            rd_df = ak.DataFrame(rd_data)
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            # test save with index true
            akdf.to_parquet(f"{tmp_dirname}/idx_multi_col_parquet", index=True, compression=comp)
            rd_data = ak.read_parquet(f"{tmp_dirname}/idx_multi_col_parquet*")
            rd_df = ak.DataFrame(rd_data)
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

    def test_small_ints(self, par_test_base_tmp):
        df_pd = pd.DataFrame(
            {
                "int16": pd.Series([2**15 - 1, -(2**15)], dtype=np.int16),
                "int32": pd.Series([2**31 - 1, -(2**31)], dtype=np.int32),
                "uint16": pd.Series([2**15 - 1, 2**15], dtype=np.uint16),
                "uint32": pd.Series([2**31 - 1, 2**31], dtype=np.uint32),
            }
        )
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/pq_small_int"
            df_pd.to_parquet(file_name)
            df_ak = ak.DataFrame(ak.read_parquet(f"{file_name}*"))
            for c in df_ak.columns.values:
                assert df_ak[c].tolist() == df_pd[c].tolist()

    def test_read_nested(self, par_test_base_tmp):
        df = ak.DataFrame(
            {
                "idx": ak.arange(5),
                "seg": ak.SegArray(ak.arange(0, 10, 2), ak.arange(10)),
            }
        )
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/read_nested_test"
            df.to_parquet(file_name)

            # test read with read_nested=true
            data = ak.read_parquet(f"{file_name}*")
            assert "idx" in data
            assert "seg" in data
            assert df["idx"].tolist() == data["idx"].tolist()
            assert df["seg"].tolist() == data["seg"].tolist()

            # test read with read_nested=false and no supplied datasets
            data = ak.read_parquet(f"{file_name}*", read_nested=False)["idx"]
            assert isinstance(data, ak.pdarray)
            assert df["idx"].tolist() == data.tolist()

            # test read with read_nested=false and user supplied datasets. Should ignore read_nested
            data = ak.read_parquet(f"{file_name}*", datasets=["idx", "seg"], read_nested=False)
            assert "idx" in data
            assert "seg" in data
            assert df["idx"].tolist() == data["idx"].tolist()
            assert df["seg"].tolist() == data["seg"].tolist()

    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_ipv4_columns(self, par_test_base_tmp, comp):
        # Added as reproducer for issue #2337
        # test with single IPv4 column
        df = ak.DataFrame({"a": ak.arange(10), "b": ak.IPv4(ak.arange(10))})
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/ipv4_df"
            df.to_parquet(file_name, compression=comp)

            data = ak.read_parquet(f"{file_name}*")
            rd_df = ak.DataFrame({"a": data["a"], "b": ak.IPv4(data["b"])})

            pd.testing.assert_frame_equal(df.to_pandas(), rd_df.to_pandas())

        # test with multiple IPv4 columns
        df = ak.DataFrame({"a": ak.IPv4(ak.arange(10)), "b": ak.IPv4(ak.arange(10))})
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/ipv4_df"
            df.to_parquet(file_name, compression=comp)

            data = ak.read_parquet(f"{file_name}*")
            rd_df = ak.DataFrame({"a": ak.IPv4(data["a"]), "b": ak.IPv4(data["b"])})

            pd.testing.assert_frame_equal(df.to_pandas(), rd_df.to_pandas())

        # test replacement of IPv4 with uint representation
        df = ak.DataFrame({"a": ak.IPv4(ak.arange(10))})
        df["a"] = df["a"].export_uint()
        assert ak.arange(10).tolist() == df["a"].tolist()

    def test_decimal_reads(self, par_test_base_tmp):
        cols = []
        data = []
        min_prec = 1
        max_prec = 39
        for i in range(min_prec, max_prec):
            cols.append(("decCol" + str(i), pa.decimal128(i, 0)))
            data.append([i])

        schema = pa.schema(cols)

        table = pa.Table.from_arrays(data, schema=schema)
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/decimal")
            ak_data = ak.read(f"{tmp_dirname}/decimal")
            for idx, i in enumerate(range(min_prec, max_prec)):
                assert np.allclose(ak_data["decCol" + str(i)].to_ndarray(), data[idx])

    def test_multi_batch_reads(self, par_test_base_tmp):
        # verify reproducer for #3074 is resolved
        # seagarray w/ empty segs multi-batch pq reads

        # bug seemed to consistently appear for val_sizes
        # exceeding 700000 (likely due to this requiring more than one batch)
        # we round up to ensure we'd hit it
        val_size = 1000000

        df_dict = dict()

        # This retains the use of default_rng to generate a seed, but it seeds the
        # rng itself with pytest.seed.

        iseed = np.random.default_rng(seed).choice(2**63)
        rng = ak.random.default_rng(iseed)
        some_nans = rng.uniform(-(2**10), 2**10, val_size)
        some_nans[ak.arange(val_size) % 2 == 0] = np.nan
        vals_list = [
            rng.uniform(-(2**10), 2**10, val_size),
            rng.integers(0, 2**32, size=val_size, dtype="uint"),
            rng.integers(0, 1, size=val_size, dtype="bool"),
            rng.integers(-(2**32), 2**32, size=val_size, dtype="int"),
            some_nans,  # contains nans
            ak.random_strings_uniform(0, 4, val_size, seed=iseed),  # contains empty strings
        ]

        for vals in vals_list:
            # segs must start with 0, all other segment lengths are random
            # by having val_size number of segments, except in the extremely unlikely case of
            # randomly getting exactly arange(val_size), we are guaranteed empty segs
            segs = ak.concatenate(
                [
                    ak.array([0]),
                    ak.sort(ak.randint(0, val_size, val_size - 1, seed=iseed)),
                ]
            )
            df_dict["rand"] = ak.SegArray(segs, vals).tolist()

            pddf = pd.DataFrame(df_dict)
            with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
                file_path = f"{tmp_dirname}/empty_segs"
                pddf.to_parquet(file_path)
                akdf = ak.DataFrame(ak.read_parquet(file_path))

                to_pd = pd.Series(akdf["rand"].tolist())
                # raises an error if the two series aren't equal
                # we can't use np.allclose(pddf['rand'].tolist, akdf['rand'].tolist) since these
                # are lists of lists. assert_series_equal handles this and properly handles nans.
                # we pass the same absolute and relative tolerances as the numpy default in allclose
                # to ensure float point differences don't cause errors
                print("\nseed: ", iseed)
                assert_series_equal(pddf["rand"], to_pd, check_names=False, rtol=1e-05, atol=1e-08)

                # test writing multi-batch non-segarrays
                file_path = f"{tmp_dirname}/multi_batch_vals"
                vals.to_parquet(file_path, dataset="my_vals")
                read = ak.read_parquet(file_path + "*")["my_vals"]
                if isinstance(vals, ak.pdarray) and vals.dtype == ak.float64:
                    assert np.allclose(read.tolist(), vals.tolist(), equal_nan=True)
                else:
                    assert (read == vals).all()

    @pytest.mark.skip_if_nl_eq(2)
    def test_bug_4076_reproducer(self, par_test_base_tmp):
        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            source_dir = "{}/resources/parquet-testing/bug_4076_reproducer".format(os.getcwd())
            for filename in os.listdir(source_dir):
                shutil.copy(source_dir + "/" + filename, tmp_dirname)

            df = ak.DataFrame(ak.read_parquet(tmp_dirname + "/df_*"))
            assert df.size == 100

            #   Save a copy for comparison later
            df_copy = df.copy(deep=True)

            #   Write it again
            df.to_parquet(tmp_dirname + "/df")

            #   Read it in again
            df = ak.DataFrame(ak.read_parquet(tmp_dirname + "/df_*"))
            assert_frame_equal(df_copy, df)

            df2 = ak.DataFrame(ak.read_parquet(tmp_dirname + "/df2*"))
            assert df2.size == 111

            df3 = ak.DataFrame(ak.read_parquet(tmp_dirname + "/d_*"))
            assert df3.size == 119

    @pytest.mark.optional_parquet
    def test_against_standard_files(self):
        datadir = "resources/parquet-testing"
        filenames = [
            "alltypes_plain.parquet",
            "alltypes_plain.snappy.parquet",
            "delta_byte_array.parquet",
        ]
        columns1 = [
            "id",
            "bool_col",
            "tinyint_col",
            "smallint_col",
            "int_col",
            "bigint_col",
            "float_col",
            "double_col",
            "date_string_col",
            "string_col",
            "timestamp_col",
        ]
        columns2 = [
            "c_customer_id",
            "c_salutation",
            "c_first_name",
            "c_last_name",
            "c_preferred_cust_flag",
            "c_birth_country",
            "c_login",
            "c_email_address",
            "c_last_review_date",
        ]
        for basename, ans in zip(filenames, (columns1, columns1, columns2)):
            filename = os.path.join(datadir, basename)
            columns = ak.get_datasets(filename)
            assert columns == ans
            # Merely test that read succeeds, do not check output
            if "delta_byte_array.parquet" not in filename:
                ak.read_parquet(filename, datasets=columns)
            else:
                # Since delta encoding is not supported, the columns in
                # this file should raise an error and not crash the server
                with pytest.raises(RuntimeError):
                    ak.read_parquet(filename, datasets=columns)

    def test_null_handling_all(self, par_test_base_tmp):
        df = pd.DataFrame(
            {
                "ints": [1, None, 3, 4, None, 6],  # pandas None → null
                "floats": [1.0, 2.5, None, 4.2, None, 6.7],
                "doubles": [1.0, None, 3.14, None, 5.55, 6.66],
            }
        )

        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            filename = f"{tmp_dirname}/null_handling_all.parquet"
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, filename)

            ak_df = ak.DataFrame(ak.read_parquet(filename, null_handling="all"))

            pd.testing.assert_frame_equal(df, ak_df.to_pandas())

    def test_null_handling_only_floats(self, par_test_base_tmp):
        df = pd.DataFrame(
            {
                "floats": [1.0, 2.5, None, 4.2, None, 6.7],
                "doubles": [1.0, None, 3.14, None, 5.55, 6.66],
            }
        )

        with tempfile.TemporaryDirectory(dir=par_test_base_tmp) as tmp_dirname:
            filename = f"{tmp_dirname}/null_handling_only_floats.parquet"
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, filename)

            ak_df = ak.DataFrame(ak.read_parquet(filename, null_handling="only floats"))

            pd.testing.assert_frame_equal(df, ak_df.to_pandas())

    def test_incorrect_null_handling(self):
        try:
            ak.read_parquet("bogus", null_handling="bogus")
        except RuntimeError as err:
            assert "null_handling argument only accepts" in str(err)

    def test_deprecated_has_non_float_nulls(self):
        with pytest.deprecated_call():
            try:
                ak.read_parquet("bogus", has_non_float_nulls=True)
            except RuntimeError as err:
                assert "File bogus does not exist" in str(err)

    def test_both_null_warning(self):
        with pytest.warns(UserWarning):
            try:
                ak.read_parquet("bogus", has_non_float_nulls=True, null_handling="only floats")
            except RuntimeError as err:
                assert "File bogus does not exist" in str(err)


class TestHDF5:
    @pytest.fixture(autouse=True)
    def set_attributes(self):
        self.int_tens_pdarray = ak.array(np.random.randint(-100, 100, 1000))
        self.int_tens_ndarray = self.int_tens_pdarray.to_ndarray()
        self.int_tens_ndarray.sort()
        self.int_tens_pdarray_dupe = ak.array(np.random.randint(-100, 100, 1000))

        self.int_hundreds_pdarray = ak.array(np.random.randint(-1000, 1000, 1000))
        self.int_hundreds_ndarray = self.int_hundreds_pdarray.to_ndarray()
        self.int_hundreds_ndarray.sort()
        self.int_hundreds_pdarray_dupe = ak.array(np.random.randint(-1000, 1000, 1000))

        self.float_pdarray = ak.array(np.random.uniform(-100, 100, 1000))
        self.float_ndarray = self.float_pdarray.to_ndarray()
        self.float_ndarray.sort()
        self.float_pdarray_dupe = ak.array(np.random.uniform(-100, 100, 1000))

        self.bool_pdarray = ak.randint(0, 1, 1000, dtype=ak.bool_)
        self.bool_pdarray_dupe = ak.randint(0, 1, 1000, dtype=ak.bool_)

        self.dict_columns = {
            "int_tens_pdarray": self.int_tens_pdarray,
            "int_hundreds_pdarray": self.int_hundreds_pdarray,
            "float_pdarray": self.float_pdarray,
            "bool_pdarray": self.bool_pdarray,
        }

        self.dict_columns_dupe = {
            "int_tens_pdarray": self.int_tens_pdarray_dupe,
            "int_hundreds_pdarray": self.int_hundreds_pdarray_dupe,
            "float_pdarray": self.float_pdarray_dupe,
            "bool_pdarray": self.bool_pdarray_dupe,
        }

        self.dict_single_column = {"int_tens_pdarray": self.int_tens_pdarray}

        self.list_columns = [
            self.int_tens_pdarray,
            self.int_hundreds_pdarray,
            self.float_pdarray,
            self.bool_pdarray,
        ]

        self.names = [
            "int_tens_pdarray",
            "int_hundreds_pdarray",
            "float_pdarray",
            "bool_pdarray",
        ]

    def _create_file(
        self,
        prefix_path: str,
        columns: Union[Mapping[str, ak.array]],
        names: List[str] = None,
    ) -> None:
        """
        Creates an hdf5 file with dataset(s) from the specified columns and path prefix
        via the ak.save_all method. If columns is a List, then the names list is used
        to create the datasets

        :return: None
        :raise: ValueError if the names list is None when columns is a list
        """
        if isinstance(columns, dict):
            ak.to_hdf(columns=columns, prefix_path=prefix_path)
        else:
            if not names:
                raise ValueError("the names list must be not None if columns is a list")
            ak.to_hdf(columns=columns, prefix_path=prefix_path, names=names)

    def test_save_all_load_all_with_dict(self, hdf_test_base_tmp):
        """
        Creates 2..n files from an input columns dict depending upon the number of
        arkouda_server locales, retrieves all datasets and correspoding pdarrays,
        and confirms they match inputs

        :return: None
        :raise: AssertionError if the input and returned datasets and pdarrays don't match
        """
        self._create_file(
            columns=self.dict_columns,
            prefix_path="{}/iotest_dict".format(hdf_test_base_tmp),
        )
        retrieved_columns = ak.load_all("{}/iotest_dict".format(hdf_test_base_tmp))

        itp = self.dict_columns["int_tens_pdarray"].to_ndarray()
        ritp = retrieved_columns["int_tens_pdarray"].to_ndarray()
        itp.sort()
        ritp.sort()
        ihp = self.dict_columns["int_hundreds_pdarray"].to_ndarray()
        rihp = retrieved_columns["int_hundreds_pdarray"].to_ndarray()
        ihp.sort()
        rihp.sort()
        ifp = self.dict_columns["float_pdarray"].to_ndarray()
        rifp = retrieved_columns["float_pdarray"].to_ndarray()
        ifp.sort()
        rifp.sort()

        assert 4 == len(retrieved_columns)
        assert itp.tolist() == ritp.tolist()
        assert ihp.tolist() == rihp.tolist()
        assert ifp.tolist() == rifp.tolist()
        assert len(self.dict_columns["bool_pdarray"]) == len(retrieved_columns["bool_pdarray"])
        assert 4 == len(ak.get_datasets("{}/iotest_dict_LOCALE0000".format(hdf_test_base_tmp)))

    def test_save_all_load_all_with_list(self, hdf_test_base_tmp):
        """
        Creates 2..n files from an input columns and names list depending upon the number of
        arkouda_server locales, retrieves all datasets and correspoding pdarrays, and confirms
        they match inputs

        :return: None
        :raise: AssertionError if the input and returned datasets and pdarrays don't match
        """
        self._create_file(
            columns=self.list_columns,
            prefix_path="{}/iotest_list".format(hdf_test_base_tmp),
            names=self.names,
        )
        retrieved_columns = ak.load_all(path_prefix="{}/iotest_list".format(hdf_test_base_tmp))

        itp = self.list_columns[0].to_ndarray()
        itp.sort()
        ritp = retrieved_columns["int_tens_pdarray"].to_ndarray()
        ritp.sort()
        ihp = self.list_columns[1].to_ndarray()
        ihp.sort()
        rihp = retrieved_columns["int_hundreds_pdarray"].to_ndarray()
        rihp.sort()
        fp = self.list_columns[2].to_ndarray()
        fp.sort()
        rfp = retrieved_columns["float_pdarray"].to_ndarray()
        rfp.sort()

        assert 4 == len(retrieved_columns)
        assert itp.tolist() == ritp.tolist()
        assert ihp.tolist() == rihp.tolist()
        assert fp.tolist() == rfp.tolist()
        assert len(self.list_columns[3]) == len(retrieved_columns["bool_pdarray"])
        assert 4 == len(ak.get_datasets("{}/iotest_list_LOCALE0000".format(hdf_test_base_tmp)))

    def test_read_hdf(self, hdf_test_base_tmp):
        """
        Creates 2..n files depending upon the number of arkouda_server locales, reads the files
        with an explicit list of file names to the read_all method, and confirms the datasets
        and embedded pdarrays match the input dataset and pdarrays

        :return: None
        :raise: AssertionError if the input and returned datasets don't match
        """
        self._create_file(
            columns=self.dict_columns,
            prefix_path="{}/iotest_dict_columns".format(hdf_test_base_tmp),
        )

        # test with read_hdf
        dataset = ak.read_hdf(filenames=["{}/iotest_dict_columns_LOCALE0000".format(hdf_test_base_tmp)])
        assert 4 == len(list(dataset.keys()))

        # test with generic read function
        dataset = ak.read(filenames=["{}/iotest_dict_columns_LOCALE0000".format(hdf_test_base_tmp)])
        assert 4 == len(list(dataset.keys()))

    def test_read_hdf_with_glob(self, hdf_test_base_tmp):
        """
        Creates 2..n files depending upon the number of arkouda_server locales with two
        files each containing different-named datasets with the same pdarrays, reads the files
        with the glob feature of the read_all method, and confirms the datasets and embedded
        pdarrays match the input dataset and pdarrays

        :return: None
        :raise: AssertionError if the input and returned datasets don't match
        """
        self._create_file(
            columns=self.dict_columns,
            prefix_path="{}/iotest_dict_columns".format(hdf_test_base_tmp),
        )

        retrieved_columns = ak.read_hdf(filenames="{}/iotest_dict_columns*".format(hdf_test_base_tmp))

        itp = self.list_columns[0].to_ndarray()
        itp.sort()
        ritp = retrieved_columns["int_tens_pdarray"].to_ndarray()
        ritp.sort()
        ihp = self.list_columns[1].to_ndarray()
        ihp.sort()
        rihp = retrieved_columns["int_hundreds_pdarray"].to_ndarray()
        rihp.sort()
        fp = self.list_columns[2].to_ndarray()
        fp.sort()
        rfp = retrieved_columns["float_pdarray"].to_ndarray()
        rfp.sort()

        assert 4 == len(list(retrieved_columns.keys()))
        assert itp.tolist() == ritp.tolist()
        assert ihp.tolist() == rihp.tolist()
        assert fp.tolist() == rfp.tolist()
        assert len(self.bool_pdarray) == len(retrieved_columns["bool_pdarray"])

    def test_load(self, hdf_test_base_tmp):
        """
        Creates 1..n files depending upon the number of arkouda_server locales with three columns
        AKA datasets, loads each corresponding dataset and confirms each corresponding pdarray
        equals the input pdarray.

        :return: None
        :raise: AssertionError if the input and returned datasets (pdarrays) don't match
        """
        self._create_file(
            columns=self.dict_columns,
            prefix_path="{}/iotest_dict_columns".format(hdf_test_base_tmp),
        )
        result_array_tens = ak.load(
            path_prefix="{}/iotest_dict_columns".format(hdf_test_base_tmp),
            dataset="int_tens_pdarray",
        )["int_tens_pdarray"]
        result_array_hundreds = ak.load(
            path_prefix="{}/iotest_dict_columns".format(hdf_test_base_tmp),
            dataset="int_hundreds_pdarray",
        )["int_hundreds_pdarray"]
        result_array_floats = ak.load(
            path_prefix="{}/iotest_dict_columns".format(hdf_test_base_tmp),
            dataset="float_pdarray",
        )["float_pdarray"]
        result_array_bools = ak.load(
            path_prefix="{}/iotest_dict_columns".format(hdf_test_base_tmp),
            dataset="bool_pdarray",
        )["bool_pdarray"]

        ratens = result_array_tens.to_ndarray()
        ratens.sort()

        rahundreds = result_array_hundreds.to_ndarray()
        rahundreds.sort()

        rafloats = result_array_floats.to_ndarray()
        rafloats.sort()

        assert self.int_tens_ndarray.tolist() == ratens.tolist()
        assert self.int_hundreds_ndarray.tolist() == rahundreds.tolist()
        assert self.float_ndarray.tolist() == rafloats.tolist()
        assert len(self.bool_pdarray) == len(result_array_bools)

        # test load_all with file_format parameter usage
        ak.to_parquet(
            columns=self.dict_columns,
            prefix_path="{}/iotest_dict_columns_parquet".format(hdf_test_base_tmp),
        )
        result_array_tens = ak.load(
            path_prefix="{}/iotest_dict_columns_parquet".format(hdf_test_base_tmp),
            dataset="int_tens_pdarray",
            file_format="Parquet",
        )["int_tens_pdarray"]
        result_array_hundreds = ak.load(
            path_prefix="{}/iotest_dict_columns_parquet".format(hdf_test_base_tmp),
            dataset="int_hundreds_pdarray",
            file_format="Parquet",
        )["int_hundreds_pdarray"]
        result_array_floats = ak.load(
            path_prefix="{}/iotest_dict_columns_parquet".format(hdf_test_base_tmp),
            dataset="float_pdarray",
            file_format="Parquet",
        )["float_pdarray"]
        result_array_bools = ak.load(
            path_prefix="{}/iotest_dict_columns_parquet".format(hdf_test_base_tmp),
            dataset="bool_pdarray",
            file_format="Parquet",
        )["bool_pdarray"]
        ratens = result_array_tens.to_ndarray()
        ratens.sort()

        rahundreds = result_array_hundreds.to_ndarray()
        rahundreds.sort()

        rafloats = result_array_floats.to_ndarray()
        rafloats.sort()
        assert self.int_tens_ndarray.tolist() == ratens.tolist()
        assert self.int_hundreds_ndarray.tolist() == rahundreds.tolist()
        assert self.float_ndarray.tolist() == rafloats.tolist()
        assert len(self.bool_pdarray) == len(result_array_bools)

        # Test load with invalid prefix
        with pytest.raises(RuntimeError):
            ak.load(
                path_prefix="{}/iotest_dict_column".format(hdf_test_base_tmp),
                dataset="int_tens_pdarray",
            )["int_tens_pdarray"]

        # Test load with invalid file
        with pytest.raises(RuntimeError):
            ak.load(
                path_prefix="{}/not-a-file".format(hdf_test_base_tmp),
                dataset="int_tens_pdarray",
            )["int_tens_pdarray"]

    def test_load_all(self, hdf_test_base_tmp):
        self._create_file(
            columns=self.dict_columns,
            prefix_path="{}/iotest_dict_columns".format(hdf_test_base_tmp),
        )

        results = ak.load_all(path_prefix="{}/iotest_dict_columns".format(hdf_test_base_tmp))
        assert "bool_pdarray" in results
        assert "float_pdarray" in results
        assert "int_tens_pdarray" in results
        assert "int_hundreds_pdarray" in results

        # test load_all with file_format parameter usage
        ak.to_parquet(
            columns=self.dict_columns,
            prefix_path="{}/iotest_dict_columns_parquet".format(hdf_test_base_tmp),
        )
        results = ak.load_all(
            file_format="Parquet",
            path_prefix="{}/iotest_dict_columns_parquet".format(hdf_test_base_tmp),
        )
        assert "bool_pdarray" in results
        assert "float_pdarray" in results
        assert "int_tens_pdarray" in results
        assert "int_hundreds_pdarray" in results

        # # Test load_all with invalid prefix
        with pytest.raises(ValueError):
            ak.load_all(path_prefix="{}/iotest_dict_column".format(hdf_test_base_tmp))

        # Test load with invalid file
        with pytest.raises(RuntimeError):
            ak.load_all(path_prefix="{}/not-a-file".format(hdf_test_base_tmp))

    def test_get_data_sets(self, hdf_test_base_tmp):
        """
        Creates 1..n files depending upon the number of arkouda_server locales containing three
        datasets and confirms the expected number of datasets along with the dataset names

        :return: None
        :raise: AssertionError if the input and returned dataset names don't match
        """
        self._create_file(
            columns=self.dict_columns,
            prefix_path="{}/iotest_dict_columns".format(hdf_test_base_tmp),
        )
        datasets = ak.get_datasets("{}/iotest_dict_columns_LOCALE0000".format(hdf_test_base_tmp))

        assert 4 == len(datasets)
        for dataset in datasets:
            assert dataset in self.names

        # Test load_all with invalid filename
        with pytest.raises(RuntimeError):
            ak.get_datasets("{}/iotest_dict_columns_LOCALE000".format(hdf_test_base_tmp))

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    def test_read_and_write(self, prob_size, dtype, hdf_test_base_tmp):
        ak_arr = make_ak_arrays(prob_size * pytest.nl, dtype)
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/hdf_test_correct"
            ak_arr.to_hdf(file_name)

            # test read_hdf with glob
            gen_arr = ak.read_hdf(f"{file_name}*").popitem()[1]
            assert (ak_arr == gen_arr).all()

            # test read_hdf with filenames
            gen_arr = ak.read_hdf(
                filenames=[f"{file_name}_LOCALE{i:04d}" for i in range(pytest.nl)]
            ).popitem()[1]
            assert (ak_arr == gen_arr).all()

            # verify generic read works
            gen_arr = ak.read(f"{file_name}*").popitem()[1]
            assert (ak_arr == gen_arr).all()

            # verify generic load works
            if dtype == "str":
                # we have to specify the dataset for strings since it differs from default of "array"
                gen_arr = ak.load(path_prefix=file_name, dataset="strings_array")["strings_array"]
            else:
                gen_arr = ak.load(path_prefix=file_name).popitem()[1]
            assert (ak_arr == gen_arr).all()

            # verify generic load works with file_format parameter
            if dtype == "str":
                # we have to specify the dataset for strings since it differs from default of "array"
                gen_arr = ak.load(path_prefix=file_name, dataset="strings_array", file_format="HDF5")[
                    "strings_array"
                ]
            else:
                gen_arr = ak.load(path_prefix=file_name, file_format="HDF5").popitem()[1]
            assert (ak_arr == gen_arr).all()

            # verify load_all works
            gen_arr = ak.load_all(path_prefix=file_name)
            if dtype == "str":
                # we have to specify the dataset for strings since it differs from default of "array"
                assert (ak_arr == gen_arr["strings_array"]).all()
            else:
                assert (ak_arr == gen_arr["array"]).all()

            # Test load with invalid file
            with pytest.raises(RuntimeError):
                ak.load(path_prefix=f"{hdf_test_base_tmp}/not-a-file")

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    def test_read_and_write_dset_provided(self, prob_size, dtype, hdf_test_base_tmp):
        ak_arr = make_ak_arrays(prob_size * pytest.nl, dtype)
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/hdf_test_correct"
            ak_arr.to_hdf(file_name, "my_dset")

            # test read_hdf with glob
            gen_arr = ak.read_hdf(f"{file_name}*", "my_dset")["my_dset"]
            assert (ak_arr == gen_arr).all()

            # test read_hdf with filenames
            gen_arr = ak.read_hdf(
                filenames=[f"{file_name}_LOCALE{i:04d}" for i in range(pytest.nl)],
                datasets="my_dset",
            )["my_dset"]
            assert (ak_arr == gen_arr).all()

            # verify generic read works
            gen_arr = ak.read(f"{file_name}*", "my_dset")["my_dset"]
            assert (ak_arr == gen_arr).all()

            # verify generic load works
            gen_arr = ak.load(path_prefix=file_name, dataset="my_dset")["my_dset"]
            assert (ak_arr == gen_arr).all()

            # verify generic load works with file_format parameter
            gen_arr = ak.load(path_prefix=file_name, dataset="my_dset", file_format="HDF5")["my_dset"]
            assert (ak_arr == gen_arr).all()

            # verify load_all works
            gen_arr = ak.load_all(path_prefix=file_name)
            assert (ak_arr == gen_arr["my_dset"]).all()

            # Test load with invalid file
            with pytest.raises(RuntimeError):
                ak.load(path_prefix=f"{hdf_test_base_tmp}/not-a-file", dataset="my_dset")

    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    def test_edge_case_read_write(self, dtype, hdf_test_base_tmp):
        np_edge_case = make_edge_case_arrays(dtype)
        ak_edge_case = ak.array(np_edge_case)
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            ak_edge_case.to_hdf(f"{tmp_dirname}/hdf_test_edge_case", "my-dset")
            hdf_arr = ak.read_hdf(f"{tmp_dirname}/hdf_test_edge_case*", "my-dset")["my-dset"]
            if dtype == "float64":
                assert np.allclose(np_edge_case, hdf_arr.to_ndarray(), equal_nan=True)
            else:
                assert (np_edge_case == hdf_arr.to_ndarray()).all()

    def test_read_and_write_with_dict(self, hdf_test_base_tmp):
        df_dict = make_multi_dtype_dict()
        # extend to include categoricals
        df_dict["cat"] = ak.Categorical(ak.array(["c", "b", "a", "b"]))
        df_dict["cat_from_codes"] = ak.Categorical.from_codes(
            codes=ak.array([2, 1, 0, 1]), categories=ak.array(["a", "b", "c"])
        )
        akdf = ak.DataFrame(df_dict)
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/multi_col_hdf"
            # use multi-column write to generate hdf file
            akdf.to_hdf(file_name)

            # test read_hdf with glob, no datasets specified
            rd_data = ak.read_hdf(f"{file_name}*")
            rd_df = ak.DataFrame(rd_data)
            # fix column ordering see issue #2611
            rd_df = rd_df[akdf.columns.values]
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            # test read_hdf with only one dataset specified (each tested)
            for col_name in akdf.columns.values:
                gen_arr = ak.read_hdf(f"{file_name}*", datasets=[col_name])[col_name]
                if akdf[col_name].dtype != ak.float64:
                    assert akdf[col_name].tolist() == gen_arr.tolist()
                else:
                    a = akdf[col_name].to_ndarray()
                    b = gen_arr.to_ndarray()
                    if isinstance(a[0], np.ndarray):
                        assert all(np.allclose(a1, b1, equal_nan=True) for a1, b1 in zip(a, b))
                    else:
                        assert np.allclose(a, b, equal_nan=True)

            # test read_hdf with half of columns names specified as datasets
            half_cols = akdf.columns.values[: len(akdf.columns.values) // 2]
            rd_data = ak.read_hdf(f"{file_name}*", datasets=half_cols)
            rd_df = ak.DataFrame(rd_data)
            pd.testing.assert_frame_equal(akdf[half_cols].to_pandas(), rd_df[half_cols].to_pandas())

            # test read_hdf with all columns names specified as datasets
            rd_data = ak.read_hdf(f"{file_name}*", datasets=akdf.columns.values)
            rd_df = ak.DataFrame(rd_data)
            # fix column ordering see issue #2611
            rd_df = rd_df[akdf.columns.values]
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            # test read_hdf with filenames
            rd_data = ak.read_hdf(filenames=[f"{file_name}_LOCALE{i:04d}" for i in range(pytest.nl)])
            rd_df = ak.DataFrame(rd_data)
            # fix column ordering see issue #2611
            rd_df = rd_df[akdf.columns.values]
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            # verify generic read works
            rd_data = ak.read(f"{file_name}*")
            rd_df = ak.DataFrame(rd_data)
            # fix column ordering see issue #2611
            rd_df = rd_df[akdf.columns.values]
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            for col_name in akdf.columns.values:
                # verify generic load works
                gen_arr = ak.load(path_prefix=file_name, dataset=col_name)[col_name]
                if akdf[col_name].dtype != ak.float64:
                    assert akdf[col_name].tolist() == gen_arr.tolist()
                else:
                    a = akdf[col_name].to_ndarray()
                    b = gen_arr.to_ndarray()
                    if isinstance(a[0], np.ndarray):
                        assert all(np.allclose(a1, b1, equal_nan=True) for a1, b1 in zip(a, b))
                    else:
                        assert np.allclose(a, b, equal_nan=True)

                # verify generic load works with file_format parameter
                gen_arr = ak.load(path_prefix=file_name, dataset=col_name, file_format="HDF5")[col_name]
                if akdf[col_name].dtype != ak.float64:
                    assert akdf[col_name].tolist() == gen_arr.tolist()
                else:
                    a = akdf[col_name].to_ndarray()
                    b = gen_arr.to_ndarray()
                    if isinstance(a[0], np.ndarray):
                        assert all(np.allclose(a1, b1, equal_nan=True) for a1, b1 in zip(a, b))
                    else:
                        assert np.allclose(a, b, equal_nan=True)

            # Test load with invalid file
            with pytest.raises(RuntimeError):
                ak.load(
                    path_prefix=f"{hdf_test_base_tmp}/not-a-file",
                    dataset=akdf.columns.values[0],
                )

            # verify load_all works
            rd_data = ak.load_all(path_prefix=file_name)
            rd_df = ak.DataFrame(rd_data)
            # fix column ordering see issue #2611
            rd_df = rd_df[akdf.columns.values]
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            # Test load_all with invalid file
            with pytest.raises(ValueError):
                ak.load_all(path_prefix=f"{hdf_test_base_tmp}/does-not-exist")

            # test get_datasets
            datasets = ak.get_datasets(f"{file_name}*")
            assert sorted(datasets) == sorted(akdf.columns.values)

            # test save with index true
            akdf.to_hdf(file_name, index=True)
            rd_data = ak.read_hdf(f"{file_name}*")
            rd_df = ak.DataFrame(rd_data)
            # fix column ordering see issue #2611
            rd_df = rd_df[akdf.columns.values]
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            # test get_datasets with index
            datasets = ak.get_datasets(f"{file_name}*")
            assert sorted(datasets) == ["Index"] + sorted(akdf.columns.values)

    def test_ls_hdf(self, hdf_test_base_tmp):
        df_dict = make_multi_dtype_dict()
        akdf = ak.DataFrame(df_dict)
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/test_ls_hdf"
            # use multi-column write to generate hdf file
            akdf.to_hdf(file_name)

            message = ak.ls(f"{file_name}_LOCALE0000")
            for col_name in akdf.columns.values:
                assert col_name in message

            with pytest.raises(RuntimeError):
                ak.ls(f"{tmp_dirname}/not-a-file_LOCALE0000")

    def test_ls_hdf_empty(self):
        # Test filename empty/whitespace-only condition
        with pytest.raises(ValueError):
            ak.ls("")

        with pytest.raises(ValueError):
            ak.ls("   ")

        with pytest.raises(ValueError):
            ak.ls(" \n\r\t  ")

    def test_read_hdf_with_error_and_warn(self, hdf_test_base_tmp):
        df_dict = make_multi_dtype_dict()
        akdf = ak.DataFrame(df_dict)
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/test_error_hdf"
            # use multi-column write to generate hdf file
            akdf.to_hdf(file_name)
            akdf.to_hdf(f"{file_name}_dupe")

            # Make sure we can read ok
            dataset = ak.read_hdf(
                filenames=[
                    f"{file_name}_LOCALE0000",
                    f"{file_name}_dupe_LOCALE0000",
                ]
            )
            assert dataset is not None

            # Change the name of the first file we try to raise an error due to file missing.
            with pytest.raises(RuntimeError):
                ak.read_hdf(
                    filenames=[
                        f"{file_name}_MISSING_LOCALE0000",
                        f"{file_name}_dupe_LOCALE0000",
                    ]
                )

            # Run the same test with missing file, but this time with the warning flag for read_all
            with pytest.warns(
                RuntimeWarning,
                match=r"There were .* errors reading files on the server.*",
            ):
                dataset = ak.read_hdf(
                    filenames=[
                        f"{file_name}_MISSING_LOCALE0000",
                        f"{file_name}_dupe_LOCALE0000",
                    ],
                    strict_types=False,
                    allow_errors=True,
                )
                assert dataset is not None

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_save_strings_dataset(self, prob_size, hdf_test_base_tmp):
        reg_strings = make_ak_arrays(prob_size, "str")
        # hard coded at 26 because we don't need to test long strings at large scale
        # passing data from python to chpl this way can really slow down as size increases
        long_strings = ak.array(
            [f"testing a longer string{num} to be written, loaded and appended" for num in range(26)]
        )

        for strings_array in [reg_strings, long_strings]:
            with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
                file_name = f"{tmp_dirname}/test_strings_hdf"
                strings_array.to_hdf(file_name)
                r_strings_array = ak.read_hdf(f"{file_name}*").popitem()[1]
                assert (strings_array == r_strings_array).all()

                # Read a part of a saved Strings dataset from one hdf5 file
                r_strings_subset = ak.read_hdf(filenames=f"{file_name}_LOCALE0000").popitem()[1]
                assert isinstance(r_strings_subset, ak.Strings)
                assert (strings_array[: r_strings_subset.size] == r_strings_subset).all()

                # Repeat the test using the calc_string_offsets=True option to
                # have server calculate offsets array
                r_strings_subset = ak.read_hdf(
                    filenames=f"{file_name}_LOCALE0000", calc_string_offsets=True
                ).popitem()[1]
                assert isinstance(r_strings_subset, ak.Strings)
                assert (strings_array[: r_strings_subset.size] == r_strings_subset).all()

                # test append
                strings_array.to_hdf(file_name, dataset="strings-dupe", mode="append")
                r_strings = ak.read_hdf(f"{file_name}*", datasets="strings_array")["strings_array"]
                r_strings_dupe = ak.read_hdf(f"{file_name}*", datasets="strings-dupe")["strings-dupe"]
                assert (r_strings == r_strings_dupe).all()

    def testStringsWithoutOffsets(self, hdf_test_base_tmp):
        """
        This tests both saving & reading a strings array without saving and reading the offsets to HDF5.
        Instead the offsets array will be derived from the values/bytes area by looking for null-byte
        terminator strings
        """
        strings_array = ak.array(["testing string{}".format(num) for num in list(range(0, 25))])
        strings_array.to_hdf(
            "{}/strings-test".format(hdf_test_base_tmp),
            dataset="strings",
            save_offsets=False,
        )
        r_strings_array = ak.load(
            "{}/strings-test".format(hdf_test_base_tmp),
            dataset="strings",
            calc_string_offsets=True,
        )["strings"]
        strings = strings_array.to_ndarray()
        strings.sort()
        r_strings = r_strings_array.to_ndarray()
        r_strings.sort()
        assert strings.tolist() == r_strings.tolist()

    def testSaveLongStringsDataset(self, hdf_test_base_tmp):
        # Create, save, and load Strings dataset
        strings = ak.array(
            [
                "testing a longer string{} to be written, loaded and appended".format(num)
                for num in list(range(0, 26))
            ]
        )
        strings.to_hdf("{}/strings-test".format(hdf_test_base_tmp), dataset="strings")

        n_strings = strings.to_ndarray()
        n_strings.sort()
        r_strings = ak.load("{}/strings-test".format(hdf_test_base_tmp), dataset="strings")[
            "strings"
        ].to_ndarray()
        r_strings.sort()

        assert n_strings.tolist() == r_strings.tolist()

    def testSaveMixedStringsDataset(self, hdf_test_base_tmp):
        strings_array = ak.array(["string {}".format(num) for num in list(range(0, 25))])
        m_floats = ak.array([x / 10.0 for x in range(0, 10)])
        m_ints = ak.array(list(range(0, 10)))
        ak.to_hdf(
            {"m_strings": strings_array, "m_floats": m_floats, "m_ints": m_ints},
            "{}/multi-type-test".format(hdf_test_base_tmp),
        )
        r_mixed = ak.load_all("{}/multi-type-test".format(hdf_test_base_tmp))

        assert (
            np.sort(strings_array.to_ndarray()).tolist()
            == np.sort(r_mixed["m_strings"].to_ndarray()).tolist()
        )

        assert r_mixed["m_floats"] is not None
        assert r_mixed["m_ints"] is not None

        r_floats = ak.sort(
            ak.load("{}/multi-type-test".format(hdf_test_base_tmp), dataset="m_floats")["m_floats"]
        )
        assert m_floats.tolist() == r_floats.tolist()

        r_ints = ak.sort(
            ak.load("{}/multi-type-test".format(hdf_test_base_tmp), dataset="m_ints")["m_ints"]
        )
        assert m_ints.tolist() == r_ints.tolist()

        strings = strings_array.to_ndarray()
        strings.sort()
        r_strings = ak.load("{}/multi-type-test".format(hdf_test_base_tmp), dataset="m_strings")[
            "m_strings"
        ].to_ndarray()
        r_strings.sort()

        assert strings.tolist() == r_strings.tolist()

    def testAppendStringsDataset(self, hdf_test_base_tmp):
        strings_array = ak.array(["string {}".format(num) for num in list(range(0, 25))])
        strings_array.to_hdf("{}/append-strings-test".format(hdf_test_base_tmp), dataset="strings")
        strings_array.to_hdf(
            "{}/append-strings-test".format(hdf_test_base_tmp),
            dataset="strings-dupe",
            mode="append",
        )

        r_strings = ak.load("{}/append-strings-test".format(hdf_test_base_tmp), dataset="strings")[
            "strings"
        ]
        r_strings_dupe = ak.load(
            "{}/append-strings-test".format(hdf_test_base_tmp), dataset="strings-dupe"
        )["strings-dupe"]
        assert r_strings.tolist() == r_strings_dupe.tolist()

    def testAppendMixedStringsDataset(self, hdf_test_base_tmp):
        strings_array = ak.array(["string {}".format(num) for num in list(range(0, 25))])
        strings_array.to_hdf("{}/append-multi-type-test".format(hdf_test_base_tmp), dataset="m_strings")
        m_floats = ak.array([x / 10.0 for x in range(0, 10)])
        m_ints = ak.array(list(range(0, 10)))
        ak.to_hdf(
            {"m_floats": m_floats, "m_ints": m_ints},
            "{}/append-multi-type-test".format(hdf_test_base_tmp),
            mode="append",
        )
        r_mixed = ak.load_all("{}/append-multi-type-test".format(hdf_test_base_tmp))

        assert r_mixed["m_floats"] is not None
        assert r_mixed["m_ints"] is not None

        r_floats = ak.sort(
            ak.load(
                "{}/append-multi-type-test".format(hdf_test_base_tmp),
                dataset="m_floats",
            )["m_floats"]
        )
        r_ints = ak.sort(
            ak.load("{}/append-multi-type-test".format(hdf_test_base_tmp), dataset="m_ints")["m_ints"]
        )
        assert m_floats.tolist() == r_floats.tolist()
        assert m_ints.tolist() == r_ints.tolist()

        strings = strings_array.to_ndarray()
        strings.sort()
        r_strings = r_mixed["m_strings"].to_ndarray()
        r_strings.sort()

        assert strings.tolist() == r_strings.tolist()

    def test_save_multi_type_dict_dataset(self, hdf_test_base_tmp):
        df_dict = make_multi_dtype_dict()
        # extend to include categoricals
        df_dict["cat"] = ak.Categorical(ak.array(["c", "b", "a", "b"]))
        df_dict["cat_from_codes"] = ak.Categorical.from_codes(
            codes=ak.array([2, 1, 0, 1]), categories=ak.array(["a", "b", "c"])
        )
        keys = list(df_dict.keys())
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/multi_type_dict_test"
            # use multi-column write to generate hdf file
            ak.to_hdf(df_dict, file_name)
            r_mixed = ak.read_hdf(f"{file_name}*")

            for col_name in keys:
                # verify load by dataset and returned mixed dict at col_name
                loaded = ak.load(file_name, dataset=col_name)[col_name]
                for arr in [loaded, r_mixed[col_name]]:
                    if df_dict[col_name].dtype != ak.float64:
                        assert df_dict[col_name].tolist() == arr.tolist()
                    else:
                        a = df_dict[col_name].to_ndarray()
                        b = arr.to_ndarray()
                        if isinstance(a[0], np.ndarray):
                            assert all(np.allclose(a1, b1, equal_nan=True) for a1, b1 in zip(a, b))
                        else:
                            assert np.allclose(a, b, equal_nan=True)

        # test append for multi type dict
        single_arr = df_dict[keys[0]]
        rest_dict = {k: df_dict[k] for k in keys[1:]}
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/multi_type_dict_test"
            single_arr.to_hdf(file_name, dataset=keys[0])

            ak.to_hdf(rest_dict, file_name, mode="append")
            r_mixed = ak.read_hdf(f"{file_name}*")

            for col_name in keys:
                # verify load by dataset and returned mixed dict at col_name
                loaded = ak.load(file_name, dataset=col_name)[col_name]
                for arr in [loaded, r_mixed[col_name]]:
                    if df_dict[col_name].dtype != ak.float64:
                        assert df_dict[col_name].tolist() == arr.tolist()
                    else:
                        a = df_dict[col_name].to_ndarray()
                        b = arr.to_ndarray()
                        if isinstance(a[0], np.ndarray):
                            assert all(np.allclose(a1, b1, equal_nan=True) for a1, b1 in zip(a, b))
                        else:
                            assert np.allclose(a, b, equal_nan=True)

    def test_strict_types(self, hdf_test_base_tmp):
        N = 100
        int_types = [np.uint32, np.int64, np.uint16, np.int16]
        float_types = [np.float32, np.float64, np.float32, np.float64]
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            prefix = f"{tmp_dirname}/strict-type-test"
            for i, (it, ft) in enumerate(zip(int_types, float_types)):
                with h5py.File("{}-{}".format(prefix, i), "w") as f:
                    idata = np.arange(i * N, (i + 1) * N, dtype=it)
                    id = f.create_dataset("integers", data=idata)
                    id.attrs["ObjType"] = 1
                    fdata = np.arange(i * N, (i + 1) * N, dtype=ft)
                    fd = f.create_dataset("floats", data=fdata)
                    fd.attrs["ObjType"] = 1
            with pytest.raises(RuntimeError):
                ak.read_hdf(f"{prefix}*")

            a = ak.read_hdf(f"{prefix}*", strict_types=False)
            assert a["integers"].tolist() == np.arange(len(int_types) * N).tolist()
            assert np.allclose(
                a["floats"].to_ndarray(),
                np.arange(len(float_types) * N, dtype=np.float64),
            )

    def test_small_arrays(self, hdf_test_base_tmp):
        for arr in [ak.array([1]), ak.array(["ab", "cd"]), ak.array(["123456789"])]:
            with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
                arr.to_hdf(f"{tmp_dirname}/small_numeric")
                ret_arr = ak.read_hdf(f"{tmp_dirname}/small_numeric*").popitem()[1]
                assert (arr == ret_arr).all()

    def test_uint64_to_from_HDF5(self, hdf_test_base_tmp):
        """Test our ability to read/write uint64 to HDF5."""
        npa1 = np.array(
            [18446744073709551500, 18446744073709551501, 18446744073709551502],
            dtype=np.uint64,
        )
        pda1 = ak.array(npa1)
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            pda1.to_hdf(f"{tmp_dirname}/small_numeric", dataset="pda1")
            # Now load it back in
            pda2 = ak.load(f"{tmp_dirname}/small_numeric", dataset="pda1")["pda1"]
            assert str(pda1) == str(pda2)
            assert 18446744073709551500 == pda2[0]
            assert pda2.tolist() == npa1.tolist()

    def test_uint64_to_from_array(self, hdf_test_base_tmp):
        """
        Test conversion to and from numpy array / pdarray.

        Uses unsigned 64bit integer (uint64).
        """
        npa1 = np.array(
            [18446744073709551500, 18446744073709551501, 18446744073709551502],
            dtype=np.uint64,
        )
        pda1 = ak.array(npa1)
        assert 18446744073709551500 == pda1[0]
        assert pda1.tolist() == npa1.tolist()

    def test_bigint(self, hdf_test_base_tmp):
        df_dict = {
            "pdarray": ak.arange(2**200, 2**200 + 3, max_bits=201),
            "groupby": ak.GroupBy(ak.arange(2**200, 2**200 + 5)),
            "segarray": ak.SegArray(ak.arange(0, 10, 2), ak.arange(2**200, 2**200 + 10, max_bits=212)),
        }
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/bigint_test"
            ak.to_hdf(df_dict, file_name)
            ret_dict = ak.read_hdf(f"{tmp_dirname}/bigint_test*")

            pda_loaded = ak.read_hdf(f"{tmp_dirname}/bigint_test*", datasets="pdarray")["pdarray"]
            a = df_dict["pdarray"]
            for rd_a in [ret_dict["pdarray"], pda_loaded]:
                assert isinstance(rd_a, ak.pdarray)
                assert a.tolist() == rd_a.tolist()
                assert a.max_bits == rd_a.max_bits

            g_loaded = ak.read_hdf(f"{tmp_dirname}/bigint_test*", datasets="groupby")["groupby"]
            g = df_dict["groupby"]
            for rd_g in [ret_dict["groupby"], g_loaded]:
                assert isinstance(rd_g, ak.GroupBy)
                assert g.keys.tolist() == rd_g.keys.tolist()
                assert g.unique_keys.tolist() == rd_g.unique_keys.tolist()
                assert g.permutation.tolist() == rd_g.permutation.tolist()
                assert g.segments.tolist() == rd_g.segments.tolist()

            sa_loaded = ak.read_hdf(f"{tmp_dirname}/bigint_test*", datasets="segarray")["segarray"]
            sa = df_dict["segarray"]
            for rd_sa in [ret_dict["segarray"], sa_loaded]:
                assert isinstance(rd_sa, ak.SegArray)
                assert sa.values.tolist() == rd_sa.values.tolist()
                assert sa.segments.tolist() == rd_sa.segments.tolist()

    def test_unsanitized_dataset_names(self, hdf_test_base_tmp):
        # Test when quotes are part of the dataset name
        my_arrays = {'foo"0"': ak.arange(100), 'bar"': ak.arange(100)}
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            ak.to_hdf(my_arrays, f"{tmp_dirname}/bad_dataset_names")
            ak.read_hdf(f"{tmp_dirname}/bad_dataset_names*")

    def test_hdf_groupby(self, hdf_test_base_tmp):
        # test for categorical and multiple keys
        string = ak.array(["a", "b", "a", "b", "c"])
        cat = ak.Categorical(string)
        cat_from_codes = ak.Categorical.from_codes(
            codes=ak.array([0, 1, 0, 1, 2]), categories=ak.array(["a", "b", "c"])
        )
        pda = ak.array([0, 1, 2, 0, 2])

        pda_grouping = ak.GroupBy(pda)
        str_grouping = ak.GroupBy(string)
        cat_grouping = ak.GroupBy([cat, cat_from_codes])

        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            for g in [pda_grouping, str_grouping, cat_grouping]:
                g.to_hdf(f"{tmp_dirname}/groupby_test")
                g_load = ak.read(f"{tmp_dirname}/groupby_test*").popitem()[1]
                assert len(g_load.keys) == len(g.keys)
                assert g_load.permutation.tolist() == g.permutation.tolist()
                assert g_load.segments.tolist() == g.segments.tolist()
                assert g_load._uki.tolist() == g._uki.tolist()
                if isinstance(g.keys[0], ak.Categorical):
                    for k, kload in zip(g.keys, g_load.keys):
                        assert k.tolist() == kload.tolist()
                else:
                    assert g_load.keys.tolist() == g.keys.tolist()

    def test_hdf_categorical(self, hdf_test_base_tmp):
        cat = ak.Categorical(ak.array(["a", "b", "a", "b", "c"]))
        cat_from_codes = ak.Categorical.from_codes(
            codes=ak.array([0, 1, 0, 1, 2]), categories=ak.array(["a", "b", "c"])
        )
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            for c in cat, cat_from_codes:
                c.to_hdf(f"{tmp_dirname}/categorical_test")
                c_load = ak.read(f"{tmp_dirname}/categorical_test*").popitem()[1]

                assert c_load.categories.tolist() == (["a", "b", "c", "N/A"])
                if c.segments is not None:
                    assert c.segments.tolist() == c_load.segments.tolist()
                    assert c.permutation.tolist() == c_load.permutation.tolist()

    def test_segarray_hdf(self, hdf_test_base_tmp):
        a = [0, 1, 2, 3]
        b = [4, 0, 5, 6, 0, 7, 8, 0]
        c = [9, 0, 0]

        # int64 test
        flat = a + b + c
        segments = ak.array([0, len(a), len(a) + len(b)])
        dtype = ak.numpy.dtypes.int64
        akflat = ak.array(flat, dtype)
        segarr = ak.SegArray(segments, akflat)

        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            segarr.to_hdf(f"{tmp_dirname}/segarray_int")
            # Now load it back in
            seg2 = ak.load(f"{tmp_dirname}/segarray_int", dataset="segarray")["segarray"]
            assert segarr.segments.tolist() == seg2.segments.tolist()
            assert segarr.values.tolist() == seg2.values.tolist()

        # uint64 test
        dtype = ak.numpy.dtypes.uint64
        akflat = ak.array(flat, dtype)
        segarr = ak.SegArray(segments, akflat)

        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            segarr.to_hdf(f"{tmp_dirname}/segarray_uint")
            # Now load it back in
            seg2 = ak.load(f"{tmp_dirname}/segarray_uint", dataset="segarray")["segarray"]
            assert segarr.segments.tolist() == seg2.segments.tolist()
            assert segarr.values.tolist() == seg2.values.tolist()

        # float64 test
        dtype = ak.numpy.dtypes.float64
        akflat = ak.array(flat, dtype)
        segarr = ak.SegArray(segments, akflat)

        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            segarr.to_hdf(f"{tmp_dirname}/segarray_float")
            # Now load it back in
            seg2 = ak.load(f"{tmp_dirname}/segarray_float", dataset="segarray")["segarray"]
            assert segarr.segments.tolist() == seg2.segments.tolist()
            assert segarr.values.tolist() == seg2.values.tolist()

        # bool test
        dtype = ak.numpy.dtypes.bool_
        akflat = ak.array(flat, dtype)
        segarr = ak.SegArray(segments, akflat)

        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            segarr.to_hdf(f"{tmp_dirname}/segarray_bool")
            # Now load it back in
            seg2 = ak.load(f"{tmp_dirname}/segarray_bool", dataset="segarray")["segarray"]
            assert segarr.segments.tolist() == seg2.segments.tolist()
            assert segarr.values.tolist() == seg2.values.tolist()

    def test_dataframe_segarr(self, hdf_test_base_tmp):
        a = [0, 1, 2, 3]
        b = [4, 0, 5, 6, 0, 7, 8, 0]
        c = [9, 0, 0]

        # int64 test
        flat = a + b + c
        segments = ak.array([0, len(a), len(a) + len(b)])
        dtype = ak.numpy.dtypes.int64
        akflat = ak.array(flat, dtype)
        segarr = ak.SegArray(segments, akflat)

        s = ak.array(["abc", "def", "ghi"])
        df = ak.DataFrame([segarr, s])
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            df.to_hdf(f"{tmp_dirname}/dataframe_segarr")
            df_load = ak.DataFrame.load(f"{tmp_dirname}/dataframe_segarr")
            assert df.to_pandas().equals(df_load.to_pandas())

    def test_segarray_str_hdf5(self, hdf_test_base_tmp):
        words = ak.array(["one,two,three", "uno,dos,tres"])
        strs, segs = words.regex_split(",", return_segments=True)

        x = ak.SegArray(segs, strs)
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            x.to_hdf(f"{tmp_dirname}/test_file")
            rd = ak.read_hdf(f"{tmp_dirname}/test_file*").popitem()[1]
            assert isinstance(rd, ak.SegArray)
            assert x.segments.tolist() == rd.segments.tolist()
            assert x.values.tolist() == rd.values.tolist()

    def test_hdf_overwrite_pdarray(self, hdf_test_base_tmp):
        # test repack with a single object
        a = ak.arange(1000)
        b = ak.randint(0, 100, 1000)
        c = ak.arange(15)
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/pda_test"
            for repack in [True, False]:
                a.to_hdf(file_name)
                b.to_hdf(file_name, dataset="array_2", mode="append")
                f_list = glob.glob(f"{file_name}*")
                orig_size = sum(os.path.getsize(f) for f in f_list)
                # hdf5 only releases memory if overwriting last dset so overwrite first
                c.update_hdf(file_name, dataset="array", repack=repack)

                new_size = sum(os.path.getsize(f) for f in f_list)

                # ensure that the column was actually overwritten
                # test that repack on/off the file gets smaller/larger respectively
                assert new_size < orig_size if repack else new_size >= orig_size
                data = ak.read_hdf(f"{file_name}*")
                assert data["array"].tolist() == c.tolist()

        # test overwrites with different types
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/pda_test"
            a.to_hdf(file_name)
            for size, dtype in [(15, ak.uint64), (150, ak.float64), (1000, ak.bool_)]:
                b = ak.arange(size, dtype=dtype)
                b.update_hdf(file_name)
                data = ak.read_hdf(f"{file_name}*").popitem()[1]
                assert data.tolist() == b.tolist()

    def test_hdf_overwrite_strings(self, hdf_test_base_tmp):
        # test repack with a single object
        a = ak.random_strings_uniform(0, 16, 1000)
        b = ak.random_strings_uniform(0, 16, 1000)
        c = ak.random_strings_uniform(0, 16, 10)
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/str_test"
            for repack in [True, False]:
                a.to_hdf(file_name, dataset="test_str")
                b.to_hdf(file_name, mode="append")
                f_list = glob.glob(f"{file_name}*")
                orig_size = sum(os.path.getsize(f) for f in f_list)
                # hdf5 only releases memory if overwriting last dset so overwrite first
                c.update_hdf(file_name, dataset="test_str", repack=repack)

                new_size = sum(os.path.getsize(f) for f in f_list)

                # ensure that the column was actually overwritten
                # test that repack on/off the file gets smaller/larger respectively
                assert new_size < orig_size if repack else new_size >= orig_size
                data = ak.read_hdf(f"{file_name}*")
                assert data["test_str"].tolist() == c.tolist()

    def test_overwrite_categorical(self, hdf_test_base_tmp):
        a = ak.Categorical(ak.array([f"cat_{i % 3}" for i in range(100)]))
        b = ak.Categorical(ak.array([f"cat_{i % 4}" for i in range(100)]))
        c = ak.Categorical(ak.array([f"cat_{i % 5}" for i in range(10)]))
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/cat_test"
            for repack in [True, False]:
                a.to_hdf(file_name, dataset="test_cat")
                b.to_hdf(file_name, mode="append")
                f_list = glob.glob(f"{file_name}*")
                orig_size = sum(os.path.getsize(f) for f in f_list)
                # hdf5 only releases memory if overwriting last dset so overwrite first
                c.update_hdf(file_name, dataset="test_cat", repack=repack)

                new_size = sum(os.path.getsize(f) for f in f_list)

                # ensure that the column was actually overwritten
                # test that repack on/off the file gets smaller/larger respectively
                assert new_size < orig_size if repack else new_size >= orig_size
                data = ak.read_hdf(f"{file_name}*")
                assert (data["test_cat"] == c).all()

            dset_name = "categorical_array"  # name of categorical array
            dset_name2 = "to_replace"
            dset_name3 = "cat_array2"
            a.to_hdf(file_name, dataset=dset_name)
            b.to_hdf(file_name, dataset=dset_name2, mode="append")
            c.to_hdf(file_name, dataset=dset_name3, mode="append")

            a.update_hdf(file_name, dataset=dset_name2)
            data = ak.read_hdf(f"{file_name}*")
            assert all(name in data for name in (dset_name, dset_name2, dset_name3))
            d = data[dset_name2]
            for attr in "categories", "codes", "permutation", "segments", "_akNAcode":
                assert getattr(d, attr).tolist() == getattr(a, attr).tolist()

    def test_hdf_overwrite_dataframe(self, hdf_test_base_tmp):
        df = ak.DataFrame(
            {
                "a": ak.arange(1000),
                "b": ak.random_strings_uniform(0, 16, 1000),
                "c": ak.arange(1000, dtype=bool),
                "d": ak.randint(0, 50, 1000),
            }
        )
        odf = ak.DataFrame(
            {
                "b": ak.randint(0, 25, 50),
                "c": ak.arange(50, dtype=bool),
            }
        )
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/df_test"
            for repack in [True, False]:
                df.to_hdf(file_name)
                f_list = glob.glob(f"{file_name}*")
                orig_size = sum(os.path.getsize(f) for f in f_list)
                # hdf5 only releases memory if overwriting last dset so overwrite first
                odf.update_hdf(file_name, repack=repack)

                new_size = sum(os.path.getsize(f) for f in f_list)
                # ensure that the column was actually overwritten
                # test that repack on/off the file gets smaller/larger respectively
                assert new_size <= orig_size if repack else new_size >= orig_size
                data = ak.read_hdf(f"{file_name}*")
                odf_keys = list(odf.keys())
                for key in df.keys():
                    assert (data[key] == (odf[key] if key in odf_keys else df[key])).all()

    def test_overwrite_segarray(self, hdf_test_base_tmp):
        sa1 = ak.SegArray(ak.arange(0, 1000, 5), ak.arange(1000))
        sa2 = ak.SegArray(ak.arange(0, 100, 5), ak.arange(100))
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/segarray_test"
            for repack in [True, False]:
                sa1.to_hdf(file_name)
                sa1.to_hdf(file_name, dataset="seg2", mode="append")
                f_list = glob.glob(f"{file_name}*")
                orig_size = sum(os.path.getsize(f) for f in f_list)

                sa2.update_hdf(file_name, repack=repack)

                new_size = sum(os.path.getsize(f) for f in f_list)
                # ensure that the column was actually overwritten
                # test that repack on/off the file gets smaller/larger respectively
                assert new_size <= orig_size if repack else new_size >= orig_size
                data = ak.read_hdf(f"{file_name}*")
                assert (data["segarray"].values == sa2.values).all()
                assert (data["segarray"].segments == sa2.segments).all()

    def test_overwrite_single_dset(self, hdf_test_base_tmp):
        # we need to test that both repack=False and repack=True generate the same file size here
        a = ak.arange(1000)
        b = ak.arange(15)
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            a.to_hdf(f"{tmp_dirname}/test_file")
            b.update_hdf(f"{tmp_dirname}/test_file")
            f_list = glob.glob(f"{tmp_dirname}/test_file*")
            f1_size = sum(os.path.getsize(f) for f in f_list)

            a.to_hdf(f"{tmp_dirname}/test_file_2")
            b.update_hdf(f"{tmp_dirname}/test_file_2", repack=False)
            f_list = glob.glob(f"{tmp_dirname}/test_file_2_*")
            f2_size = sum(os.path.getsize(f) for f in f_list)

            assert f1_size == f2_size

    def test_overwrite_dataframe(self, hdf_test_base_tmp):
        df = ak.DataFrame(
            {
                "a": ak.arange(1000),
                "b": ak.random_strings_uniform(0, 16, 1000),
                "c": ak.arange(1000, dtype=bool),
                "d": ak.randint(0, 50, 1000),
            }
        )
        replace = {
            "b": ak.randint(0, 25, 50),
            "c": ak.arange(50, dtype=bool),
        }
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            df.to_hdf(f"{tmp_dirname}/overwrite_test")
            f_list = glob.glob(f"{tmp_dirname}/overwrite_test_*")
            orig_size = sum(os.path.getsize(f) for f in f_list)
            # hdf5 only releases memory if overwritting last dset so overwrite first
            ak.update_hdf(replace, f"{tmp_dirname}/overwrite_test")

            new_size = sum(os.path.getsize(f) for f in f_list)
            # ensure that the column was actually overwritten
            assert new_size < orig_size
            data = ak.read_hdf(f"{tmp_dirname}/overwrite_test_*")
            assert data["b"].tolist() == replace["b"].tolist()
            assert data["c"].tolist() == replace["c"].tolist()

        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            df.to_hdf(f"{tmp_dirname}/overwrite_test")
            f_list = glob.glob(f"{tmp_dirname}/overwrite_test_*")
            orig_size = sum(os.path.getsize(f) for f in f_list)
            # hdf5 only releases memory if overwritting last dset so overwrite first
            ak.update_hdf(replace, f"{tmp_dirname}/overwrite_test", repack=False)

            new_size = sum(os.path.getsize(f) for f in f_list)
            # ensure that the column was actually overwritten
            assert new_size >= orig_size
            data = ak.read_hdf(f"{tmp_dirname}/overwrite_test_*")
            assert data["b"].tolist() == replace["b"].tolist()
            assert data["c"].tolist() == replace["c"].tolist()

    def test_snapshot(self, hdf_test_base_tmp):
        df = ak.DataFrame(make_multi_dtype_dict())
        df_str_idx = df.copy()
        df_str_idx._set_index([f"A{i}" for i in range(len(df))])
        col_order = df.columns.values
        df_ref = df.to_pandas()
        df_str_idx_ref = df_str_idx.to_pandas(retain_index=True)
        a = ak.randint(0, 10, 100)
        s = ak.random_strings_uniform(0, 5, 50)
        c = ak.Categorical(s)
        g = ak.GroupBy(a)
        ref_data = {"a": a, "s": s, "c": c, "g": g}

        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            ak.snapshot(f"{tmp_dirname}/snapshot_test")
            for v in [df, df_str_idx, a, s, c, g]:
                # delete variables and verify no longer in the namespace
                del v
                with pytest.raises(NameError):
                    assert not v  # noqa: F821

            # restore the variables
            data = ak.restore(f"{tmp_dirname}/snapshot_test")
            for vn in ["df", "df_str_idx", "a", "s", "c", "g"]:
                # ensure all variable names returned
                assert vn in data.keys()

            # validate that restored variables are correct
            pd.testing.assert_frame_equal(
                df_ref[col_order], data["df"].to_pandas(retain_index=True)[col_order]
            )
            pd.testing.assert_frame_equal(
                df_str_idx_ref[col_order],
                data["df_str_idx"].to_pandas(retain_index=True)[col_order],
            )
            for key in ref_data.keys():
                if isinstance(data[key], ak.GroupBy):
                    assert (ref_data[key].permutation == data[key].permutation).all()
                    assert (ref_data[key].keys == data[key].keys).all()
                    assert (ref_data[key].segments == data[key].segments).all()
                else:
                    assert (ref_data[key] == data[key]).all()

    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_index_save_and_load(self, dtype, size, hdf_test_base_tmp):
        idx = ak.Index(make_ak_arrays(size, dtype))
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            idx.to_hdf(f"{tmp_dirname}/idx_test")
            rd_idx = ak.read_hdf(f"{tmp_dirname}/idx_test*").popitem()[1]

            assert isinstance(rd_idx, ak.Index)
            assert type(rd_idx.values) is type(idx.values)
            assert idx.tolist() == rd_idx.tolist()

        if dtype == ak.str_:
            # if strings we need to also test Categorical
            idx = ak.Index(ak.Categorical(make_ak_arrays(size, dtype)))
            with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
                idx.to_hdf(f"{tmp_dirname}/idx_test")
                rd_idx = ak.read_hdf(f"{tmp_dirname}/idx_test*").popitem()[1]

                assert isinstance(rd_idx, ak.Index)
                assert type(rd_idx.values) is type(idx.values)
                assert idx.tolist() == rd_idx.tolist()

    @pytest.mark.parametrize("dtype1", NUMERIC_AND_STR_TYPES)
    @pytest.mark.parametrize("dtype2", NUMERIC_AND_STR_TYPES)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_multi_index(self, dtype1, dtype2, size, hdf_test_base_tmp):
        t1 = make_ak_arrays(size, dtype1)
        t2 = make_ak_arrays(size, dtype2)
        idx = ak.Index.factory([t1, t2])
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            idx.to_hdf(f"{tmp_dirname}/idx_test")
            rd_idx = ak.read_hdf(f"{tmp_dirname}/idx_test*").popitem()[1]

            assert isinstance(rd_idx, ak.MultiIndex)
            assert idx.tolist() == rd_idx.tolist()

        # handle categorical cases as well
        if ak.str_ in [dtype1, dtype2]:
            if dtype1 == ak.str_:
                t1 = ak.Categorical(t1)
            if dtype2 == ak.str_:
                t2 = ak.Categorical(t2)
            idx = ak.Index.factory([t1, t2])
            with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
                idx.to_hdf(f"{tmp_dirname}/idx_test")
                rd_idx = ak.read_hdf(f"{tmp_dirname}/idx_test*").popitem()[1]

                assert isinstance(rd_idx, ak.MultiIndex)
                assert idx.tolist() == rd_idx.tolist()

    def test_hdf_overwrite_index(self, hdf_test_base_tmp):
        # test repack with a single object
        a = ak.Index(ak.arange(1000))
        b = ak.Index(ak.randint(0, 100, 1000))
        c = ak.Index(ak.arange(15))
        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/idx_test"
            for repack in [True, False]:
                a.to_hdf(file_name, dataset="index")
                b.to_hdf(file_name, dataset="index_2", mode="append")
                f_list = glob.glob(f"{file_name}*")
                orig_size = sum(os.path.getsize(f) for f in f_list)
                # hdf5 only releases memory if overwriting last dset so overwrite first
                c.update_hdf(file_name, dataset="index", repack=repack)

                new_size = sum(os.path.getsize(f) for f in f_list)

                # ensure that the column was actually overwritten
                # test that repack on/off the file gets smaller/larger respectively
                assert new_size < orig_size if repack else new_size >= orig_size
                data = ak.read_hdf(f"{file_name}*")
                assert isinstance(data["index"], ak.Index)
                assert data["index"].tolist() == c.tolist()

    def test_special_objtype(self, hdf_test_base_tmp):
        """
        This test is simply to ensure that the dtype is persisted through the io
        operation. It ultimately uses the process of pdarray, but need to ensure
        correct Arkouda Object Type is returned
        """
        ip = ak.IPv4(ak.arange(10))
        dt = ak.Datetime(ak.arange(10))
        td = ak.Timedelta(ak.arange(10))
        df = ak.DataFrame({"ip": ip, "datetime": dt, "timedelta": td})

        with tempfile.TemporaryDirectory(dir=hdf_test_base_tmp) as tmp_dirname:
            ip.to_hdf(f"{tmp_dirname}/ip_test")
            rd_ip = ak.read_hdf(f"{tmp_dirname}/ip_test*").popitem()[1]
            assert isinstance(rd_ip, ak.IPv4)
            assert ip.tolist() == rd_ip.tolist()

            dt.to_hdf(f"{tmp_dirname}/dt_test")
            rd_dt = ak.read_hdf(f"{tmp_dirname}/dt_test*").popitem()[1]
            assert isinstance(rd_dt, ak.Datetime)
            assert dt.tolist() == rd_dt.tolist()

            td.to_hdf(f"{tmp_dirname}/td_test")
            rd_td = ak.read_hdf(f"{tmp_dirname}/td_test*").popitem()[1]
            assert isinstance(rd_td, ak.Timedelta)
            assert td.tolist() == rd_td.tolist()

            df.to_hdf(f"{tmp_dirname}/df_test")
            rd_df = ak.read_hdf(f"{tmp_dirname}/df_test*")

            assert isinstance(rd_df["ip"], ak.IPv4)
            assert isinstance(rd_df["datetime"], ak.Datetime)
            assert isinstance(rd_df["timedelta"], ak.Timedelta)
            assert df["ip"].tolist() == rd_df["ip"].tolist()
            assert df["datetime"].tolist() == rd_df["datetime"].tolist()
            assert df["timedelta"].tolist() == rd_df["timedelta"].tolist()


@pytest.mark.skip_if_max_rank_greater_than(1)
class TestCSV:
    def test_csv_read_write(self, csv_test_base_tmp):
        # first test that can read csv with no header not written by Arkouda
        cols = ["ColA", "ColB", "ColC"]
        a = ["ABC", "DEF"]
        b = ["123", "345"]
        c = ["3.14", "5.56"]
        with tempfile.TemporaryDirectory(dir=csv_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/non_ak.csv"
            with open(file_name, "w") as f:
                f.write(",".join(cols) + "\n")
                f.write(f"{a[0]},{b[0]},{c[0]}\n")
                f.write(f"{a[1]},{b[1]},{c[1]}\n")

            data = ak.read_csv(file_name)
            assert list(data.keys()) == cols
            assert data["ColA"].tolist() == a
            assert data["ColB"].tolist() == b
            assert data["ColC"].tolist() == c

            data = ak.read_csv(file_name, datasets="ColB")["ColB"]
            assert isinstance(data, ak.Strings)
            assert data.tolist() == b

        d = {
            cols[0]: ak.array(a),
            cols[1]: ak.array([int(x) for x in b]),
            cols[2]: ak.array([round(float(x), 2) for x in c]),
        }
        with tempfile.TemporaryDirectory(dir=csv_test_base_tmp) as tmp_dirname:
            # test can read csv with header not written by Arkouda
            non_ak_file_name = f"{tmp_dirname}/non_ak.csv"
            with open(non_ak_file_name, "w") as f:
                f.write("**HEADER**\n")
                f.write("str,int64,float64\n")
                f.write("*/HEADER/*\n")
                f.write(",".join(cols) + "\n")
                f.write(f"{a[0]},{b[0]},{c[0]}\n")
                f.write(f"{a[1]},{b[1]},{c[1]}\n")

            # test writing file with Arkouda with non-standard delim
            non_standard_delim_file_name = f"{tmp_dirname}/non_standard_delim"
            ak.to_csv(d, f"{non_standard_delim_file_name}.csv", col_delim="|*|")

            for file_name, delim in [
                (non_ak_file_name, ","),
                (f"{non_standard_delim_file_name}*", "|*|"),
            ]:
                data = ak.read_csv(file_name, column_delim=delim)
                assert list(data.keys()) == cols
                assert data["ColA"].tolist() == a
                assert data["ColB"].tolist() == [int(x) for x in b]
                assert data["ColC"].tolist() == [round(float(x), 2) for x in c]

                # test reading subset of columns
                data = ak.read_csv(file_name, datasets="ColB", column_delim=delim)["ColB"]
                assert isinstance(data, ak.pdarray)
                assert data.tolist() == [int(x) for x in b]

        # larger data set testing
        d = {
            "ColA": ak.randint(0, 50, 101),
            "ColB": ak.randint(0, 50, 101),
            "ColC": ak.randint(0, 50, 101),
        }
        with tempfile.TemporaryDirectory(dir=csv_test_base_tmp) as tmp_dirname:
            ak.to_csv(d, f"{tmp_dirname}/non_equal_set.csv")
            data = ak.read_csv(f"{tmp_dirname}/non_equal_set*")
            assert data["ColA"].tolist() == d["ColA"].tolist()
            assert data["ColB"].tolist() == d["ColB"].tolist()
            assert data["ColC"].tolist() == d["ColC"].tolist()


class TestImportExport:
    @classmethod
    def setup_class(cls):
        cls.pddf = pd.DataFrame(
            data={
                "c_1": np.array([np.iinfo(np.int64).min, -1, 0, np.iinfo(np.int64).max]),
                "c_3": np.array([False, True, False, False]),
                "c_4": np.array([-0.0, np.finfo(np.float64).min, np.nan, np.inf]),
                "c_5": np.array(["abc", " ", "xyz", ""]),
            },
            index=np.arange(4),
        )
        cls.akdf = ak.DataFrame(cls.pddf)

    def test_import_hdf(self, import_export_base_tmp):
        locales = pytest.nl
        with tempfile.TemporaryDirectory(dir=import_export_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/import_hdf_test"

            self.pddf.to_hdf(f"{file_name}_table.h5", key="dataframe", format="table", mode="w")
            akdf = ak.import_data(f"{file_name}_table.h5", write_file=f"{file_name}_ak_table.h5")
            assert len(glob.glob(f"{file_name}_ak_table*.h5")) == locales
            assert self.pddf.equals(akdf.to_pandas())

            self.pddf.to_hdf(
                f"{file_name}_table_cols.h5",
                key="dataframe",
                format="table",
                data_columns=True,
                mode="w",
            )
            akdf = ak.import_data(
                f"{file_name}_table_cols.h5", write_file=f"{file_name}_ak_table_cols.h5"
            )
            assert len(glob.glob(f"{file_name}_ak_table_cols*.h5")) == locales
            assert self.pddf.equals(akdf.to_pandas())

            self.pddf.to_hdf(
                f"{file_name}_fixed.h5",
                key="dataframe",
                format="fixed",
                data_columns=True,
                mode="w",
            )
            akdf = ak.import_data(f"{file_name}_fixed.h5", write_file=f"{file_name}_ak_fixed.h5")
            assert len(glob.glob(f"{file_name}_ak_fixed*.h5")) == locales
            assert self.pddf.equals(akdf.to_pandas())

            with pytest.raises(FileNotFoundError):
                ak.import_data(f"{file_name}_foo.h5", write_file=f"{file_name}_ak_fixed.h5")
            with pytest.raises(RuntimeError):
                ak.import_data(f"{file_name}_*.h5", write_file=f"{file_name}_ak_fixed.h5")

    def test_export_hdf(self, import_export_base_tmp):
        with tempfile.TemporaryDirectory(dir=import_export_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/export_hdf_test"

            self.akdf.to_hdf(f"{file_name}_ak_write")

            pddf = ak.export(
                f"{file_name}_ak_write",
                write_file=f"{file_name}_pd_from_ak.h5",
                index=True,
            )
            assert len(glob.glob(f"{file_name}_pd_from_ak.h5")) == 1
            assert pddf.equals(self.akdf.to_pandas())

            with pytest.raises(RuntimeError):
                ak.export(
                    f"{tmp_dirname}_foo.h5",
                    write_file=f"{tmp_dirname}/pd_from_ak.h5",
                    index=True,
                )

    def test_import_parquet(self, import_export_base_tmp):
        locales = pytest.nl
        with tempfile.TemporaryDirectory(dir=import_export_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/import_pq_test"

            self.pddf.to_parquet(f"{file_name}_table.parquet")
            akdf = ak.import_data(
                f"{file_name}_table.parquet", write_file=f"{file_name}_ak_table.parquet"
            )
            assert len(glob.glob(f"{file_name}_ak_table*.parquet")) == locales
            assert self.pddf.equals(akdf.to_pandas())

    def test_export_parquet(self, import_export_base_tmp):
        with tempfile.TemporaryDirectory(dir=import_export_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/export_pq_test"

            self.akdf.to_parquet(f"{file_name}_ak_write")

            pddf = ak.export(
                f"{file_name}_ak_write",
                write_file=f"{file_name}_pd_from_ak.parquet",
                index=True,
            )
            assert len(glob.glob(f"{file_name}_pd_from_ak.parquet")) == 1
            assert pddf[self.akdf.columns.values].equals(self.akdf.to_pandas())

            with pytest.raises(RuntimeError):
                ak.export(
                    f"{tmp_dirname}_foo.parquet",
                    write_file=f"{tmp_dirname}/pd_from_ak.parquet",
                    index=True,
                )


class TestZarr:
    @pytest.mark.skip
    def test_zarr_read_write(self, zarr_test_base_tmp):
        import arkouda.array_api as Array

        shapes = [(10,), (20,)]
        chunk_shapes = [(2,), (3,)]
        dtypes = [ak.int64, ak.float64]
        for shape, chunk_shape in zip(shapes, chunk_shapes):
            for dtype in dtypes:
                a = Array.full(shape, 7, dtype=dtype)
                with tempfile.TemporaryDirectory(dir=zarr_test_base_tmp) as tmp_dirname:
                    to_zarr(f"{tmp_dirname}", a._array, chunk_shape)
                    b = read_zarr(f"{tmp_dirname}", len(shape), dtype)
                    assert np.allclose(a.to_ndarray(), b.to_ndarray())
