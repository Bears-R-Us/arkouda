import copy
import glob
import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import arkouda as ak
from arkouda import io_util

NUMERIC_TYPES = ["int64", "float64", "bool", "uint64"]
NUMERIC_AND_STR_TYPES = NUMERIC_TYPES + ["str"]


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
        "c_6": ak.SegArray(ak.array([0, 0, 5, 10]), ak.randint(0, 1, 15, dtype=ak.bool)),
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
            ak.array([0, 2, 5, 5]), ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"])
        ),
        "c_11": ak.SegArray(
            ak.array([0, 2, 2, 2]), ak.array(["a", "b", "", "c", "d", "e", "f", "g", "h", "i"])
        ),
        "c_12": ak.SegArray(
            ak.array([0, 0, 2, 2]), ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"])
        ),
        "c_13": ak.SegArray(
            ak.array([0, 2, 3, 3]), ak.array(["", "'", " ", "test", "", "'", "", " ", ""])
        ),
        "c_14": ak.SegArray(
            ak.array([0, 5, 5, 8]), ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"])
        ),
        "c_15": ak.SegArray(
            ak.array([0, 5, 8, 8]), ak.array(["abc", "123", "xyz", "l", "m", "n", "o", "p", "arkouda"])
        ),
    }


class TestParquet:
    par_test_base_tmp = f"{os.getcwd()}/par_io_test"
    io_util.get_directory(par_test_base_tmp)
    COMPRESSIONS = [None, "snappy", "gzip", "brotli", "zstd", "lz4"]

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_read_and_write(self, prob_size, dtype, comp):
        ak_arr = make_ak_arrays(prob_size * pytest.nl, dtype)
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/pq_test_correct"
            ak_arr.to_parquet(file_name, "my-dset", compression=comp)
            pq_arr = ak.read_parquet(f"{file_name}*", "my-dset")
            assert (ak_arr == pq_arr).all()

            # verify generic read works
            gen_arr = ak.read(f"{file_name}*", "my-dset")
            assert (ak_arr == gen_arr).all()

            # verify generic load works
            gen_arr = ak.load(path_prefix=file_name, dataset="my-dset")
            assert (ak_arr == gen_arr).all()

            # verify generic load works with file_format parameter
            gen_arr = ak.load(path_prefix=file_name, dataset="my-dset", file_format="Parquet")
            assert (ak_arr == gen_arr).all()

            # verify load_all works
            gen_arr = ak.load_all(path_prefix=file_name)
            assert (ak_arr == gen_arr["my-dset"]).all()

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    def test_multi_file(self, prob_size, dtype):
        is_multi_loc = pytest.nl != 1
        NUM_FILES = pytest.nl if is_multi_loc else 2
        adjusted_size = int(prob_size / NUM_FILES) * NUM_FILES
        ak_arr = make_ak_arrays(adjusted_size, dtype)

        per_arr = int(adjusted_size / NUM_FILES)
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
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
            pq_arr = ak.read_parquet(f"{file_name}*", "test-dset")
            assert (ak_arr == pq_arr).all()

    def test_wrong_dset_name(self):
        ak_arr = ak.randint(0, 2**32, 100)
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/pq_test"
            ak_arr.to_parquet(file_name, "test-dset-name")

            with pytest.raises(RuntimeError):
                ak.read_parquet(f"{file_name}*", "wrong-dset-name")

            with pytest.raises(ValueError):
                ak.read_parquet(f"{file_name}*", ["test-dset-name", "wrong-dset-name"])

    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_edge_case_read_write(self, dtype, comp):
        np_edge_case = make_edge_case_arrays(dtype)
        ak_edge_case = ak.array(np_edge_case)
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            ak_edge_case.to_parquet(f"{tmp_dirname}/pq_test_edge_case", "my-dset", compression=comp)
            pq_arr = ak.read_parquet(f"{tmp_dirname}/pq_test_edge_case*", "my-dset")
            if dtype == "float64":
                assert np.allclose(np_edge_case, pq_arr.to_ndarray(), equal_nan=True)
            else:
                assert (np_edge_case == pq_arr.to_ndarray()).all()

    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    def test_get_datasets(self, dtype):
        ak_arr = make_ak_arrays(10, dtype)
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            ak_arr.to_parquet(f"{tmp_dirname}/pq_test", "TEST_DSET")
            dsets = ak.get_datasets(f"{tmp_dirname}/pq_test*")
            assert ["TEST_DSET"] == dsets

    def test_append(self):
        # use small size to cut down on execution time
        append_size = 32

        base_dset = ak.randint(0, 2**32, append_size)
        ak_dict = {dt: make_ak_arrays(append_size, dt) for dt in NUMERIC_AND_STR_TYPES}

        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/pq_test"
            base_dset.to_parquet(file_name, "base-dset")

            for key in ak_dict.keys():
                ak_dict[key].to_parquet(file_name, key, mode="append")

            ak_vals = ak.read_parquet(f"{file_name}*")

            for key in ak_dict:
                assert ak_vals[key].to_list() == ak_dict[key].to_list()

    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    def test_append_empty(self, dtype):
        # use small size to cut down on execution time
        ak_arr = make_ak_arrays(32, dtype)
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            ak_arr.to_parquet(f"{tmp_dirname}/pq_test_correct", "my-dset", mode="append")
            pq_arr = ak.read_parquet(f"{tmp_dirname}/pq_test_correct*", "my-dset")

            assert ak_arr.to_list() == pq_arr.to_list()

    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_null_strings(self, comp):
        null_strings = ak.array(["first-string", "", "string2", "", "third", "", ""])
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/null_strings"
            null_strings.to_parquet(file_name, compression=comp)

            ak_data = ak.read_parquet(f"{file_name}*")
            assert (null_strings == ak_data).all()

            # datasets must be specified for get_null_indices
            res = ak.get_null_indices(f"{file_name}*", datasets="strings_array")
            assert [0, 1, 0, 1, 0, 1, 1] == res.to_list()

    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_nan_compressions(self, comp):
        # Reproducer for issue #2005 specifically for gzip
        pdf = pd.DataFrame(
            {
                "all_nan": np.array([np.nan, np.nan, np.nan, np.nan]),
                "some_nan": np.array([3.14, np.nan, 7.12, 4.44]),
            }
        )

        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            pdf.to_parquet(f"{tmp_dirname}/nan_compressed_pq", engine="pyarrow", compression=comp)

            ak_data = ak.read_parquet(f"{tmp_dirname}/nan_compressed_pq")
            rd_df = ak.DataFrame(ak_data)
            pd.testing.assert_frame_equal(rd_df.to_pandas(), pdf)

    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_segarray_read(self, comp):
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
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/segarray_parquet"
            pq.write_table(table, file_name, compression=comp)

            # verify full file read with various object types
            ak_data = ak.read_parquet(f"{file_name}*")
            for k, v in ak_data.items():
                assert isinstance(v, ak.SegArray)
                for x, y in zip(df[k].tolist(), v.to_list()):
                    if isinstance(x, np.ndarray):
                        x = x.tolist()
                    assert x == y if k != "FloatList" else np.allclose(x, y, equal_nan=True)

            # verify individual column selection
            for k, v in df.items():
                ak_data = ak.read_parquet(f"{file_name}*", datasets=k)
                assert isinstance(ak_data, ak.SegArray)
                for x, y in zip(v.tolist(), ak_data.to_list()):
                    if isinstance(x, np.ndarray):
                        x = x.tolist()
                    assert x == y if k != "FloatList" else np.allclose(x, y, equal_nan=True)

        # test for handling empty segments only reading single segarray
        df = pd.DataFrame({"ListCol": [[8], [0, 1], [], [3, 4, 5, 6], []]})
        table = pa.Table.from_pandas(df)
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/empty_segments", compression=comp)

            ak_data = ak.read_parquet(f"{tmp_dirname}/empty_segments*")
            assert isinstance(ak_data, ak.SegArray)
            assert ak_data.size == 5
            for i in range(5):
                assert df["ListCol"][i] == ak_data[i].to_list()

        df = pd.DataFrame(
            {"IntCol": [0, 1, 2, 3], "ListCol": [[0, 1, 2], [0, 1], [3, 4, 5, 6], [1, 2, 3]]}
        )
        table = pa.Table.from_pandas(df)
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/segarray_varied_parquet"
            pq.write_table(table, file_name, compression=comp)

            # read full file
            ak_data = ak.read_parquet(f"{file_name}*")
            for k, v in ak_data.items():
                assert df[k].tolist() == v.to_list()

            # read individual datasets
            ak_data = ak.read_parquet(f"{file_name}*", datasets="IntCol")
            assert isinstance(ak_data, ak.pdarray)
            assert df["IntCol"].to_list() == ak_data.to_list()
            ak_data = ak.read_parquet(f"{file_name}*", datasets="ListCol")
            assert isinstance(ak_data, ak.SegArray)
            assert df["ListCol"].to_list() == ak_data.to_list()

        # test for multi-file with and without empty segs
        is_multi_loc = pytest.nl != 1
        NUM_FILES = pytest.nl if is_multi_loc else 2
        regular = (
            [[0, 1, 2], [0, 1], [3, 4, 5, 6], [1, 2, 3]],
            [[0, 1, 11], [0, 1], [3, 4, 5, 6], [1]],
        )
        first_empty = ([[], [0, 1], [], [3, 4, 5, 6], []], [[0, 1], [], [3, 4, 5, 6], [], [1, 2, 3]])
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
            with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
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
                    concat_segarr.to_parquet(file_name, "test-dset", compression=comp)
                else:
                    # when single locale artifically create multiple files
                    for i in range(NUM_FILES):
                        pq.write_table(tables[i], f"{file_name}_LOCALE{i:04d}", compression=comp)
                ak_data = ak.read_parquet(f"{file_name}*")
                assert isinstance(ak_data, ak.SegArray)
                assert ak_data.size == len(lists[0]) * NUM_FILES
                for i in range(ak_data.size):
                    assert combo["ListCol"][i] == ak_data[i].to_list()

    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    @pytest.mark.parametrize("segarray_create", [segarray_setup, edge_case_segarray_setup])
    def test_segarray_write(self, dtype, segarray_create):
        a, b, c = segarray_create(dtype)
        s = ak.SegArray(ak.array([0, len(a), len(a) + len(b)]), ak.array(a + b + c))
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/segarray_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/segarray_test*")
            for i in range(3):
                x, y = s[i].to_list(), rd_data[i].to_list()
                assert x == y if dtype != "float64" else np.allclose(x, y, equal_nan=True)

        s = ak.SegArray(ak.array([0, 0, len(a), len(a), len(a), len(a) + len(c)]), ak.array(a + c))
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/segarray_test_empty")

            rd_data = ak.read_parquet(f"{tmp_dirname}/segarray_test_empty*")
            for i in range(6):
                x, y = s[i].to_list(), rd_data[i].to_list()
                assert x == y if dtype != "float64" else np.allclose(x, y, equal_nan=True)

    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_multi_col_write(self, comp):
        df_dict = make_multi_dtype_dict()
        akdf = ak.DataFrame(df_dict)
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            # use multi-column write to generate parquet file
            akdf.to_parquet(f"{tmp_dirname}/multi_col_parquet", compression=comp)
            # read files and ensure that all resulting fields are as expected
            rd_data = ak.read_parquet(f"{tmp_dirname}/multi_col_parquet*")
            rd_df = ak.DataFrame(rd_data)
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            # test save with index true
            akdf.to_parquet(f"{tmp_dirname}/multi_col_parquet", index=True, compression=comp)
            rd_data = ak.read_parquet(f"{tmp_dirname}/multi_col_parquet*")
            rd_df = ak.DataFrame(rd_data)
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

    def test_small_ints(self):
        df_pd = pd.DataFrame(
            {
                "int16": pd.Series([2**15 - 1, -(2**15)], dtype=np.int16),
                "int32": pd.Series([2**31 - 1, -(2**31)], dtype=np.int32),
                "uint16": pd.Series([2**15 - 1, 2**15], dtype=np.uint16),
                "uint32": pd.Series([2**31 - 1, 2**31], dtype=np.uint32),
            }
        )
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/pq_small_int"
            df_pd.to_parquet(file_name)
            df_ak = ak.DataFrame(ak.read_parquet(f"{file_name}*"))
            for c in df_ak.column_names:
                assert df_ak[c].to_list() == df_pd[c].to_list()

    def test_read_nested(self):
        df = ak.DataFrame({"idx": ak.arange(5), "seg": ak.SegArray(ak.arange(0, 10, 2), ak.arange(10))})
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/read_nested_test"
            df.to_parquet(file_name)

            # test read with read_nested=true
            data = ak.read_parquet(f"{file_name}*")
            assert "idx" in data
            assert "seg" in data
            assert df["idx"].to_list() == data["idx"].to_list()
            assert df["seg"].to_list() == data["seg"].to_list()

            # test read with read_nested=false and no supplied datasets
            data = ak.read_parquet(f"{file_name}*", read_nested=False)
            assert isinstance(data, ak.pdarray)
            assert df["idx"].to_list() == data.to_list()

            # test read with read_nested=false and user supplied datasets. Should ignore read_nested
            data = ak.read_parquet(f"{file_name}*", datasets=["idx", "seg"], read_nested=False)
            assert "idx" in data
            assert "seg" in data
            assert df["idx"].to_list() == data["idx"].to_list()
            assert df["seg"].to_list() == data["seg"].to_list()

    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_ipv4_columns(self, comp):
        # Added as reproducer for issue #2337
        # test with single IPv4 column
        df = ak.DataFrame({"a": ak.arange(10), "b": ak.IPv4(ak.arange(10))})
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/ipv4_df"
            df.to_parquet(file_name, compression=comp)

            data = ak.read_parquet(f"{file_name}*")
            rd_df = ak.DataFrame({"a": data["a"], "b": ak.IPv4(data["b"])})

            pd.testing.assert_frame_equal(df.to_pandas(), rd_df.to_pandas())

        # test with multiple IPv4 columns
        df = ak.DataFrame({"a": ak.IPv4(ak.arange(10)), "b": ak.IPv4(ak.arange(10))})
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/ipv4_df"
            df.to_parquet(file_name, compression=comp)

            data = ak.read_parquet(f"{file_name}*")
            rd_df = ak.DataFrame({"a": ak.IPv4(data["a"]), "b": ak.IPv4(data["b"])})

            pd.testing.assert_frame_equal(df.to_pandas(), rd_df.to_pandas())

        # test replacement of IPv4 with uint representation
        df = ak.DataFrame({"a": ak.IPv4(ak.arange(10))})
        df["a"] = df["a"].export_uint()
        assert ak.arange(10).to_list() == df["a"].to_list()


class TestHDF5:
    hdf_test_base_tmp = f"{os.getcwd()}/hdf_io_test"
    io_util.get_directory(hdf_test_base_tmp)

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    def test_read_and_write(self, prob_size, dtype):
        ak_arr = make_ak_arrays(prob_size * pytest.nl, dtype)
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/hdf_test_correct"
            ak_arr.to_hdf(file_name)

            # test read_hdf with glob
            gen_arr = ak.read_hdf(f"{file_name}*")
            assert (ak_arr == gen_arr).all()

            # test read_hdf with filenames
            gen_arr = ak.read_hdf(filenames=[f"{file_name}_LOCALE{i:04d}" for i in range(pytest.nl)])
            assert (ak_arr == gen_arr).all()

            # verify generic read works
            gen_arr = ak.read(f"{file_name}*")
            assert (ak_arr == gen_arr).all()

            # verify generic load works
            if dtype == "str":
                # we have to specify the dataset for strings since it differs from default of "array"
                gen_arr = ak.load(path_prefix=file_name, dataset="strings_array")
            else:
                gen_arr = ak.load(path_prefix=file_name)
            assert (ak_arr == gen_arr).all()

            # verify generic load works with file_format parameter
            if dtype == "str":
                # we have to specify the dataset for strings since it differs from default of "array"
                gen_arr = ak.load(path_prefix=file_name, dataset="strings_array", file_format="HDF5")
            else:
                gen_arr = ak.load(path_prefix=file_name, file_format="HDF5")
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
                ak.load(path_prefix=f"{TestHDF5.hdf_test_base_tmp}/not-a-file")

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    def test_read_and_write_dset_provided(self, prob_size, dtype):
        ak_arr = make_ak_arrays(prob_size * pytest.nl, dtype)
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/hdf_test_correct"
            ak_arr.to_hdf(file_name, "my_dset")

            # test read_hdf with glob
            gen_arr = ak.read_hdf(f"{file_name}*", "my_dset")
            assert (ak_arr == gen_arr).all()

            # test read_hdf with filenames
            gen_arr = ak.read_hdf(
                filenames=[f"{file_name}_LOCALE{i:04d}" for i in range(pytest.nl)], datasets="my_dset"
            )
            assert (ak_arr == gen_arr).all()

            # verify generic read works
            gen_arr = ak.read(f"{file_name}*", "my_dset")
            assert (ak_arr == gen_arr).all()

            # verify generic load works
            gen_arr = ak.load(path_prefix=file_name, dataset="my_dset")
            assert (ak_arr == gen_arr).all()

            # verify generic load works with file_format parameter
            gen_arr = ak.load(path_prefix=file_name, dataset="my_dset", file_format="HDF5")
            assert (ak_arr == gen_arr).all()

            # verify load_all works
            gen_arr = ak.load_all(path_prefix=file_name)
            assert (ak_arr == gen_arr["my_dset"]).all()

            # Test load with invalid file
            with pytest.raises(RuntimeError):
                ak.load(path_prefix=f"{TestHDF5.hdf_test_base_tmp}/not-a-file", dataset="my_dset")

    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    def test_edge_case_read_write(self, dtype):
        np_edge_case = make_edge_case_arrays(dtype)
        ak_edge_case = ak.array(np_edge_case)
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            ak_edge_case.to_hdf(f"{tmp_dirname}/hdf_test_edge_case", "my-dset")
            hdf_arr = ak.read_hdf(f"{tmp_dirname}/hdf_test_edge_case*", "my-dset")
            if dtype == "float64":
                assert np.allclose(np_edge_case, hdf_arr.to_ndarray(), equal_nan=True)
            else:
                assert (np_edge_case == hdf_arr.to_ndarray()).all()

    def test_read_and_write_with_dict(self):
        df_dict = make_multi_dtype_dict()
        # extend to include categoricals
        df_dict["cat"] = ak.Categorical(ak.array(["c", "b", "a", "b"]))
        df_dict["cat_from_codes"] = ak.Categorical.from_codes(
            codes=ak.array([2, 1, 0, 1]), categories=ak.array(["a", "b", "c"])
        )
        akdf = ak.DataFrame(df_dict)
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/multi_col_hdf"
            # use multi-column write to generate hdf file
            akdf.to_hdf(file_name)

            # test read_hdf with glob, no datasets specified
            rd_data = ak.read_hdf(f"{file_name}*")
            rd_df = ak.DataFrame(rd_data)
            # fix column ordering see issue #2611
            rd_df = rd_df[akdf.column_names]
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            # test read_hdf with only one dataset specified (each tested)
            for col_name in akdf.column_names:
                gen_arr = ak.read_hdf(f"{file_name}*", datasets=[col_name])
                if akdf[col_name].dtype != ak.float64:
                    assert akdf[col_name].to_list() == gen_arr.to_list()
                else:
                    a = akdf[col_name].to_ndarray()
                    b = gen_arr.to_ndarray()
                    if isinstance(a[0], np.ndarray):
                        assert all(np.allclose(a1, b1, equal_nan=True) for a1, b1 in zip(a, b))
                    else:
                        assert np.allclose(a, b, equal_nan=True)

            # test read_hdf with half of columns names specified as datasets
            half_cols = akdf.column_names[: len(akdf.column_names) // 2]
            rd_data = ak.read_hdf(f"{file_name}*", datasets=half_cols)
            rd_df = ak.DataFrame(rd_data)
            pd.testing.assert_frame_equal(akdf[half_cols].to_pandas(), rd_df[half_cols].to_pandas())

            # test read_hdf with all columns names specified as datasets
            rd_data = ak.read_hdf(f"{file_name}*", datasets=akdf.column_names)
            rd_df = ak.DataFrame(rd_data)
            # fix column ordering see issue #2611
            rd_df = rd_df[akdf.column_names]
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            # test read_hdf with filenames
            rd_data = ak.read_hdf(filenames=[f"{file_name}_LOCALE{i:04d}" for i in range(pytest.nl)])
            rd_df = ak.DataFrame(rd_data)
            # fix column ordering see issue #2611
            rd_df = rd_df[akdf.column_names]
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            # verify generic read works
            rd_data = ak.read(f"{file_name}*")
            rd_df = ak.DataFrame(rd_data)
            # fix column ordering see issue #2611
            rd_df = rd_df[akdf.column_names]
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            for col_name in akdf.column_names:
                # verify generic load works
                gen_arr = ak.load(path_prefix=file_name, dataset=col_name)
                if akdf[col_name].dtype != ak.float64:
                    assert akdf[col_name].to_list() == gen_arr.to_list()
                else:
                    a = akdf[col_name].to_ndarray()
                    b = gen_arr.to_ndarray()
                    if isinstance(a[0], np.ndarray):
                        assert all(np.allclose(a1, b1, equal_nan=True) for a1, b1 in zip(a, b))
                    else:
                        assert np.allclose(a, b, equal_nan=True)

                # verify generic load works with file_format parameter
                gen_arr = ak.load(path_prefix=file_name, dataset=col_name, file_format="HDF5")
                if akdf[col_name].dtype != ak.float64:
                    assert akdf[col_name].to_list() == gen_arr.to_list()
                else:
                    a = akdf[col_name].to_ndarray()
                    b = gen_arr.to_ndarray()
                    if isinstance(a[0], np.ndarray):
                        assert all(np.allclose(a1, b1, equal_nan=True) for a1, b1 in zip(a, b))
                    else:
                        assert np.allclose(a, b, equal_nan=True)

            # Test load with invalid file
            with pytest.raises(RuntimeError):
                ak.load(path_prefix=f"{TestHDF5.hdf_test_base_tmp}/not-a-file", dataset=akdf.column_names[0])

            # verify load_all works
            rd_data = ak.load_all(path_prefix=file_name)
            rd_df = ak.DataFrame(rd_data)
            # fix column ordering see issue #2611
            rd_df = rd_df[akdf.column_names]
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            # Test load_all with invalid file
            with pytest.raises(ValueError):
                ak.load_all(path_prefix=f"{TestHDF5.hdf_test_base_tmp}/not-a-file")

            # test get_datasets
            datasets = ak.get_datasets(f"{file_name}*")
            assert sorted(datasets) == sorted(akdf.column_names)

            # test save with index true
            akdf.to_hdf(file_name, index=True)
            rd_data = ak.read_hdf(f"{file_name}*")
            rd_df = ak.DataFrame(rd_data)
            # fix column ordering see issue #2611
            rd_df = rd_df[akdf.column_names]
            pd.testing.assert_frame_equal(akdf.to_pandas(), rd_df.to_pandas())

            # test get_datasets with index
            datasets = ak.get_datasets(f"{file_name}*")
            assert sorted(datasets) == ["Index"] + sorted(akdf.column_names)

    def test_ls_hdf(self):
        df_dict = make_multi_dtype_dict()
        akdf = ak.DataFrame(df_dict)
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/test_ls_hdf"
            # use multi-column write to generate hdf file
            akdf.to_hdf(file_name)

            message = ak.ls(f"{file_name}_LOCALE0000")
            for col_name in akdf.column_names:
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

    def test_read_hdf_with_error_and_warn(self):
        df_dict = make_multi_dtype_dict()
        akdf = ak.DataFrame(df_dict)
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
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
                RuntimeWarning, match=r"There were .* errors reading files on the server.*"
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
    def test_save_strings_dataset(self, prob_size):
        reg_strings = make_ak_arrays(prob_size, "str")
        # hard coded at 26 because we don't need to test long strings at large scale
        # passing data from python to chpl this way can really slow down as size increases
        long_strings = ak.array(
            [f"testing a longer string{num} to be written, loaded and appended" for num in range(26)]
        )

        for strings_array in [reg_strings, long_strings]:
            with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
                file_name = f"{tmp_dirname}/test_strings_hdf"
                strings_array.to_hdf(file_name)
                r_strings_array = ak.read_hdf(f"{file_name}*")
                assert (strings_array == r_strings_array).all()

                # Read a part of a saved Strings dataset from one hdf5 file
                r_strings_subset = ak.read_hdf(filenames=f"{file_name}_LOCALE0000")
                assert isinstance(r_strings_subset, ak.Strings)
                assert (strings_array[: r_strings_subset.size] == r_strings_subset).all()

                # Repeat the test using the calc_string_offsets=True option to
                # have server calculate offsets array
                r_strings_subset = ak.read_hdf(
                    filenames=f"{file_name}_LOCALE0000", calc_string_offsets=True
                )
                assert isinstance(r_strings_subset, ak.Strings)
                assert (strings_array[: r_strings_subset.size] == r_strings_subset).all()

                # test append
                strings_array.to_hdf(file_name, dataset="strings-dupe", mode="append")
                r_strings = ak.read_hdf(f"{file_name}*", datasets="strings_array")
                r_strings_dupe = ak.read_hdf(f"{file_name}*", datasets="strings-dupe")
                assert (r_strings == r_strings_dupe).all()

    def test_save_multi_type_dict_dataset(self):
        df_dict = make_multi_dtype_dict()
        # extend to include categoricals
        df_dict["cat"] = ak.Categorical(ak.array(["c", "b", "a", "b"]))
        df_dict["cat_from_codes"] = ak.Categorical.from_codes(
            codes=ak.array([2, 1, 0, 1]), categories=ak.array(["a", "b", "c"])
        )
        keys = list(df_dict.keys())
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/multi_type_dict_test"
            # use multi-column write to generate hdf file
            ak.to_hdf(df_dict, file_name)
            r_mixed = ak.read_hdf(f"{file_name}*")

            for col_name in keys:
                # verify load by dataset and returned mixed dict at col_name
                loaded = ak.load(file_name, dataset=col_name)
                for arr in [loaded, r_mixed[col_name]]:
                    if df_dict[col_name].dtype != ak.float64:
                        assert df_dict[col_name].to_list() == arr.to_list()
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
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/multi_type_dict_test"
            single_arr.to_hdf(file_name, dataset=keys[0])

            ak.to_hdf(rest_dict, file_name, mode="append")
            r_mixed = ak.read_hdf(f"{file_name}*")

            for col_name in keys:
                # verify load by dataset and returned mixed dict at col_name
                loaded = ak.load(file_name, dataset=col_name)
                for arr in [loaded, r_mixed[col_name]]:
                    if df_dict[col_name].dtype != ak.float64:
                        assert df_dict[col_name].to_list() == arr.to_list()
                    else:
                        a = df_dict[col_name].to_ndarray()
                        b = arr.to_ndarray()
                        if isinstance(a[0], np.ndarray):
                            assert all(np.allclose(a1, b1, equal_nan=True) for a1, b1 in zip(a, b))
                        else:
                            assert np.allclose(a, b, equal_nan=True)

    def test_strict_types(self):
        N = 100
        int_types = [np.uint32, np.int64, np.uint16, np.int16]
        float_types = [np.float32, np.float64, np.float32, np.float64]
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
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
            assert a["integers"].to_list() == np.arange(len(int_types) * N).tolist()
            assert np.allclose(
                a["floats"].to_ndarray(), np.arange(len(float_types) * N, dtype=np.float64)
            )

    def test_small_arrays(self):
        for arr in [ak.array([1]), ak.array(["ab", "cd"]), ak.array(["123456789"])]:
            with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
                arr.to_hdf(f"{tmp_dirname}/small_numeric")
                ret_arr = ak.read_hdf(f"{tmp_dirname}/small_numeric*")
                assert (arr == ret_arr).all()

    def test_bigint(self):
        df_dict = {
            "pdarray": ak.arange(2**200, 2**200 + 3, max_bits=201),
            "arrayview": ak.arange(2**200, 2**200 + 27, max_bits=201).reshape((3, 3, 3)),
            "groupby": ak.GroupBy(ak.arange(2**200, 2**200 + 5)),
            "segarray": ak.SegArray(
                ak.arange(0, 10, 2), ak.arange(2**200, 2**200 + 10, max_bits=212)
            ),
        }
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/bigint_test"
            ak.to_hdf(df_dict, file_name)
            ret_dict = ak.read_hdf(f"{tmp_dirname}/bigint_test*")

            pda_loaded = ak.read_hdf(f"{tmp_dirname}/bigint_test*", datasets="pdarray")
            a = df_dict["pdarray"]
            for rd_a in [ret_dict["pdarray"], pda_loaded]:
                assert isinstance(rd_a, ak.pdarray)
                assert a.to_list() == rd_a.to_list()
                assert a.max_bits == rd_a.max_bits

            av_loaded = ak.read_hdf(f"{tmp_dirname}/bigint_test*", datasets="arrayview")
            av = df_dict["arrayview"]
            for rd_av in [ret_dict["arrayview"], av_loaded]:
                assert isinstance(rd_av, ak.ArrayView)
                assert av.base.to_list() == rd_av.base.to_list()
                assert av.base.max_bits == rd_av.base.max_bits

            g_loaded = ak.read_hdf(f"{tmp_dirname}/bigint_test*", datasets="groupby")
            g = df_dict["groupby"]
            for rd_g in [ret_dict["groupby"], g_loaded]:
                assert isinstance(rd_g, ak.GroupBy)
                assert g.keys.to_list() == rd_g.keys.to_list()
                assert g.unique_keys.to_list() == rd_g.unique_keys.to_list()
                assert g.permutation.to_list() == rd_g.permutation.to_list()
                assert g.segments.to_list() == rd_g.segments.to_list()

            sa_loaded = ak.read_hdf(f"{tmp_dirname}/bigint_test*", datasets="segarray")
            sa = df_dict["segarray"]
            for rd_sa in [ret_dict["segarray"], sa_loaded]:
                assert isinstance(rd_sa, ak.SegArray)
                assert sa.values.to_list() == rd_sa.values.to_list()
                assert sa.segments.to_list() == rd_sa.segments.to_list()

    def test_unsanitized_dataset_names(self):
        # Test when quotes are part of the dataset name
        my_arrays = {'foo"0"': ak.arange(100), 'bar"': ak.arange(100)}
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            ak.to_hdf(my_arrays, f"{tmp_dirname}/bad_dataset_names")
            ak.read_hdf(f"{tmp_dirname}/bad_dataset_names*")

    def test_internal_versions_and_legacy_read(self):
        """
        Test loading legacy files to ensure they can still be read.
        Test loading internal arkouda hdf5 structuring by loading v0 and v1 files.
        v1 contains _arkouda_metadata group and attributes, v0 does not.
        Files are located under `test/resources` ... where server-side unit tests are located.
        """
        # Note: pytest unit tests are located under "tests/" vs chapel "test/"
        cwd = os.getcwd()
        if cwd.endswith("tests"):  # IDEs may launch unit tests from this location
            # resources live two levels up
            cwd += "/../../resources/hdf5-testing"
        else:  # assume arkouda root dir
            cwd += "/resources/hdf5-testing"

        rd_arr = ak.read_hdf(f"{cwd}/Legacy_String.hdf5")
        assert ["ABC", "DEF", "GHI"] == rd_arr.to_list()

        v0 = ak.load(f"{cwd}/array_v0.hdf5", file_format="hdf5")
        v1 = ak.load(f"{cwd}/array_v1.hdf5", file_format="hdf5")
        assert 50 == v0.size
        assert 50 == v1.size

    def test_multi_dim_read_write(self):
        av = ak.ArrayView(ak.arange(27), ak.array([3, 3, 3]))
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            av.to_hdf(f"{tmp_dirname}/multi_dim_test", dataset="MultiDimObj", mode="append")
            read_av = ak.read_hdf(f"{tmp_dirname}/multi_dim_test*", datasets="MultiDimObj")
            assert np.array_equal(av.to_ndarray(), read_av.to_ndarray())

    def test_hdf_groupby(self):
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

        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            for g in [pda_grouping, str_grouping, cat_grouping]:
                g.to_hdf(f"{tmp_dirname}/groupby_test")
                g_load = ak.read(f"{tmp_dirname}/groupby_test*")
                assert len(g_load.keys) == len(g.keys)
                assert g_load.permutation.to_list() == g.permutation.to_list()
                assert g_load.segments.to_list() == g.segments.to_list()
                assert g_load._uki.to_list() == g._uki.to_list()
                if isinstance(g.keys[0], ak.Categorical):
                    for k, kload in zip(g.keys, g_load.keys):
                        assert k.to_list() == kload.to_list()
                else:
                    assert g_load.keys.to_list() == g.keys.to_list()

    def test_hdf_categorical(self):
        cat = ak.Categorical(ak.array(["a", "b", "a", "b", "c"]))
        cat_from_codes = ak.Categorical.from_codes(
            codes=ak.array([0, 1, 0, 1, 2]), categories=ak.array(["a", "b", "c"])
        )
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            for c in cat, cat_from_codes:
                c.to_hdf(f"{tmp_dirname}/categorical_test")
                c_load = ak.read(f"{tmp_dirname}/categorical_test*")

                assert c_load.categories.to_list() == (["a", "b", "c", "N/A"])
                if c.segments is not None:
                    assert c.segments.to_list() == c_load.segments.to_list()
                    assert c.permutation.to_list() == c_load.permutation.to_list()

    def test_hdf_overwrite_pdarray(self):
        # test repack with a single object
        a = ak.arange(1000)
        b = ak.randint(0, 100, 1000)
        c = ak.arange(15)
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
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
                assert data["array"].to_list() == c.to_list()

        # test overwrites with different types
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/pda_test"
            a.to_hdf(file_name)
            for size, dtype in [(15, ak.uint64), (150, ak.float64), (1000, ak.bool)]:
                b = ak.arange(size, dtype=dtype)
                b.update_hdf(file_name)
                data = ak.read_hdf(f"{file_name}*")
                assert data.to_list() == b.to_list()

    def test_hdf_overwrite_strings(self):
        # test repack with a single object
        a = ak.random_strings_uniform(0, 16, 1000)
        b = ak.random_strings_uniform(0, 16, 1000)
        c = ak.random_strings_uniform(0, 16, 10)
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
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
                assert data["test_str"].to_list() == c.to_list()

    def test_overwrite_categorical(self):
        a = ak.Categorical(ak.array([f"cat_{i%3}" for i in range(100)]))
        b = ak.Categorical(ak.array([f"cat_{i%4}" for i in range(100)]))
        c = ak.Categorical(ak.array([f"cat_{i%5}" for i in range(10)]))
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
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
                assert getattr(d, attr).to_list() == getattr(a, attr).to_list()

    def test_hdf_overwrite_dataframe(self):
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
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
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

    def test_overwrite_segarray(self):
        sa1 = ak.SegArray(ak.arange(0, 1000, 5), ak.arange(1000))
        sa2 = ak.SegArray(ak.arange(0, 100, 5), ak.arange(100))
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
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

    def test_overwrite_arrayview(self):
        av = ak.arange(27).reshape((3, 3, 3))
        av2 = ak.arange(8).reshape((2, 2, 2))
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/array_view_test"
            av.to_hdf(file_name)
            av2.update_hdf(file_name, repack=False)
            data = ak.read_hdf(f"{file_name}*")
            assert av2.to_list() == data.to_list()

    def test_overwrite_single_dset(self):
        # we need to test that both repack=False and repack=True generate the same file size here
        a = ak.arange(1000)
        b = ak.arange(15)
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            a.to_hdf(f"{tmp_dirname}/test_file")
            b.update_hdf(f"{tmp_dirname}/test_file")
            f_list = glob.glob(f"{tmp_dirname}/test_file*")
            f1_size = sum(os.path.getsize(f) for f in f_list)

            a.to_hdf(f"{tmp_dirname}/test_file_2")
            b.update_hdf(f"{tmp_dirname}/test_file_2", repack=False)
            f_list = glob.glob(f"{tmp_dirname}/test_file_2_*")
            f2_size = sum(os.path.getsize(f) for f in f_list)

            assert f1_size == f2_size

    def test_snapshot(self):
        df = ak.DataFrame(make_multi_dtype_dict())
        df_str_idx = df.copy()
        df_str_idx._set_index([f"A{i}" for i in range(len(df))])
        col_order = df.column_names
        df_ref = df.to_pandas()
        df_str_idx_ref = df_str_idx.to_pandas(retain_index=True)
        a = ak.randint(0, 10, 100)
        s = ak.random_strings_uniform(0, 5, 50)
        c = ak.Categorical(s)
        g = ak.GroupBy(a)
        ref_data = {"a": a, "s": s, "c": c, "g": g}

        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
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
                df_str_idx_ref[col_order], data["df_str_idx"].to_pandas(retain_index=True)[col_order]
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
    def test_index_save_and_load(self, dtype, size):
        idx = ak.Index(make_ak_arrays(size, dtype))
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            idx.to_hdf(f"{tmp_dirname}/idx_test")
            rd_idx = ak.read_hdf(f"{tmp_dirname}/idx_test*")

            assert isinstance(rd_idx, ak.Index)
            assert type(rd_idx.values) == type(idx.values)
            assert idx.to_list() == rd_idx.to_list()

        if dtype == ak.str_:
            # if strings we need to also test Categorical
            idx = ak.Index(ak.Categorical(make_ak_arrays(size, dtype)))
            with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
                idx.to_hdf(f"{tmp_dirname}/idx_test")
                rd_idx = ak.read_hdf(f"{tmp_dirname}/idx_test*")

                assert isinstance(rd_idx, ak.Index)
                assert type(rd_idx.values) == type(idx.values)
                assert idx.to_list() == rd_idx.to_list()

    @pytest.mark.parametrize("dtype1", NUMERIC_AND_STR_TYPES)
    @pytest.mark.parametrize("dtype2", NUMERIC_AND_STR_TYPES)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_multi_index(self, dtype1, dtype2, size):
        t1 = make_ak_arrays(size, dtype1)
        t2 = make_ak_arrays(size, dtype2)
        idx = ak.Index.factory([t1, t2])
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            idx.to_hdf(f"{tmp_dirname}/idx_test")
            rd_idx = ak.read_hdf(f"{tmp_dirname}/idx_test*")

            assert isinstance(rd_idx, ak.MultiIndex)
            assert idx.to_list() == rd_idx.to_list()

        # handle categorical cases as well
        if ak.str_ in [dtype1, dtype2]:
            if dtype1 == ak.str_:
                t1 = ak.Categorical(t1)
            if dtype2 == ak.str_:
                t2 = ak.Categorical(t2)
            idx = ak.Index.factory([t1, t2])
            with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
                idx.to_hdf(f"{tmp_dirname}/idx_test")
                rd_idx = ak.read_hdf(f"{tmp_dirname}/idx_test*")

                assert isinstance(rd_idx, ak.MultiIndex)
                assert idx.to_list() == rd_idx.to_list()

    def test_hdf_overwrite_index(self):
        # test repack with a single object
        a = ak.Index(ak.arange(1000))
        b = ak.Index(ak.randint(0, 100, 1000))
        c = ak.Index(ak.arange(15))
        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
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
                assert data["index"].to_list() == c.to_list()

    def test_special_objtype(self):
        """
        This test is simply to ensure that the dtype is persisted through the io
        operation. It ultimately uses the process of pdarray, but need to ensure
        correct Arkouda Object Type is returned
        """
        ip = ak.IPv4(ak.arange(10))
        dt = ak.Datetime(ak.arange(10))
        td = ak.Timedelta(ak.arange(10))
        df = ak.DataFrame({"ip": ip, "datetime": dt, "timedelta": td})

        with tempfile.TemporaryDirectory(dir=TestHDF5.hdf_test_base_tmp) as tmp_dirname:
            ip.to_hdf(f"{tmp_dirname}/ip_test")
            rd_ip = ak.read_hdf(f"{tmp_dirname}/ip_test*")
            assert isinstance(rd_ip, ak.IPv4)
            assert ip.to_list() == rd_ip.to_list()

            dt.to_hdf(f"{tmp_dirname}/dt_test")
            rd_dt = ak.read_hdf(f"{tmp_dirname}/dt_test*")
            assert isinstance(rd_dt, ak.Datetime)
            assert dt.to_list() == rd_dt.to_list()

            td.to_hdf(f"{tmp_dirname}/td_test")
            rd_td = ak.read_hdf(f"{tmp_dirname}/td_test*")
            assert isinstance(rd_td, ak.Timedelta)
            assert td.to_list() == rd_td.to_list()

            df.to_hdf(f"{tmp_dirname}/df_test")
            rd_df = ak.read_hdf(f"{tmp_dirname}/df_test*")

            assert isinstance(rd_df["ip"], ak.IPv4)
            assert isinstance(rd_df["datetime"], ak.Datetime)
            assert isinstance(rd_df["timedelta"], ak.Timedelta)
            assert df["ip"].to_list() == rd_df["ip"].to_list()
            assert df["datetime"].to_list() == rd_df["datetime"].to_list()
            assert df["timedelta"].to_list() == rd_df["timedelta"].to_list()


class TestCSV:
    csv_test_base_tmp = f"{os.getcwd()}/csv_io_test"
    io_util.get_directory(csv_test_base_tmp)

    def test_csv_read_write(self):
        # first test that can read csv with no header not written by Arkouda
        cols = ["ColA", "ColB", "ColC"]
        a = ["ABC", "DEF"]
        b = ["123", "345"]
        c = ["3.14", "5.56"]
        with tempfile.TemporaryDirectory(dir=TestCSV.csv_test_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/non_ak.csv"
            with open(file_name, "w") as f:
                f.write(",".join(cols) + "\n")
                f.write(f"{a[0]},{b[0]},{c[0]}\n")
                f.write(f"{a[1]},{b[1]},{c[1]}\n")

            data = ak.read_csv(file_name)
            assert list(data.keys()) == cols
            assert data["ColA"].to_list() == a
            assert data["ColB"].to_list() == b
            assert data["ColC"].to_list() == c

            data = ak.read_csv(file_name, datasets="ColB")
            assert isinstance(data, ak.Strings)
            assert data.to_list() == b

        d = {
            cols[0]: ak.array(a),
            cols[1]: ak.array([int(x) for x in b]),
            cols[2]: ak.array([round(float(x), 2) for x in c]),
        }
        with tempfile.TemporaryDirectory(dir=TestCSV.csv_test_base_tmp) as tmp_dirname:
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
                assert data["ColA"].to_list() == a
                assert data["ColB"].to_list() == [int(x) for x in b]
                assert data["ColC"].to_list() == [round(float(x), 2) for x in c]

                # test reading subset of columns
                data = ak.read_csv(file_name, datasets="ColB", column_delim=delim)
                assert isinstance(data, ak.pdarray)
                assert data.to_list() == [int(x) for x in b]

        # larger data set testing
        d = {
            "ColA": ak.randint(0, 50, 101),
            "ColB": ak.randint(0, 50, 101),
            "ColC": ak.randint(0, 50, 101),
        }
        with tempfile.TemporaryDirectory(dir=TestCSV.csv_test_base_tmp) as tmp_dirname:
            ak.to_csv(d, f"{tmp_dirname}/non_equal_set.csv")
            data = ak.read_csv(f"{tmp_dirname}/non_equal_set*")
            assert data["ColA"].to_list() == d["ColA"].to_list()
            assert data["ColB"].to_list() == d["ColB"].to_list()
            assert data["ColC"].to_list() == d["ColC"].to_list()


class TestImportExport:
    import_export_base_tmp = f"{os.getcwd()}/import_export_test"
    io_util.get_directory(import_export_base_tmp)

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

    def test_import_hdf(self):
        locales = pytest.nl
        with tempfile.TemporaryDirectory(dir=TestImportExport.import_export_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/import_hdf_test"

            self.pddf.to_hdf(f"{file_name}_table.h5", "dataframe", format="Table", mode="w")
            akdf = ak.import_data(f"{file_name}_table.h5", write_file=f"{file_name}_ak_table.h5")
            assert len(glob.glob(f"{file_name}_ak_table*.h5")) == locales
            assert self.pddf.equals(akdf.to_pandas())

            self.pddf.to_hdf(
                f"{file_name}_table_cols.h5", "dataframe", format="Table", data_columns=True, mode="w"
            )
            akdf = ak.import_data(
                f"{file_name}_table_cols.h5", write_file=f"{file_name}_ak_table_cols.h5"
            )
            assert len(glob.glob(f"{file_name}_ak_table_cols*.h5")) == locales
            assert self.pddf.equals(akdf.to_pandas())

            self.pddf.to_hdf(
                f"{file_name}_fixed.h5", "dataframe", format="fixed", data_columns=True, mode="w"
            )
            akdf = ak.import_data(f"{file_name}_fixed.h5", write_file=f"{file_name}_ak_fixed.h5")
            assert len(glob.glob(f"{file_name}_ak_fixed*.h5")) == locales
            assert self.pddf.equals(akdf.to_pandas())

            with pytest.raises(FileNotFoundError):
                ak.import_data(f"{file_name}_foo.h5", write_file=f"{file_name}_ak_fixed.h5")
            with pytest.raises(RuntimeError):
                ak.import_data(f"{file_name}_*.h5", write_file=f"{file_name}_ak_fixed.h5")

    def test_export_hdf(self):
        with tempfile.TemporaryDirectory(dir=TestImportExport.import_export_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/export_hdf_test"

            self.akdf.to_hdf(f"{file_name}_ak_write")

            pddf = ak.export(
                f"{file_name}_ak_write", write_file=f"{file_name}_pd_from_ak.h5", index=True
            )
            assert len(glob.glob(f"{file_name}_pd_from_ak.h5")) == 1
            assert pddf.equals(self.akdf.to_pandas())

            with pytest.raises(RuntimeError):
                ak.export(f"{tmp_dirname}_foo.h5", write_file=f"{tmp_dirname}/pd_from_ak.h5", index=True)

    def test_import_parquet(self):
        locales = pytest.nl
        with tempfile.TemporaryDirectory(dir=TestImportExport.import_export_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/import_pq_test"

            self.pddf.to_parquet(f"{file_name}_table.parquet")
            akdf = ak.import_data(
                f"{file_name}_table.parquet", write_file=f"{file_name}_ak_table.parquet"
            )
            assert len(glob.glob(f"{file_name}_ak_table*.parquet")) == locales
            assert self.pddf.equals(akdf.to_pandas())

    def test_export_parquet(self):
        with tempfile.TemporaryDirectory(dir=TestImportExport.import_export_base_tmp) as tmp_dirname:
            file_name = f"{tmp_dirname}/export_pq_test"

            self.akdf.to_parquet(f"{file_name}_ak_write")

            pddf = ak.export(
                f"{file_name}_ak_write", write_file=f"{file_name}_pd_from_ak.parquet", index=True
            )
            assert len(glob.glob(f"{file_name}_pd_from_ak.parquet")) == 1
            assert pddf[self.akdf.column_names].equals(self.akdf.to_pandas())

            with pytest.raises(RuntimeError):
                ak.export(
                    f"{tmp_dirname}_foo.parquet",
                    write_file=f"{tmp_dirname}/pd_from_ak.parquet",
                    index=True,
                )
