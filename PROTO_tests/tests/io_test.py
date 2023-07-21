import copy
import glob
import os
import tempfile

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


def make_multi_col_df():
    return {
        "c_1": ak.arange(3),
        "c_2": ak.SegArray(ak.array([0, 9, 14]), ak.arange(20)),
        "c_3": ak.arange(3, 6, dtype=ak.uint64),
        "c_4": ak.SegArray(ak.array([0, 5, 10]), ak.arange(15, dtype=ak.uint64)),
        "c_5": ak.array([False, True, False]),
        "c_6": ak.SegArray(ak.array([0, 5, 10]), ak.randint(0, 1, 15, dtype=ak.bool)),
        "c_7": ak.array(np.random.uniform(0, 100, 3)),
        "c_8": ak.SegArray(ak.array([0, 9, 14]), ak.array(np.random.uniform(0, 100, 20))),
        "c_9": ak.array(["abc", "123", "xyz"]),
        "c_10": ak.SegArray(
            ak.array([0, 2, 5]), ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"])
        ),
        "c_11": ak.SegArray(
            ak.array([0, 2, 2]), ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"])
        ),
        "c_12": ak.SegArray(
            ak.array([0, 0, 2]), ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"])
        ),
        "c_13": ak.SegArray(
            ak.array([0, 5, 8]), ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"])
        ),
        "c_14": ak.SegArray(
            ak.array([0, 5, 8]), ak.array(["abc", "123", "xyz", "l", "m", "n", "o", "p", "arkouda"])
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
            # compression doesn't work for bool see issue #2579
            ak_arr.to_parquet(
                f"{tmp_dirname}/pq_test_correct",
                "my-dset",
                compression=comp #if dtype != "bool" else None,
            )
            pq_arr = ak.read_parquet(f"{tmp_dirname}/pq_test_correct*", "my-dset")
            assert (ak_arr == pq_arr).all()

            # verify generic read works
            gen_arr = ak.read(f"{tmp_dirname}/pq_test_correct*", "my-dset")
            assert (ak_arr == gen_arr).all()

            # verify generic load works
            gen_arr = ak.load(path_prefix=f"{tmp_dirname}/pq_test_correct", dataset="my-dset")
            assert (ak_arr == gen_arr).all()

            # verify load_all works
            gen_arr = ak.load_all(path_prefix=f"{tmp_dirname}/pq_test_correct")
            assert (ak_arr == gen_arr['my-dset']).all()

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
            ak_arr.to_parquet(f"{tmp_dirname}/pq_test", "test-dset-name")

            with pytest.raises(RuntimeError):
                ak.read_parquet(f"{tmp_dirname}/pq_test*", "wrong-dset-name")

            with pytest.raises(ValueError):
                ak.read_parquet(f"{tmp_dirname}/pq_test*", ["test-dset-name", "wrong-dset-name"])

    @pytest.mark.parametrize("dtype", NUMERIC_AND_STR_TYPES)
    @pytest.mark.parametrize("comp", COMPRESSIONS)
    def test_edge_case_read_write(self, dtype, comp):
        np_edge_case = make_edge_case_arrays(dtype)
        ak_edge_case = ak.array(np_edge_case)
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            # compression doesn't work for bool see issue #2579
            ak_edge_case.to_parquet(
                f"{tmp_dirname}/pq_test_edge_case",
                "my-dset",
                compression=comp #if dtype != "bool" else None,
            )
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
            base_dset.to_parquet(f"{tmp_dirname}/pq_test", "base-dset")

            for key in ak_dict.keys():
                ak_dict[key].to_parquet(f"{tmp_dirname}/pq_test", key, mode="append")

            ak_vals = ak.read_parquet(f"{tmp_dirname}/pq_test*")

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
            null_strings.to_parquet(f"{tmp_dirname}/null_strings", compression=comp)

            ak_data = ak.read_parquet(f"{tmp_dirname}/null_strings*")
            assert (null_strings == ak_data).all()

            # datasets must be specified for get_null_indices
            res = ak.get_null_indices(f"{tmp_dirname}/null_strings*", datasets="strings_array")
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
            pq.write_table(table, f"{tmp_dirname}/segarray_parquet", compression=comp)

            # verify full file read with various object types
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_parquet*")
            for k, v in ak_data.items():
                assert isinstance(v, ak.SegArray)
                for x, y in zip(df[k].tolist(), v.to_list()):
                    if isinstance(x, np.ndarray):
                        x = x.tolist()
                    assert x == y if k != "FloatList" else np.allclose(x, y, equal_nan=True)

            # verify individual column selection
            for k, v in df.items():
                ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_parquet*", datasets=k)
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
            pq.write_table(table, f"{tmp_dirname}/segarray_varied_parquet", compression=comp)

            # read full file
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_varied_parquet*")
            for k, v in ak_data.items():
                assert df[k].tolist() == v.to_list()

            # read individual datasets
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_varied_parquet*", datasets="IntCol")
            assert isinstance(ak_data, ak.pdarray)
            assert df["IntCol"].to_list() == ak_data.to_list()
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_varied_parquet*", datasets="ListCol")
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
                            ak.SegArray.from_multi_array([ak.array(a, dtype=ak.int64) for a in l])
                            for l in lists
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
        # TODO update to add compression after this is added for bools in issue #2579
        df_dict = make_multi_col_df()
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
            fname = f"{tmp_dirname}/pq_small_int"
            df_pd.to_parquet(fname)
            df_ak = ak.DataFrame(ak.read_parquet(fname + "*"))
            for c in df_ak.columns:
                assert df_ak[c].to_list() == df_pd[c].to_list()

    def test_read_nested(self):
        df = ak.DataFrame({"idx": ak.arange(5), "seg": ak.SegArray(ak.arange(0, 10, 2), ak.arange(10))})
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            fname = tmp_dirname + "/read_nested_test"
            df.to_parquet(fname)

            # test read with read_nested=true
            data = ak.read_parquet(fname + "_*")
            assert "idx" in data
            assert "seg" in data
            assert df["idx"].to_list() == data["idx"].to_list()
            assert df["seg"].to_list() == data["seg"].to_list()

            # test read with read_nested=false and no supplied datasets
            data = ak.read_parquet(fname + "_*", read_nested=False)
            assert isinstance(data, ak.pdarray)
            assert df["idx"].to_list() == data.to_list()

            # test read with read_nested=false and user supplied datasets. Should ignore read_nested
            data = ak.read_parquet(fname + "_*", datasets=["idx", "seg"], read_nested=False)
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
            fname = f"{tmp_dirname}/ipv4_df"
            df.to_parquet(fname, compression=comp)

            data = ak.read_parquet(f"{fname}*")
            rd_df = ak.DataFrame({"a": data["a"], "b": ak.IPv4(data["b"])})

            pd.testing.assert_frame_equal(df.to_pandas(), rd_df.to_pandas())

        # test with multiple IPv4 columns
        df = ak.DataFrame({"a": ak.IPv4(ak.arange(10)), "b": ak.IPv4(ak.arange(10))})
        with tempfile.TemporaryDirectory(dir=TestParquet.par_test_base_tmp) as tmp_dirname:
            fname = f"{tmp_dirname}/ipv4_df"
            df.to_parquet(fname, compression=comp)

            data = ak.read_parquet(f"{fname}*")
            rd_df = ak.DataFrame({"a": ak.IPv4(data["a"]), "b": ak.IPv4(data["b"])})

            pd.testing.assert_frame_equal(df.to_pandas(), rd_df.to_pandas())

        # test replacement of IPv4 with uint representation
        df = ak.DataFrame({"a": ak.IPv4(ak.arange(10))})
        df["a"] = df["a"].export_uint()
        assert ak.arange(10).to_list() == df["a"].to_list()
