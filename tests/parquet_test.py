import glob
import os
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from base_test import ArkoudaTest
from context import arkouda as ak
from pandas.testing import assert_series_equal

from arkouda import io_util

TYPES = ("int64", "uint64", "bool", "float64", "str")
COMPRESSIONS = ["snappy", "gzip", "brotli", "zstd", "lz4"]
SIZE = 100
NUMFILES = 5
verbose = True


class ParquetTest(ArkoudaTest):
    @classmethod
    def setUpClass(cls):
        super(ParquetTest, cls).setUpClass()
        ParquetTest.par_test_base_tmp = "{}/par_io_test".format(os.getcwd())
        io_util.get_directory(ParquetTest.par_test_base_tmp)

    def test_parquet(self):
        for dtype in TYPES:
            if dtype == "int64":
                ak_arr = ak.randint(0, 2**32, SIZE)
            elif dtype == "uint64":
                ak_arr = ak.randint(0, 2**32, SIZE, dtype=ak.uint64)
            elif dtype == "bool":
                ak_arr = ak.randint(0, 1, SIZE, dtype=ak.bool_)
            elif dtype == "float64":
                ak_arr = ak.randint(0, 2**32, SIZE, dtype=ak.float64)
            elif dtype == "str":
                ak_arr = ak.random_strings_uniform(1, 10, SIZE)

            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                ak_arr.to_parquet(f"{tmp_dirname}/pq_testcorrect", "my-dset")
                pq_arr = ak.read_parquet(f"{tmp_dirname}/pq_testcorrect*", "my-dset")["my-dset"]
                self.assertListEqual(ak_arr.to_list(), pq_arr.to_list())

            # verify generic read
            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                ak_arr.to_parquet(f"{tmp_dirname}/pq_testcorrect", "my-dset")
                pq_arr = ak.read(f"{tmp_dirname}/pq_testcorrect*", "my-dset")["my-dset"]
                self.assertListEqual(ak_arr.to_list(), pq_arr.to_list())

    def test_multi_file(self):
        for dtype in TYPES:
            adjusted_size = int(SIZE / NUMFILES) * NUMFILES
            test_arrs = []
            if dtype == "int64":
                elems = ak.randint(0, 2**32, adjusted_size)
            elif dtype == "uint64":
                elems = ak.randint(0, 2**32, adjusted_size, dtype=ak.uint64)
            elif dtype == "bool":
                elems = ak.randint(0, 1, adjusted_size, dtype=ak.bool_)
            elif dtype == "float64":
                elems = ak.randint(0, 2**32, adjusted_size, dtype=ak.float64)
            elif dtype == "str":
                elems = ak.random_strings_uniform(1, 10, adjusted_size)

            per_arr = int(adjusted_size / NUMFILES)
            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                for i in range(NUMFILES):
                    test_arrs.append(elems[(i * per_arr) : (i * per_arr) + per_arr])
                    test_arrs[i].to_parquet(f"{tmp_dirname}/pq_test{i:04d}", "test-dset")

                pq_arr = ak.read_parquet(f"{tmp_dirname}/pq_test*", "test-dset")["test-dset"]
                self.assertListEqual(elems.to_list(), pq_arr.to_list())

    def test_wrong_dset_name(self):
        ak_arr = ak.randint(0, 2**32, SIZE)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            ak_arr.to_parquet(f"{tmp_dirname}/pq_test", "test-dset-name")

            with self.assertRaises(RuntimeError):
                ak.read_parquet(f"{tmp_dirname}/pq_test*", "wrong-dset-name")

            with self.assertRaises(ValueError):
                ak.read_parquet(f"{tmp_dirname}/pq_test*", ["test-dset-name", "wrong-dset-name"])

    def test_max_read_write(self):
        for dtype in TYPES:
            if dtype == "int64":
                val = np.iinfo(np.int64).max
            elif dtype == "uint64":
                val = np.iinfo(np.uint64).max
            elif dtype == "bool":
                val = True
            elif dtype == "float64":
                val = np.finfo(np.float64).max
            elif dtype == "str":
                val = "max"
            a = ak.array([val])
            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                a.to_parquet(f"{tmp_dirname}/pq_test", "test-dset")
                ak_res = ak.read_parquet(f"{tmp_dirname}/pq_test*", "test-dset")["test-dset"]
                self.assertEqual(ak_res[0], val)

    def test_get_datasets(self):
        for dtype in TYPES:
            if dtype == "int64":
                ak_arr = ak.randint(0, 2**32, 10)
            elif dtype == "uint64":
                ak_arr = ak.randint(0, 2**32, 10, dtype=ak.uint64)
            elif dtype == "bool":
                ak_arr = ak.randint(0, 1, 10, dtype=ak.bool_)
            elif dtype == "float64":
                ak_arr = ak.randint(0, 2**32, SIZE, dtype=ak.float64)
            elif dtype == "str":
                ak_arr = ak.random_strings_uniform(1, 10, SIZE)
            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                ak_arr.to_parquet(f"{tmp_dirname}/pq_testdset", "TEST_DSET")

                dsets = ak.get_datasets(f"{tmp_dirname}/pq_testdset*")
                self.assertEqual(["TEST_DSET"], dsets)

    def test_append(self):
        # use small size to cut down on execution time
        append_size = 32

        base_dset = ak.randint(0, 2**32, append_size)
        ak_dict = {}
        ak_dict["uint-dset"] = ak.randint(0, 2**32, append_size, dtype=ak.uint64)
        ak_dict["bool-dset"] = ak.randint(0, 1, append_size, dtype=ak.bool_)
        ak_dict["float-dset"] = ak.randint(0, 2**32, append_size, dtype=ak.float64)
        ak_dict["int-dset"] = ak.randint(0, 2**32, append_size)
        ak_dict["str-dset"] = ak.random_strings_uniform(1, 10, append_size)

        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            base_dset.to_parquet(f"{tmp_dirname}/pq_testcorrect", "base-dset")

            for key in ak_dict:
                ak_dict[key].to_parquet(f"{tmp_dirname}/pq_testcorrect", key, mode="append")

            ak_vals = ak.read_parquet(f"{tmp_dirname}/pq_testcorrect*")

            for key in ak_dict:
                self.assertListEqual(ak_vals[key].to_list(), ak_dict[key].to_list())

    def test_null_strings(self):
        datadir = "resources/parquet-testing"
        basename = "null-strings.parquet"
        expected = ["first-string", "", "string2", "", "third", "", ""]

        filename = os.path.join(datadir, basename)
        res = ak.read_parquet(filename).popitem()[1]

        self.assertListEqual(expected, res.to_list())

    def test_null_indices(self):
        datadir = "resources/parquet-testing"
        basename = "null-strings.parquet"

        filename = os.path.join(datadir, basename)
        res = ak.get_null_indices(filename, datasets="col1")["col1"]

        self.assertListEqual([0, 1, 0, 1, 0, 1, 1], res.to_list())

    def test_append_empty(self):
        for dtype in TYPES:
            if dtype == "int64":
                ak_arr = ak.randint(0, 2**32, SIZE)
            elif dtype == "uint64":
                ak_arr = ak.randint(0, 2**32, SIZE, dtype=ak.uint64)
            elif dtype == "bool":
                ak_arr = ak.randint(0, 1, SIZE, dtype=ak.bool_)
            elif dtype == "float64":
                ak_arr = ak.randint(0, 2**32, SIZE, dtype=ak.float64)
            elif dtype == "str":
                ak_arr = ak.random_strings_uniform(1, 10, SIZE)
            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                ak_arr.to_parquet(f"{tmp_dirname}/pq_testcorrect", "my-dset", mode="append")
                pq_arr = ak.read_parquet(f"{tmp_dirname}/pq_testcorrect*", "my-dset")["my-dset"]

                self.assertListEqual(ak_arr.to_list(), pq_arr.to_list())

    def test_compression(self):
        a = ak.arange(150)

        for comp in COMPRESSIONS:
            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                # write with the selected compression
                a.to_parquet(f"{tmp_dirname}/compress_test", compression=comp)

                # ensure read functions
                rd_arr = ak.read_parquet(f"{tmp_dirname}/compress_test*", "array")["array"]

                # validate the list read out matches the array used to write
                self.assertListEqual(rd_arr.to_list(), a.to_list())

        b = ak.randint(0, 2, 150, dtype=ak.bool_)
        for comp in COMPRESSIONS:
            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                # write with the selected compression
                b.to_parquet(f"{tmp_dirname}/compress_test", compression=comp)

                # ensure read functions
                rd_arr = ak.read_parquet(f"{tmp_dirname}/compress_test*", "array")["array"]

                # validate the list read out matches the array used to write
                self.assertListEqual(rd_arr.to_list(), b.to_list())

    def test_gzip_nan_rd(self):
        # create pandas dataframe
        pdf = pd.DataFrame(
            {
                "all_nan": np.array([np.nan, np.nan, np.nan, np.nan]),
                "some_nan": np.array([3.14, np.nan, 7.12, 4.44]),
            }
        )

        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pdf.to_parquet(f"{tmp_dirname}/gzip_pq", engine="pyarrow", compression="gzip")

            ak_data = ak.read_parquet(f"{tmp_dirname}/gzip_pq")
            rd_df = ak.DataFrame(ak_data)
            self.assertTrue(pdf.equals(rd_df.to_pandas()))

    def test_segarray_read(self):
        df = pd.DataFrame(
            {
                "ListCol": [[0, 1, 2], [0, 1], [3, 4, 5, 6], [1, 2, 3]],
                "BoolList": [[True, False], [False, False, False], [True], [True, False, True]],
                "FloatList": [[3.14, 5.56, 2.23], [3.08], [6.5, 6.8], [7.0]],
                "UintList": [
                    np.array([1, 2], np.uint64),
                    np.array([3, 4, 5], np.uint64),
                    np.array([2, 2, 2], np.uint64),
                    np.array([11], np.uint64),
                ],
            }
        )
        table = pa.Table.from_pandas(df)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/segarray_parquet")

            # verify full file read with various object types
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_parquet*")
            for k, v in ak_data.items():
                self.assertIsInstance(v, ak.SegArray)
                for x, y in zip(df[k].tolist(), v.to_list()):
                    if isinstance(x, np.ndarray):
                        x = x.tolist()
                    self.assertListEqual(x, y)

            # verify individual column selection
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_parquet*", datasets="FloatList")[
                "FloatList"
            ]
            self.assertIsInstance(ak_data, ak.SegArray)
            for x, y in zip(df["FloatList"].tolist(), ak_data.to_list()):
                self.assertListEqual(x, y)

        df = pd.DataFrame(
            {"IntCol": [0, 1, 2, 3], "ListCol": [[0, 1, 2], [0, 1], [3, 4, 5, 6], [1, 2, 3]]}
        )
        table = pa.Table.from_pandas(df)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/segarray_varied_parquet")

            # read full file
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_varied_parquet*")
            for k, v in ak_data.items():
                self.assertListEqual(df[k].tolist(), v.to_list())

            # read individual datasets
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_varied_parquet*", datasets="IntCol")[
                "IntCol"
            ]
            self.assertIsInstance(ak_data, ak.pdarray)
            self.assertListEqual(df["IntCol"].to_list(), ak_data.to_list())
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_varied_parquet*", datasets="ListCol")[
                "ListCol"
            ]
            self.assertIsInstance(ak_data, ak.SegArray)
            self.assertListEqual(df["ListCol"].to_list(), ak_data.to_list())

        # test for multi-file
        df = pd.DataFrame({"ListCol": [[0, 1, 2], [0, 1], [3, 4, 5, 6], [1, 2, 3]]})
        table = pa.Table.from_pandas(df)
        df2 = pd.DataFrame({"ListCol": [[0, 1, 11], [0, 1], [3, 4, 5, 6], [1]]})
        table2 = pa.Table.from_pandas(df2)
        combo = pd.concat([df, df2], ignore_index=True)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/segarray_varied_parquet_LOCALE0000")
            pq.write_table(table2, f"{tmp_dirname}/segarray_varied_parquet_LOCALE0001")
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_varied_parquet*")["ListCol"]
            self.assertIsInstance(ak_data, ak.SegArray)
            self.assertEqual(ak_data.size, 8)
            for i in range(8):
                self.assertListEqual(combo["ListCol"][i], ak_data[i].to_list())

        # test for handling empty segments
        df = pd.DataFrame({"ListCol": [[], [0, 1], [], [3, 4, 5, 6], []]})
        table = pa.Table.from_pandas(df)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/empty_segments")

            ak_data = ak.read_parquet(f"{tmp_dirname}/empty_segments*")["ListCol"]
            self.assertIsInstance(ak_data, ak.SegArray)
            self.assertEqual(ak_data.size, 5)
            for i in range(5):
                self.assertListEqual(df["ListCol"][i], ak_data[i].to_list())

        # test for handling empty segments
        df = pd.DataFrame({"ListCol": [[8], [0, 1], [], [3, 4, 5, 6], []]})
        table = pa.Table.from_pandas(df)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/empty_segments")

            ak_data = ak.read_parquet(f"{tmp_dirname}/empty_segments*")["ListCol"]
            self.assertIsInstance(ak_data, ak.SegArray)
            self.assertEqual(ak_data.size, 5)
            for i in range(5):
                self.assertListEqual(df["ListCol"][i], ak_data[i].to_list())

        # multi-file with empty segs
        df = pd.DataFrame({"ListCol": [[], [0, 1], [], [3, 4, 5, 6], []]})
        df2 = pd.DataFrame({"ListCol": [[0, 1], [], [3, 4, 5, 6], [1, 2, 3]]})
        table = pa.Table.from_pandas(df)
        table2 = pa.Table.from_pandas(df2)
        combo = pd.concat([df, df2], ignore_index=True)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/empty_segments_LOCALE0000")
            pq.write_table(table2, f"{tmp_dirname}/empty_segments_LOCALE0001")

            ak_data = ak.read_parquet(f"{tmp_dirname}/empty_segments*")["ListCol"]
            self.assertIsInstance(ak_data, ak.SegArray)
            self.assertEqual(ak_data.size, 9)
            for i in range(9):
                self.assertListEqual(combo["ListCol"][i], ak_data[i].to_list())

        # multi-file with empty segs
        df = pd.DataFrame({"ListCol": [[8], [0, 1], [], [3, 4, 5, 6], []]})
        df2 = pd.DataFrame({"ListCol": [[0, 1], [], [3, 4, 5, 6], [1, 2, 3]]})
        table = pa.Table.from_pandas(df)
        table2 = pa.Table.from_pandas(df2)
        combo = pd.concat([df, df2], ignore_index=True)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/empty_segments_LOCALE0000")
            pq.write_table(table2, f"{tmp_dirname}/empty_segments_LOCALE0001")

            ak_data = ak.read_parquet(f"{tmp_dirname}/empty_segments*")["ListCol"]
            self.assertIsInstance(ak_data, ak.SegArray)
            self.assertEqual(ak_data.size, 9)
            for i in range(9):
                self.assertListEqual(combo["ListCol"][i], ak_data[i].to_list())

    def test_segarray_write(self):
        # integer test
        a = [0, 1, 2]
        b = [1]
        c = [15, 21]
        s = ak.SegArray(ak.array([0, len(a), len(a) + len(b)]), ak.array(a + b + c))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/int_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/int_test*").popitem()[1]
            for i in range(3):
                self.assertListEqual(s[i].to_list(), rd_data[i].to_list())

        # integer with empty segments
        a = [0, 1, 2]
        c = [15, 21]
        s = ak.SegArray(ak.array([0, 0, len(a), len(a), len(a), len(a) + len(c)]), ak.array(a + c))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/int_test_empty")

            rd_data = ak.read_parquet(f"{tmp_dirname}/int_test_empty*").popitem()[1]
            for i in range(6):
                self.assertListEqual(s[i].to_list(), rd_data[i].to_list())

        # uint test
        a = [0, 1, 2]
        b = [1]
        c = [15, 21]
        s = ak.SegArray(ak.array([0, len(a), len(a) + len(b)]), ak.array(a + b + c, dtype=ak.uint64))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/uint_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/uint_test*").popitem()[1]
            for i in range(3):
                self.assertListEqual(s[i].to_list(), rd_data[i].to_list())

        # uint with empty segments
        a = [0, 1, 2]
        c = [15, 21]
        s = ak.SegArray(
            ak.array([0, 0, len(a), len(a), len(a), len(a) + len(c)]), ak.array(a + c, dtype=ak.uint64)
        )
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/uint_test_empty")

            rd_data = ak.read_parquet(f"{tmp_dirname}/uint_test_empty*").popitem()[1]
            for i in range(6):
                self.assertListEqual(s[i].to_list(), rd_data[i].to_list())

        # bool test
        a = [0, 1, 1]
        b = [0]
        c = [1, 0]
        s = ak.SegArray(ak.array([0, len(a), len(a) + len(b)]), ak.array(a + b + c, dtype=ak.bool_))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/bool_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/bool_test*").popitem()[1]
            for i in range(3):
                self.assertListEqual(s[i].to_list(), rd_data[i].to_list())

        # bool with empty segments
        a = [0, 1, 1]
        c = [1, 0]
        s = ak.SegArray(
            ak.array([0, 0, len(a), len(a), len(a), len(a) + len(c)]), ak.array(a + c, dtype=ak.bool_)
        )
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/bool_test_empty")

            rd_data = ak.read_parquet(f"{tmp_dirname}/bool_test_empty*").popitem()[1]
            for i in range(6):
                self.assertListEqual(s[i].to_list(), rd_data[i].to_list())

        # float test
        a = [1.1, 1.1, 2.7]
        b = [1.99]
        c = [15.2, 21.0]
        s = ak.SegArray(ak.array([0, len(a), len(a) + len(b)]), ak.array(a + b + c))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/float_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/float_test*").popitem()[1]
            for i in range(3):
                self.assertListEqual(s[i].to_list(), rd_data[i].to_list())

        # float with empty segments
        a = [1.1, 1.1, 2.7]
        c = [15.2, 21.0]
        s = ak.SegArray(ak.array([0, 0, len(a), len(a), len(a), len(a) + len(c)]), ak.array(a + c))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/float_test_empty")

            rd_data = ak.read_parquet(f"{tmp_dirname}/float_test_empty*").popitem()[1]
            for i in range(6):
                self.assertListEqual(s[i].to_list(), rd_data[i].to_list())

    def test_multicol_write(self):
        df_dict = {
            "c_1": ak.arange(3),
            "c_2": ak.SegArray(ak.array([0, 9, 14]), ak.arange(20)),
            "c_3": ak.arange(3, 6, dtype=ak.uint64),
            "c_4": ak.SegArray(ak.array([0, 5, 10]), ak.arange(15, dtype=ak.uint64)),
            "c_5": ak.array([False, True, False]),
            "c_6": ak.SegArray(ak.array([0, 5, 10]), ak.randint(0, 1, 15, dtype=ak.bool_)),
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
        akdf = ak.DataFrame(df_dict)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            # use multicolumn write to generate parquet file
            akdf.to_parquet(f"{tmp_dirname}/multicol_parquet")
            # read files and ensure that all resulting fields are as expected
            rd_data = ak.read_parquet(f"{tmp_dirname}/multicol_parquet*")
            for k, v in rd_data.items():
                self.assertListEqual(v.to_list(), akdf[k].to_list())

            # extra insurance, check dataframes are equivalent
            rd_df = ak.DataFrame(rd_data)
            self.assertTrue(akdf.to_pandas().equals(rd_df.to_pandas()))

    def test_small_ints(self):
        df_pd = pd.DataFrame(
            {
                "int16": pd.Series([2**15 - 1, -(2**15)], dtype=np.int16),
                "int32": pd.Series([2**31 - 1, -(2**31)], dtype=np.int32),
                "uint16": pd.Series([2**15 - 1, 2**15], dtype=np.uint16),
                "uint32": pd.Series([2**31 - 1, 2**31], dtype=np.uint32),
            }
        )
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            fname = tmp_dirname + "/pq_small_int"
            df_pd.to_parquet(fname)
            df_ak = ak.DataFrame(ak.read(fname + "*"))
            for c in df_ak.columns:
                self.assertListEqual(df_ak[c].to_list(), df_pd[c].to_list())

    def test_read_nested(self):
        df = ak.DataFrame({"idx": ak.arange(5), "seg": ak.SegArray(ak.arange(0, 10, 2), ak.arange(10))})
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            fname = tmp_dirname + "/read_nested_test"
            df.to_parquet(fname)

            # test read with read_nested=true
            data = ak.read_parquet(fname + "_*")
            self.assertTrue("idx" in data)
            self.assertTrue("seg" in data)
            self.assertListEqual(df["idx"].to_list(), data["idx"].to_list())
            self.assertListEqual(df["seg"].to_list(), data["seg"].to_list())

            # test read with read_nested=false and no supplied datasets
            data = ak.read_parquet(fname + "_*", read_nested=False).popitem()[1]
            self.assertIsInstance(data, ak.pdarray)
            self.assertListEqual(df["idx"].to_list(), data.to_list())

            # test read with read_nested=false and user supplied datasets. Should ignore read_nested
            data = ak.read_parquet(fname + "_*", datasets=["idx", "seg"], read_nested=False)
            self.assertTrue("idx" in data)
            self.assertTrue("seg" in data)
            self.assertListEqual(df["idx"].to_list(), data["idx"].to_list())
            self.assertListEqual(df["seg"].to_list(), data["seg"].to_list())

    def test_segarray_string(self):
        words = ak.array(["one,two,three", "uno,dos,tres"])
        strs, segs = words.split(",", return_segments=True)
        x = ak.SegArray(segs, strs)

        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            x.to_parquet(f"{tmp_dirname}/segarr_str")

            rd = ak.read_parquet(f"{tmp_dirname}/segarr_str_*").popitem()[1]
            self.assertIsInstance(rd, ak.SegArray)
            self.assertListEqual(x.segments.to_list(), rd.segments.to_list())
            self.assertListEqual(x.values.to_list(), rd.values.to_list())
            self.assertListEqual(x.to_list(), rd.to_list())

        # additional testing for empty segments. See Issue #2560
        a, b, c = ["one", "two", "three"], ["un", "deux", "trois"], ["uno", "dos", "tres"]
        s = ak.SegArray(ak.array([0, 0, len(a), len(a), len(a), len(a) + len(c)]), ak.array(a + c))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/segarray_test_empty")
            rd_data = ak.read_parquet(f"{tmp_dirname}/segarray_test_empty_*").popitem()[1]
            self.assertListEqual(s.to_list(), rd_data.to_list())

    def test_float_edge(self):
        df = pd.DataFrame(
            {"FloatList": [[3.14, np.nan, 2.23], [], [3.08], [np.inf, 6.8], [-0.0, np.nan, np.nan]]}
        )

        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, f"{tmp_dirname}/segarray_float_edge")
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_float_edge")
            pd_l = df["FloatList"].tolist()
            ak_l = ak_data["FloatList"].to_list()
            for i in range(len(pd_l)):
                self.assertTrue(np.allclose(pd_l[i], ak_l[i], equal_nan=True))

    def test_decimal_reads(self):
        cols = []
        data = []
        for i in range(1, 39):
            cols.append(("decCol" + str(i), pa.decimal128(i, 0)))
            data.append([i])

        schema = pa.schema(cols)

        table = pa.Table.from_arrays(data, schema=schema)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/decimal")
            ak_data = ak.read(f"{tmp_dirname}/decimal")
            for i in range(1, 39):
                self.assertTrue(np.allclose(ak_data["decCol" + str(i)].to_ndarray(), data[i - 1]))

    def test_multi_batch_reads(self):
        pytest.skip()
        # verify reproducer for #3074 is resolved
        # seagarray w/ empty segs multi-batch pq reads

        # bug seemed to consistently appear for val_sizes
        # exceeding 700000 (likely due to this requiring more than one batch)
        # we round up to ensure we'd hit it
        val_size = 1000000

        df_dict = dict()
        seed = np.random.default_rng().choice(2**63)
        rng = ak.random.default_rng(seed)
        some_nans = rng.uniform(-(2**10), 2**10, val_size)
        some_nans[ak.arange(val_size) % 2 == 0] = np.nan
        vals_list = [
            rng.uniform(-(2**10), 2**10, val_size),
            rng.integers(0, 2**32, size=val_size, dtype="uint"),
            rng.integers(0, 1, size=val_size, dtype="bool"),
            rng.integers(-(2**32), 2**32, size=val_size, dtype="int"),
            some_nans,  # contains nans
            ak.random_strings_uniform(0, 4, val_size, seed=seed),  # contains empty strings
        ]

        for vals in vals_list:
            # segs must start with 0, all other segment lengths are random
            # by having val_size number of segments, except in the extremely unlikely case of
            # randomly getting exactly arange(val_size), we are guaranteed empty segs
            segs = ak.concatenate(
                [ak.array([0]), ak.sort(ak.randint(0, val_size, val_size - 1, seed=seed))]
            )
            df_dict["rand"] = ak.SegArray(segs, vals).to_list()

            pddf = pd.DataFrame(df_dict)
            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                file_path = f"{tmp_dirname}/empty_segs"
                pddf.to_parquet(file_path)
                akdf = ak.DataFrame(ak.read_parquet(file_path))

                to_pd = pd.Series(akdf["rand"].to_list())
                # raises an error if the two series aren't equal
                # we can't use np.allclose(pddf['rand'].to_list, akdf['rand'].to_list) since these
                # are lists of lists. assert_series_equal handles this and properly handles nans.
                # we pass the same absolute and relative tolerances as the numpy default in allclose
                # to ensure float point differences don't cause errors
                print("\nseed: ", seed)
                assert_series_equal(pddf["rand"], to_pd, check_names=False, rtol=1e-05, atol=1e-08)

                # test writing multi-batch non-segarrays
                file_path = f"{tmp_dirname}/multi_batch_vals"
                vals.to_parquet(file_path, dataset="my_vals")
                read = ak.read_parquet(file_path + "*")["my_vals"]
                if isinstance(vals, ak.pdarray) and vals.dtype == ak.float64:
                    assert np.allclose(read.to_list(), vals.to_list(), equal_nan=True)
                else:
                    assert (read == vals).all()

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
            self.assertListEqual(columns, ans)
            # Merely test that read succeeds, do not check output
            if "delta_byte_array.parquet" not in filename:
                data = ak.read_parquet(filename, datasets=columns)
            else:
                # Since delta encoding is not supported, the columns in
                # this file should raise an error and not crash the server
                with self.assertRaises(RuntimeError):
                    data = ak.read_parquet(filename, datasets=columns)
