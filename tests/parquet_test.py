import glob
import os
import pandas as pd
import tempfile

import numpy as np
import pytest
from base_test import ArkoudaTest
from context import arkouda as ak
from arkouda import io_util
import pyarrow as pa
import pyarrow.parquet as pq

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
                ak_arr = ak.randint(0, 2 ** 32, SIZE)
            elif dtype == "uint64":
                ak_arr = ak.randint(0, 2 ** 32, SIZE, dtype=ak.uint64)
            elif dtype == "bool":
                ak_arr = ak.randint(0, 1, SIZE, dtype=ak.bool)
            elif dtype == "float64":
                ak_arr = ak.randint(0, 2 ** 32, SIZE, dtype=ak.float64)
            elif dtype == "str":
                ak_arr = ak.random_strings_uniform(1, 10, SIZE)

            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                ak_arr.to_parquet(f"{tmp_dirname}/pq_testcorrect", "my-dset")
                pq_arr = ak.read_parquet(f"{tmp_dirname}/pq_testcorrect*", "my-dset")
                self.assertListEqual(ak_arr.to_list(), pq_arr.to_list())

            # verify generic read
            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                ak_arr.to_parquet(f"{tmp_dirname}/pq_testcorrect", "my-dset")
                pq_arr = ak.read(f"{tmp_dirname}/pq_testcorrect*", "my-dset")
                self.assertListEqual(ak_arr.to_list(), pq_arr.to_list())

    def test_multi_file(self):
        for dtype in TYPES:
            adjusted_size = int(SIZE / NUMFILES) * NUMFILES
            test_arrs = []
            if dtype == "int64":
                elems = ak.randint(0, 2 ** 32, adjusted_size)
            elif dtype == "uint64":
                elems = ak.randint(0, 2 ** 32, adjusted_size, dtype=ak.uint64)
            elif dtype == "bool":
                elems = ak.randint(0, 1, adjusted_size, dtype=ak.bool)
            elif dtype == "float64":
                elems = ak.randint(0, 2 ** 32, adjusted_size, dtype=ak.float64)
            elif dtype == "str":
                elems = ak.random_strings_uniform(1, 10, adjusted_size)

            per_arr = int(adjusted_size / NUMFILES)
            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                for i in range(NUMFILES):
                    test_arrs.append(elems[(i * per_arr): (i * per_arr) + per_arr])
                    test_arrs[i].to_parquet(f"{tmp_dirname}/pq_test{i:04d}", "test-dset")

                pq_arr = ak.read_parquet(f"{tmp_dirname}/pq_test*", "test-dset")
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
                ak_res = ak.read_parquet(f"{tmp_dirname}/pq_test*", "test-dset")
                self.assertEqual(ak_res[0], val)

    def test_get_datasets(self):
        for dtype in TYPES:
            if dtype == "int64":
                ak_arr = ak.randint(0, 2 ** 32, 10)
            elif dtype == "uint64":
                ak_arr = ak.randint(0, 2 ** 32, 10, dtype=ak.uint64)
            elif dtype == "bool":
                ak_arr = ak.randint(0, 1, 10, dtype=ak.bool)
            elif dtype == "float64":
                ak_arr = ak.randint(0, 2 ** 32, SIZE, dtype=ak.float64)
            elif dtype == "str":
                ak_arr = ak.random_strings_uniform(1, 10, SIZE)
            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                ak_arr.to_parquet(f"{tmp_dirname}/pq_testdset", "TEST_DSET")

                dsets = ak.get_datasets(f"{tmp_dirname}/pq_testdset*")
                self.assertEqual(["TEST_DSET"], dsets)

    def test_append(self):
        # use small size to cut down on execution time
        append_size = 32

        base_dset = ak.randint(0, 2 ** 32, append_size)
        ak_dict = {}
        ak_dict["uint-dset"] = ak.randint(0, 2 ** 32, append_size, dtype=ak.uint64)
        ak_dict["bool-dset"] = ak.randint(0, 1, append_size, dtype=ak.bool)
        ak_dict["float-dset"] = ak.randint(0, 2 ** 32, append_size, dtype=ak.float64)
        ak_dict["int-dset"] = ak.randint(0, 2 ** 32, append_size)
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
        res = ak.read_parquet(filename)

        self.assertListEqual(expected, res.to_list())

    def test_null_indices(self):
        datadir = "resources/parquet-testing"
        basename = "null-strings.parquet"

        filename = os.path.join(datadir, basename)
        res = ak.get_null_indices(filename, datasets="col1")

        self.assertListEqual([0, 1, 0, 1, 0, 1, 1], res.to_list())

    def test_append_empty(self):
        for dtype in TYPES:
            if dtype == "int64":
                ak_arr = ak.randint(0, 2 ** 32, SIZE)
            elif dtype == "uint64":
                ak_arr = ak.randint(0, 2 ** 32, SIZE, dtype=ak.uint64)
            elif dtype == "bool":
                ak_arr = ak.randint(0, 1, SIZE, dtype=ak.bool)
            elif dtype == "float64":
                ak_arr = ak.randint(0, 2 ** 32, SIZE, dtype=ak.float64)
            elif dtype == "str":
                ak_arr = ak.random_strings_uniform(1, 10, SIZE)
            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                ak_arr.to_parquet(f"{tmp_dirname}/pq_testcorrect", "my-dset", mode="append")
                pq_arr = ak.read_parquet(f"{tmp_dirname}/pq_testcorrect*", "my-dset")

                self.assertListEqual(ak_arr.to_list(), pq_arr.to_list())

    def test_compression(self):
        a = ak.arange(150)

        for comp in COMPRESSIONS:
            with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
                # write with the selected compression
                a.to_parquet(f"{tmp_dirname}/compress_test", compression=comp)

                # ensure read functions
                rd_arr = ak.read_parquet(f"{tmp_dirname}/compress_test*", "array")

                # validate the list read out matches the array used to write
                self.assertListEqual(rd_arr.to_list(), a.to_list())

    def test_gzip_nan_rd(self):
        # create pandas dataframe
        pdf = pd.DataFrame({
            'all_nan': np.array([np.nan, np.nan, np.nan, np.nan]),
            'some_nan': np.array([3.14, np.nan, 7.12, 4.44])
        })

        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pdf.to_parquet(f"{tmp_dirname}/gzip_pq", engine="pyarrow", compression="gzip")

            ak_data = ak.read_parquet(f"{tmp_dirname}/gzip_pq")
            rd_df = ak.DataFrame(ak_data)
            self.assertTrue(pdf.equals(rd_df.to_pandas()))

    def test_segarray_read(self):
        df = pd.DataFrame({
            "ListCol": [
                [0, 1, 2],
                [0, 1],
                [3, 4, 5, 6],
                [1, 2, 3]
            ],
            "BoolList": [
                [True, False],
                [False, False, False],
                [True],
                [True, False, True]
            ],
            "FloatList": [
                [3.14, 5.56, 2.23],
                [3.08],
                [6.5, 6.8],
                [7.0]
            ],
            "UintList": [
                np.array([1, 2], np.uint64),
                np.array([3, 4, 5], np.uint64),
                np.array([2, 2, 2], np.uint64),
                np.array([11], np.uint64)
            ]
        })
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
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_parquet*", datasets="FloatList")
            self.assertIsInstance(ak_data, ak.SegArray)
            for x, y in zip(df["FloatList"].tolist(), ak_data.to_list()):
                self.assertListEqual(x,y)

        df = pd.DataFrame({
            "IntCol": [0, 1, 2, 3],
            "ListCol": [
                [0, 1, 2],
                [0, 1],
                [3, 4, 5, 6],
                [1, 2, 3]
            ]
        })
        table = pa.Table.from_pandas(df)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/segarray_varied_parquet")

            # read full file
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_varied_parquet*")
            for k, v in ak_data.items():
                self.assertListEqual(df[k].tolist(), v.to_list())

            # read individual datasets
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_varied_parquet*", datasets="IntCol")
            self.assertIsInstance(ak_data, ak.pdarray)
            self.assertListEqual(df["IntCol"].to_list(), ak_data.to_list())
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_varied_parquet*", datasets="ListCol")
            self.assertIsInstance(ak_data, ak.SegArray)
            self.assertListEqual(df["ListCol"].to_list(), ak_data.to_list())

        # test for multi-file
        df = pd.DataFrame({
            "ListCol": [
                [0, 1, 2],
                [0, 1],
                [3, 4, 5, 6],
                [1, 2, 3]
            ]
        })
        table = pa.Table.from_pandas(df)
        df2 = pd.DataFrame({
            "ListCol": [
                [0, 1, 11],
                [0, 1],
                [3, 4, 5, 6],
                [1]
            ]
        })
        table2 = pa.Table.from_pandas(df2)
        combo = pd.concat([df, df2], ignore_index=True)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/segarray_varied_parquet_LOCALE0000")
            pq.write_table(table2, f"{tmp_dirname}/segarray_varied_parquet_LOCALE0001")
            ak_data = ak.read_parquet(f"{tmp_dirname}/segarray_varied_parquet*")
            self.assertIsInstance(ak_data, ak.SegArray)
            self.assertEqual(ak_data.size, 8)
            for i in range(8):
                self.assertListEqual(combo["ListCol"][i], ak_data[i].tolist())

        #test for handling empty segments
        df = pd.DataFrame({
            "ListCol": [
                [],
                [0, 1],
                [],
                [3, 4, 5, 6],
                []
            ]
        })
        table = pa.Table.from_pandas(df)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/empty_segments")

            ak_data = ak.read_parquet(f"{tmp_dirname}/empty_segments*")
            self.assertIsInstance(ak_data, ak.SegArray)
            self.assertEqual(ak_data.size, 5)
            for i in range(5):
                self.assertListEqual(df["ListCol"][i], ak_data[i].tolist())

        # test for handling empty segments
        df = pd.DataFrame({
            "ListCol": [
                [8],
                [0, 1],
                [],
                [3, 4, 5, 6],
                []
            ]
        })
        table = pa.Table.from_pandas(df)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/empty_segments")

            ak_data = ak.read_parquet(f"{tmp_dirname}/empty_segments*")
            self.assertIsInstance(ak_data, ak.SegArray)
            self.assertEqual(ak_data.size, 5)
            for i in range(5):
                self.assertListEqual(df["ListCol"][i], ak_data[i].tolist())

        # multi-file with empty segs
        df = pd.DataFrame({
            "ListCol": [
                [],
                [0, 1],
                [],
                [3, 4, 5, 6],
                []
            ]
        })
        df2 = pd.DataFrame({
            "ListCol": [
                [0, 1],
                [],
                [3, 4, 5, 6],
                [1, 2, 3]
            ]
        })
        table = pa.Table.from_pandas(df)
        table2 = pa.Table.from_pandas(df2)
        combo = pd.concat([df, df2], ignore_index=True)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/empty_segments_LOCALE0000")
            pq.write_table(table2, f"{tmp_dirname}/empty_segments_LOCALE0001")

            ak_data = ak.read_parquet(f"{tmp_dirname}/empty_segments*")
            self.assertIsInstance(ak_data, ak.SegArray)
            self.assertEqual(ak_data.size, 9)
            for i in range(9):
                self.assertListEqual(combo["ListCol"][i], ak_data[i].tolist())

        # multi-file with empty segs
        df = pd.DataFrame({
            "ListCol": [
                [8],
                [0, 1],
                [],
                [3, 4, 5, 6],
                []
            ]
        })
        df2 = pd.DataFrame({
            "ListCol": [
                [0, 1],
                [],
                [3, 4, 5, 6],
                [1, 2, 3]
            ]
        })
        table = pa.Table.from_pandas(df)
        table2 = pa.Table.from_pandas(df2)
        combo = pd.concat([df, df2], ignore_index=True)
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            pq.write_table(table, f"{tmp_dirname}/empty_segments_LOCALE0000")
            pq.write_table(table2, f"{tmp_dirname}/empty_segments_LOCALE0001")

            ak_data = ak.read_parquet(f"{tmp_dirname}/empty_segments*")
            self.assertIsInstance(ak_data, ak.SegArray)
            self.assertEqual(ak_data.size, 9)
            for i in range(9):
                self.assertListEqual(combo["ListCol"][i], ak_data[i].tolist())

    def test_segarray_write(self):
        # integer test
        a = [0, 1, 2]
        b = [1]
        c = [15, 21]
        s = ak.SegArray(ak.array([0, len(a), len(a) + len(b)]), ak.array(a + b + c))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/int_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/int_test*")
            for i in range(3):
                self.assertListEqual(s[i].tolist(), rd_data[i].tolist())

        # integer with empty segments
        a = [0, 1, 2]
        c = [15, 21]
        s = ak.SegArray(ak.array([0, 0, len(a), len(a), len(a), len(a)+len(c)]), ak.array(a + c))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/int_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/int_test*")
            for i in range(6):
                self.assertListEqual(s[i].tolist(), rd_data[i].tolist())

        # uint test
        a = [0, 1, 2]
        b = [1]
        c = [15, 21]
        s = ak.SegArray(ak.array([0, len(a), len(a) + len(b)]), ak.array(a + b + c, dtype=ak.uint64))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/int_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/int_test*")
            for i in range(3):
                self.assertListEqual(s[i].tolist(), rd_data[i].tolist())

        # uint with empty segments
        a = [0, 1, 2]
        c = [15, 21]
        s = ak.SegArray(ak.array([0, 0, len(a), len(a), len(a), len(a) + len(c)]), ak.array(a + c, dtype=ak.uint64))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/int_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/int_test*")
            for i in range(6):
                self.assertListEqual(s[i].tolist(), rd_data[i].tolist())

        # bool test
        a = [0, 1, 1]
        b = [0]
        c = [1, 0]
        s = ak.SegArray(ak.array([0, len(a), len(a) + len(b)]), ak.array(a + b + c, dtype=ak.bool))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/int_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/int_test*")
            for i in range(3):
                self.assertListEqual(s[i].tolist(), rd_data[i].tolist())

        # bool with empty segments
        a = [0, 1, 1]
        c = [1, 0]
        s = ak.SegArray(ak.array([0, 0, len(a), len(a), len(a), len(a) + len(c)]), ak.array(a + c,
                                   dtype=ak.bool))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/int_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/int_test*")
            for i in range(6):
                self.assertListEqual(s[i].tolist(), rd_data[i].tolist())

        # float test
        a = [1.1, 1.1, 2.7]
        b = [1.99]
        c = [15.2, 21.0]
        s = ak.SegArray(ak.array([0, len(a), len(a) + len(b)]), ak.array(a + b + c))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/int_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/int_test*")
            for i in range(3):
                self.assertListEqual(s[i].tolist(), rd_data[i].tolist())

        # float with empty segments
        a = [1.1, 1.1, 2.7]
        c = [15.2, 21.0]
        s = ak.SegArray(ak.array([0, 0, len(a), len(a), len(a), len(a) + len(c)]), ak.array(a + c))
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            s.to_parquet(f"{tmp_dirname}/int_test")

            rd_data = ak.read_parquet(f"{tmp_dirname}/int_test*")
            for i in range(6):
                self.assertListEqual(s[i].tolist(), rd_data[i].tolist())

    def test_multicol_write(self):
        df_dict = {
            "c_1": ak.arange(3),
            "c_2": ak.SegArray(ak.array([0, 9, 14]), ak.arange(20)),
            "c_3": ak.arange(3, 6, dtype=ak.uint64),
            "c_4": ak.SegArray(ak.array([0, 5, 10]), ak.arange(15, dtype=ak.uint64)),
            "c_5": ak.array([False, True, False]),
            "c_6": ak.SegArray(ak.array([0, 5, 10]), ak.randint(0, 1, 15, dtype=ak.bool)),
            "c_7": ak.array(np.random.uniform(0, 100, 3)),
            "c_8": ak.SegArray(ak.array([0, 9, 14]), ak.array(np.random.uniform(0, 100, 20))),
            "c_9": ak.array(["abc", "123", "xyz"])
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
        df_pd = pd.DataFrame({
            "int16": pd.Series([2 ** 15 - 1, -2 ** 15], dtype=np.int16),
            "int32": pd.Series([2 ** 31 - 1, -2 ** 31], dtype=np.int32),
            "uint16": pd.Series([2 ** 15 - 1, 2 ** 15], dtype=np.uint16),
            "uint32": pd.Series([2 ** 31 - 1, 2 ** 31], dtype=np.uint32),
        })
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            fname = tmp_dirname+"/pq_small_int"
            df_pd.to_parquet(fname)
            df_ak = ak.DataFrame(ak.read(fname + "*"))
            for c in df_ak.columns:
                self.assertListEqual(df_ak[c].to_list(), df_pd[c].to_list())

    def test_read_nested(self):
        df = ak.DataFrame({
            "idx": ak.arange(5),
            "seg": ak.segarray(ak.arange(0, 10, 2), ak.arange(10))
        })
        with tempfile.TemporaryDirectory(dir=ParquetTest.par_test_base_tmp) as tmp_dirname:
            fname = tmp_dirname+"/read_nested_test"
            df.to_parquet(fname)

            # test read with read_nested=true
            data = ak.read_parquet(fname+"_*")
            self.assertTrue("idx" in data)
            self.assertTrue("seg" in data)
            self.assertListEqual(df["idx"].to_list(), data["idx"].to_list())
            self.assertListEqual(df["seg"].to_list(), data["seg"].to_list())

            # test read with read_nested=false and no supplied datasets
            data = ak.read_parquet(fname + "_*", read_nested=False)
            self.assertIsInstance(data, ak.pdarray)
            self.assertListEqual(df["idx"].to_list(), data.to_list())

            # test read with read_nested=false and user supplied datasets. Should ignore read_nested
            data = ak.read_parquet(fname + "_*", datasets=["idx", "seg"], read_nested=False)
            self.assertTrue("idx" in data)
            self.assertTrue("seg" in data)
            self.assertListEqual(df["idx"].to_list(), data["idx"].to_list())
            self.assertListEqual(df["seg"].to_list(), data["seg"].to_list())

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
