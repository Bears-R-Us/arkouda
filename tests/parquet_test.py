import glob
import os
import tempfile

import numpy as np
import pytest
from base_test import ArkoudaTest
from context import arkouda as ak
from arkouda import io_util

TYPES = ("int64", "uint64", "bool", "float64", "str")
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
