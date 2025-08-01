import glob
import os
import shutil
import tempfile
from typing import List, Mapping, Union

import h5py
import numpy as np
import pytest
from base_test import ArkoudaTest
from context import arkouda as ak

import arkouda.array_api as Array
from arkouda import read_zarr, to_zarr
from arkouda.pandas import io_util

"""
Tests writing Arkouda pdarrays to and reading from files
"""


class IOTest(ArkoudaTest):
    @classmethod
    def setUpClass(cls):
        super(IOTest, cls).setUpClass()
        IOTest.io_test_dir = "{}/io_test".format(os.getcwd())
        io_util.get_directory(IOTest.io_test_dir)

    def setUp(self):
        ArkoudaTest.setUp(self)
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

        self.names = ["int_tens_pdarray", "int_hundreds_pdarray", "float_pdarray", "bool_pdarray"]

        with open("{}/not-a-file_LOCALE0000".format(IOTest.io_test_dir), "w"):
            pass

    def _create_file(
        self, prefix_path: str, columns: Union[Mapping[str, ak.array]], names: List[str] = None
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

    def testSaveAllLoadAllWithDict(self):
        """
        Creates 2..n files from an input columns dict depending upon the number of
        arkouda_server locales, retrieves all datasets and correspoding pdarrays,
        and confirms they match inputs

        :return: None
        :raise: AssertionError if the input and returned datasets and pdarrays don't match
        """
        self._create_file(
            columns=self.dict_columns, prefix_path="{}/iotest_dict".format(IOTest.io_test_dir)
        )
        retrieved_columns = ak.load_all("{}/iotest_dict".format(IOTest.io_test_dir))

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

        self.assertEqual(4, len(retrieved_columns))
        self.assertListEqual(itp.tolist(), ritp.tolist())
        self.assertListEqual(ihp.tolist(), rihp.tolist())
        self.assertListEqual(ifp.tolist(), rifp.tolist())
        self.assertEqual(len(self.dict_columns["bool_pdarray"]), len(retrieved_columns["bool_pdarray"]))
        self.assertEqual(4, len(ak.get_datasets("{}/iotest_dict_LOCALE0000".format(IOTest.io_test_dir))))

    def testSaveAllLoadAllWithList(self):
        """
        Creates 2..n files from an input columns and names list depending upon the number of
        arkouda_server locales, retrieves all datasets and correspoding pdarrays, and confirms
        they match inputs

        :return: None
        :raise: AssertionError if the input and returned datasets and pdarrays don't match
        """
        self._create_file(
            columns=self.list_columns,
            prefix_path="{}/iotest_list".format(IOTest.io_test_dir),
            names=self.names,
        )
        retrieved_columns = ak.load_all(path_prefix="{}/iotest_list".format(IOTest.io_test_dir))

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

        self.assertEqual(4, len(retrieved_columns))
        self.assertListEqual(itp.tolist(), ritp.tolist())
        self.assertListEqual(ihp.tolist(), rihp.tolist())
        self.assertListEqual(fp.tolist(), rfp.tolist())
        self.assertEqual(len(self.list_columns[3]), len(retrieved_columns["bool_pdarray"]))
        self.assertEqual(4, len(ak.get_datasets("{}/iotest_list_LOCALE0000".format(IOTest.io_test_dir))))

    def testLsHdf(self):
        """
        Creates 1..n files depending upon the number of arkouda_server locales, invokes the
        ls method on an explicit file name reads the files and confirms the expected
        message was returned.

        :return: None
        :raise: AssertionError if the h5ls output does not match expected value
        """
        self._create_file(
            columns=self.dict_single_column,
            prefix_path="{}/iotest_single_column".format(IOTest.io_test_dir),
        )
        message = ak.ls("{}/iotest_single_column_LOCALE0000".format(IOTest.io_test_dir))
        self.assertIn("int_tens_pdarray", message)

        with self.assertRaises(RuntimeError):
            ak.ls("{}/not-a-file_LOCALE0000".format(IOTest.io_test_dir))

    def testLsHdfEmpty(self):
        # Test filename empty/whitespace-only condition
        with self.assertRaises(ValueError):
            ak.ls("")

        with self.assertRaises(ValueError):
            ak.ls("   ")

        with self.assertRaises(ValueError):
            ak.ls(" \n\r\t  ")

    def testReadHdf(self):
        """
        Creates 2..n files depending upon the number of arkouda_server locales, reads the files
        with an explicit list of file names to the read_all method, and confirms the datasets
        and embedded pdarrays match the input dataset and pdarrays

        :return: None
        :raise: AssertionError if the input and returned datasets don't match
        """
        self._create_file(
            columns=self.dict_columns, prefix_path="{}/iotest_dict_columns".format(IOTest.io_test_dir)
        )

        # test with read_hdf
        dataset = ak.read_hdf(filenames=["{}/iotest_dict_columns_LOCALE0000".format(IOTest.io_test_dir)])
        self.assertEqual(4, len(list(dataset.keys())))

        # test with generic read function
        dataset = ak.read(filenames=["{}/iotest_dict_columns_LOCALE0000".format(IOTest.io_test_dir)])
        self.assertEqual(4, len(list(dataset.keys())))

    def testReadHdfWithGlob(self):
        """
        Creates 2..n files depending upon the number of arkouda_server locales with two
        files each containing different-named datasets with the same pdarrays, reads the files
        with the glob feature of the read_all method, and confirms the datasets and embedded
        pdarrays match the input dataset and pdarrays

        :return: None
        :raise: AssertionError if the input and returned datasets don't match
        """
        self._create_file(
            columns=self.dict_columns, prefix_path="{}/iotest_dict_columns".format(IOTest.io_test_dir)
        )

        retrieved_columns = ak.read_hdf(filenames="{}/iotest_dict_columns*".format(IOTest.io_test_dir))

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

        self.assertEqual(4, len(list(retrieved_columns.keys())))
        self.assertListEqual(itp.tolist(), ritp.tolist())
        self.assertListEqual(ihp.tolist(), rihp.tolist())
        self.assertListEqual(fp.tolist(), rfp.tolist())
        self.assertEqual(len(self.bool_pdarray), len(retrieved_columns["bool_pdarray"]))

    def testReadHdfWithErrorAndWarn(self):
        self._create_file(
            columns=self.dict_single_column, prefix_path=f"{IOTest.io_test_dir}/iotest_single_column"
        )
        self._create_file(
            columns=self.dict_single_column,
            prefix_path=f"{IOTest.io_test_dir}/iotest_single_column_dupe",
        )

        # Make sure we can read ok
        dataset = ak.read_hdf(
            filenames=[
                f"{IOTest.io_test_dir}/iotest_single_column_LOCALE0000",
                f"{IOTest.io_test_dir}/iotest_single_column_dupe_LOCALE0000",
            ]
        )
        self.assertIsNotNone(dataset, "Expected dataset to be populated")

        # Change the name of the first file we try to raise an error due to file missing.
        with self.assertRaises(RuntimeError):
            dataset = ak.read_hdf(
                filenames=[
                    f"{IOTest.io_test_dir}/iotest_MISSING_single_column_LOCALE0000",
                    f"{IOTest.io_test_dir}/iotest_single_column_dupe_LOCALE0000",
                ]
            )

        # Run the same test with missing file, but this time with the warning flag for read_all
        with pytest.warns(RuntimeWarning, match=r"There were .* errors reading files on the server.*"):
            dataset = ak.read_hdf(
                filenames=[
                    f"{IOTest.io_test_dir}/iotest_MISSING_single_column_LOCALE0000",
                    f"{IOTest.io_test_dir}/iotest_single_column_dupe_LOCALE0000",
                ],
                strict_types=False,
                allow_errors=True,
            )
        self.assertIsNotNone(dataset, "Expected dataset to be populated")

    def testLoad(self):
        """
        Creates 1..n files depending upon the number of arkouda_server locales with three columns
        AKA datasets, loads each corresponding dataset and confirms each corresponding pdarray
        equals the input pdarray.

        :return: None
        :raise: AssertionError if the input and returned datasets (pdarrays) don't match
        """
        self._create_file(
            columns=self.dict_columns, prefix_path="{}/iotest_dict_columns".format(IOTest.io_test_dir)
        )
        result_array_tens = ak.load(
            path_prefix="{}/iotest_dict_columns".format(IOTest.io_test_dir), dataset="int_tens_pdarray"
        )["int_tens_pdarray"]
        result_array_hundreds = ak.load(
            path_prefix="{}/iotest_dict_columns".format(IOTest.io_test_dir),
            dataset="int_hundreds_pdarray",
        )["int_hundreds_pdarray"]
        result_array_floats = ak.load(
            path_prefix="{}/iotest_dict_columns".format(IOTest.io_test_dir), dataset="float_pdarray"
        )["float_pdarray"]
        result_array_bools = ak.load(
            path_prefix="{}/iotest_dict_columns".format(IOTest.io_test_dir), dataset="bool_pdarray"
        )["bool_pdarray"]

        ratens = result_array_tens.to_ndarray()
        ratens.sort()

        rahundreds = result_array_hundreds.to_ndarray()
        rahundreds.sort()

        rafloats = result_array_floats.to_ndarray()
        rafloats.sort()

        self.assertListEqual(self.int_tens_ndarray.tolist(), ratens.tolist())
        self.assertListEqual(self.int_hundreds_ndarray.tolist(), rahundreds.tolist())
        self.assertListEqual(self.float_ndarray.tolist(), rafloats.tolist())
        self.assertEqual(len(self.bool_pdarray), len(result_array_bools))

        # test load_all with file_format parameter usage
        ak.to_parquet(
            columns=self.dict_columns,
            prefix_path="{}/iotest_dict_columns_parquet".format(IOTest.io_test_dir),
        )
        result_array_tens = ak.load(
            path_prefix="{}/iotest_dict_columns_parquet".format(IOTest.io_test_dir),
            dataset="int_tens_pdarray",
            file_format="Parquet",
        )["int_tens_pdarray"]
        result_array_hundreds = ak.load(
            path_prefix="{}/iotest_dict_columns_parquet".format(IOTest.io_test_dir),
            dataset="int_hundreds_pdarray",
            file_format="Parquet",
        )["int_hundreds_pdarray"]
        result_array_floats = ak.load(
            path_prefix="{}/iotest_dict_columns_parquet".format(IOTest.io_test_dir),
            dataset="float_pdarray",
            file_format="Parquet",
        )["float_pdarray"]
        result_array_bools = ak.load(
            path_prefix="{}/iotest_dict_columns_parquet".format(IOTest.io_test_dir),
            dataset="bool_pdarray",
            file_format="Parquet",
        )["bool_pdarray"]
        ratens = result_array_tens.to_ndarray()
        ratens.sort()

        rahundreds = result_array_hundreds.to_ndarray()
        rahundreds.sort()

        rafloats = result_array_floats.to_ndarray()
        rafloats.sort()
        self.assertListEqual(self.int_tens_ndarray.tolist(), ratens.tolist())
        self.assertListEqual(self.int_hundreds_ndarray.tolist(), rahundreds.tolist())
        self.assertListEqual(self.float_ndarray.tolist(), rafloats.tolist())
        self.assertEqual(len(self.bool_pdarray), len(result_array_bools))

        # Test load with invalid prefix
        with self.assertRaises(RuntimeError):
            ak.load(
                path_prefix="{}/iotest_dict_column".format(IOTest.io_test_dir),
                dataset="int_tens_pdarray",
            )["int_tens_pdarray"]

        # Test load with invalid file
        with self.assertRaises(RuntimeError):
            ak.load(path_prefix="{}/not-a-file".format(IOTest.io_test_dir), dataset="int_tens_pdarray")[
                "int_tens_pdarray"
            ]

    def testLoadAll(self):
        self._create_file(
            columns=self.dict_columns, prefix_path="{}/iotest_dict_columns".format(IOTest.io_test_dir)
        )

        results = ak.load_all(path_prefix="{}/iotest_dict_columns".format(IOTest.io_test_dir))
        self.assertTrue("bool_pdarray" in results)
        self.assertTrue("float_pdarray" in results)
        self.assertTrue("int_tens_pdarray" in results)
        self.assertTrue("int_hundreds_pdarray" in results)

        # test load_all with file_format parameter usage
        ak.to_parquet(
            columns=self.dict_columns,
            prefix_path="{}/iotest_dict_columns_parquet".format(IOTest.io_test_dir),
        )
        results = ak.load_all(
            file_format="Parquet",
            path_prefix="{}/iotest_dict_columns_parquet".format(IOTest.io_test_dir),
        )
        self.assertTrue("bool_pdarray" in results)
        self.assertTrue("float_pdarray" in results)
        self.assertTrue("int_tens_pdarray" in results)
        self.assertTrue("int_hundreds_pdarray" in results)

        # # Test load_all with invalid prefix
        with self.assertRaises(ValueError):
            ak.load_all(path_prefix="{}/iotest_dict_column".format(IOTest.io_test_dir))

        # Test load with invalid file
        with self.assertRaises(RuntimeError):
            ak.load_all(path_prefix="{}/not-a-file".format(IOTest.io_test_dir))

    def testGetDataSets(self):
        """
        Creates 1..n files depending upon the number of arkouda_server locales containing three
        datasets and confirms the expected number of datasets along with the dataset names

        :return: None
        :raise: AssertionError if the input and returned dataset names don't match
        """
        self._create_file(
            columns=self.dict_columns, prefix_path="{}/iotest_dict_columns".format(IOTest.io_test_dir)
        )
        datasets = ak.get_datasets("{}/iotest_dict_columns_LOCALE0000".format(IOTest.io_test_dir))

        self.assertEqual(4, len(datasets))
        for dataset in datasets:
            self.assertIn(dataset, self.names)

        # Test load_all with invalid filename
        with self.assertRaises(RuntimeError):
            ak.get_datasets("{}/iotest_dict_columns_LOCALE000".format(IOTest.io_test_dir))

    def testSaveStringsDataset(self):
        # Create, save, and load Strings dataset
        strings_array = ak.array(["testing string{}".format(num) for num in list(range(0, 25))])
        strings_array.to_hdf("{}/strings-test".format(IOTest.io_test_dir), dataset="strings")
        r_strings_array = ak.load("{}/strings-test".format(IOTest.io_test_dir), dataset="strings")[
            "strings"
        ]

        strings = strings_array.to_ndarray()
        strings.sort()
        r_strings = r_strings_array.to_ndarray()
        r_strings.sort()
        self.assertListEqual(strings.tolist(), r_strings.tolist())

        # Read a part of a saved Strings dataset from one hdf5 file
        r_strings_subset = ak.read_hdf(
            filenames="{}/strings-test_LOCALE0000".format(IOTest.io_test_dir)
        ).popitem()[1]
        self.assertIsNotNone(r_strings_subset)
        self.assertTrue(isinstance(r_strings_subset[0], str))
        self.assertIsNotNone(
            ak.read_hdf(
                filenames="{}/strings-test_LOCALE0000".format(IOTest.io_test_dir),
                datasets="strings/values",
            )["strings/values"]
        )
        self.assertIsNotNone(
            ak.read_hdf(
                filenames="{}/strings-test_LOCALE0000".format(IOTest.io_test_dir),
                datasets="strings/segments",
            )["strings/segments"]
        )

        # Repeat the test using the calc_string_offsets=True option to
        # have server calculate offsets array
        r_strings_subset = ak.read_hdf(
            filenames=f"{IOTest.io_test_dir}/strings-test_LOCALE0000", calc_string_offsets=True
        ).popitem()[1]
        self.assertIsNotNone(r_strings_subset)
        self.assertTrue(isinstance(r_strings_subset[0], str))
        self.assertIsNotNone(
            ak.read_hdf(
                filenames=f"{IOTest.io_test_dir}/strings-test_LOCALE0000",
                datasets="strings/values",
                calc_string_offsets=True,
            )["strings/values"]
        )
        self.assertIsNotNone(
            ak.read_hdf(
                filenames=f"{IOTest.io_test_dir}/strings-test_LOCALE0000",
                datasets="strings/segments",
                calc_string_offsets=True,
            )["strings/segments"]
        )

    def testStringsWithoutOffsets(self):
        """
        This tests both saving & reading a strings array without saving and reading the offsets to HDF5.
        Instead the offsets array will be derived from the values/bytes area by looking for null-byte
        terminator strings
        """
        strings_array = ak.array(["testing string{}".format(num) for num in list(range(0, 25))])
        strings_array.to_hdf(
            "{}/strings-test".format(IOTest.io_test_dir), dataset="strings", save_offsets=False
        )
        r_strings_array = ak.load(
            "{}/strings-test".format(IOTest.io_test_dir), dataset="strings", calc_string_offsets=True
        )["strings"]
        strings = strings_array.to_ndarray()
        strings.sort()
        r_strings = r_strings_array.to_ndarray()
        r_strings.sort()
        self.assertListEqual(strings.tolist(), r_strings.tolist())

    def testSaveLongStringsDataset(self):
        # Create, save, and load Strings dataset
        strings = ak.array(
            [
                "testing a longer string{} to be written, loaded and appended".format(num)
                for num in list(range(0, 26))
            ]
        )
        strings.to_hdf("{}/strings-test".format(IOTest.io_test_dir), dataset="strings")

        n_strings = strings.to_ndarray()
        n_strings.sort()
        r_strings = ak.load("{}/strings-test".format(IOTest.io_test_dir), dataset="strings")[
            "strings"
        ].to_ndarray()
        r_strings.sort()

        self.assertListEqual(n_strings.tolist(), r_strings.tolist())

    def testSaveMixedStringsDataset(self):
        strings_array = ak.array(["string {}".format(num) for num in list(range(0, 25))])
        m_floats = ak.array([x / 10.0 for x in range(0, 10)])
        m_ints = ak.array(list(range(0, 10)))
        ak.to_hdf(
            {"m_strings": strings_array, "m_floats": m_floats, "m_ints": m_ints},
            "{}/multi-type-test".format(IOTest.io_test_dir),
        )
        r_mixed = ak.load_all("{}/multi-type-test".format(IOTest.io_test_dir))

        self.assertListEqual(
            np.sort(strings_array.to_ndarray()).tolist(),
            np.sort(r_mixed["m_strings"].to_ndarray()).tolist(),
        )
        self.assertIsNotNone(r_mixed["m_floats"])
        self.assertIsNotNone(r_mixed["m_ints"])

        r_floats = ak.sort(
            ak.load("{}/multi-type-test".format(IOTest.io_test_dir), dataset="m_floats")["m_floats"]
        )
        self.assertListEqual(m_floats.to_list(), r_floats.to_list())

        r_ints = ak.sort(
            ak.load("{}/multi-type-test".format(IOTest.io_test_dir), dataset="m_ints")["m_ints"]
        )
        self.assertListEqual(m_ints.to_list(), r_ints.to_list())

        strings = strings_array.to_ndarray()
        strings.sort()
        r_strings = ak.load("{}/multi-type-test".format(IOTest.io_test_dir), dataset="m_strings")[
            "m_strings"
        ].to_ndarray()
        r_strings.sort()

        self.assertListEqual(strings.tolist(), r_strings.tolist())

    def testAppendStringsDataset(self):
        strings_array = ak.array(["string {}".format(num) for num in list(range(0, 25))])
        strings_array.to_hdf("{}/append-strings-test".format(IOTest.io_test_dir), dataset="strings")
        strings_array.to_hdf(
            "{}/append-strings-test".format(IOTest.io_test_dir), dataset="strings-dupe", mode="append"
        )

        r_strings = ak.load("{}/append-strings-test".format(IOTest.io_test_dir), dataset="strings")[
            "strings"
        ]
        r_strings_dupe = ak.load(
            "{}/append-strings-test".format(IOTest.io_test_dir), dataset="strings-dupe"
        )["strings-dupe"]
        self.assertListEqual(r_strings.to_list(), r_strings_dupe.to_list())

    def testAppendMixedStringsDataset(self):
        strings_array = ak.array(["string {}".format(num) for num in list(range(0, 25))])
        strings_array.to_hdf("{}/append-multi-type-test".format(IOTest.io_test_dir), dataset="m_strings")
        m_floats = ak.array([x / 10.0 for x in range(0, 10)])
        m_ints = ak.array(list(range(0, 10)))
        ak.to_hdf(
            {"m_floats": m_floats, "m_ints": m_ints},
            "{}/append-multi-type-test".format(IOTest.io_test_dir),
            mode="append",
        )
        r_mixed = ak.load_all("{}/append-multi-type-test".format(IOTest.io_test_dir))

        self.assertIsNotNone(r_mixed["m_floats"])
        self.assertIsNotNone(r_mixed["m_ints"])

        r_floats = ak.sort(
            ak.load("{}/append-multi-type-test".format(IOTest.io_test_dir), dataset="m_floats")[
                "m_floats"
            ]
        )
        r_ints = ak.sort(
            ak.load("{}/append-multi-type-test".format(IOTest.io_test_dir), dataset="m_ints")["m_ints"]
        )
        self.assertListEqual(m_floats.to_list(), r_floats.to_list())
        self.assertListEqual(m_ints.to_list(), r_ints.to_list())

        strings = strings_array.to_ndarray()
        strings.sort()
        r_strings = r_mixed["m_strings"].to_ndarray()
        r_strings.sort()

        self.assertListEqual(strings.tolist(), r_strings.tolist())

    def testStrictTypes(self):
        N = 100
        prefix = "{}/strict-type-test".format(IOTest.io_test_dir)
        inttypes = [np.uint32, np.int64, np.uint16, np.int16]
        floattypes = [np.float32, np.float64, np.float32, np.float64]
        for i, (it, ft) in enumerate(zip(inttypes, floattypes)):
            with h5py.File("{}-{}".format(prefix, i), "w") as f:
                idata = np.arange(i * N, (i + 1) * N, dtype=it)
                id = f.create_dataset("integers", data=idata)
                id.attrs["ObjType"] = 1
                fdata = np.arange(i * N, (i + 1) * N, dtype=ft)
                fd = f.create_dataset("floats", data=fdata)
                fd.attrs["ObjType"] = 1
        with self.assertRaises(RuntimeError):
            ak.read_hdf(prefix + "*")

        a = ak.read_hdf(prefix + "*", strict_types=False)
        self.assertListEqual(a["integers"].to_list(), np.arange(len(inttypes) * N).tolist())
        self.assertTrue(
            np.allclose(a["floats"].to_ndarray(), np.arange(len(floattypes) * N, dtype=np.float64))
        )

    def testTo_ndarray(self):
        ones = ak.ones(10)
        n_ones = ones.to_ndarray()
        new_ones = ak.array(n_ones)
        self.assertListEqual(ones.to_list(), new_ones.to_list())

        empty_ones = ak.ones(0)
        n_empty_ones = empty_ones.to_ndarray()
        new_empty_ones = ak.array(n_empty_ones)
        self.assertListEqual(empty_ones.to_list(), new_empty_ones.to_list())

    def testSmallArrayToHDF5(self):
        a1 = ak.array([1])
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            a1.to_hdf(f"{tmp_dirname}/small_numeric", dataset="a1")
            # Now load it back in
            a2 = ak.load(f"{tmp_dirname}/small_numeric", dataset="a1")["a1"]
            self.assertEqual(str(a1), str(a2))

    # This tests small array corner cases on multi-locale environments
    def testSmallStringArrayToHDF5(self):
        a1 = ak.array(["ab", "cd"])
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            a1.to_hdf(f"{tmp_dirname}/small_string_array", dataset="a1")
            # Now load it back in
            a2 = ak.load(f"{tmp_dirname}/small_string_array", dataset="a1")["a1"]
            self.assertEqual(str(a1), str(a2))

        # Test a single string
        b1 = ak.array(["123456789"])
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            b1.to_hdf(f"{tmp_dirname}/single_string", dataset="b1")
            # Now load it back in
            b2 = ak.load(f"{tmp_dirname}/single_string", dataset="b1")["b1"]
            self.assertEqual(str(b1), str(b2))

    def testUint64ToFromHDF5(self):
        """
        Test our ability to read/write uint64 to HDF5
        """
        npa1 = np.array(
            [18446744073709551500, 18446744073709551501, 18446744073709551502], dtype=np.uint64
        )
        pda1 = ak.array(npa1)
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            pda1.to_hdf(f"{tmp_dirname}/small_numeric", dataset="pda1")
            # Now load it back in
            pda2 = ak.load(f"{tmp_dirname}/small_numeric", dataset="pda1")["pda1"]
            self.assertEqual(str(pda1), str(pda2))
            self.assertEqual(18446744073709551500, pda2[0])
            self.assertListEqual(pda2.to_list(), npa1.tolist())

    def testBigIntHdf5(self):
        # pdarray
        a = ak.arange(3, dtype=ak.bigint)
        a += 2**200
        a.max_bits = 201

        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            a.to_hdf(f"{tmp_dirname}/bigint_test", dataset="bigint_test")
            rd_a = ak.read_hdf(f"{tmp_dirname}/bigint_test*")["bigint_test"]
            self.assertListEqual(a.to_list(), rd_a.to_list())
            self.assertEqual(a.max_bits, rd_a.max_bits)

        # arrayview
        a = ak.arange(27, dtype=ak.bigint)
        a += 2**200
        a.max_bits = 201

        av = a.reshape((3, 3, 3))

        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            av.to_hdf(f"{tmp_dirname}/bigint_test")
            rd_av = ak.read_hdf(f"{tmp_dirname}/bigint_test*").popitem()[1]
            self.assertIsInstance(rd_av, ak.ArrayView)
            self.assertListEqual(av.base.to_list(), rd_av.base.to_list())
            self.assertEqual(av.base.max_bits, rd_av.base.max_bits)

        # groupby
        a = ak.arange(5, dtype=ak.bigint)
        g = ak.GroupBy(a)
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            g.to_hdf(f"{tmp_dirname}/bigint_test")
            rd_g = ak.read_hdf(f"{tmp_dirname}/bigint_test*").popitem()[1]
            self.assertIsInstance(rd_g, ak.GroupBy)
            self.assertListEqual(g.keys.to_list(), rd_g.keys.to_list())
            self.assertListEqual(g.unique_keys.to_list(), rd_g.unique_keys.to_list())
            self.assertListEqual(g.permutation.to_list(), rd_g.permutation.to_list())
            self.assertListEqual(g.segments.to_list(), rd_g.segments.to_list())

        # bigint segarray
        a = ak.arange(10, dtype=ak.bigint)
        a += 2**200
        a.max_bits = 212
        s = ak.arange(0, 10, 2)
        sa = ak.SegArray(s, a)
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            sa.to_hdf(f"{tmp_dirname}/bigint_test")
            rd_sa = ak.read_hdf(f"{tmp_dirname}/bigint_test*").popitem()[1]
            self.assertIsInstance(rd_sa, ak.SegArray)
            self.assertListEqual(sa.values.to_list(), rd_sa.values.to_list())
            self.assertListEqual(sa.segments.to_list(), rd_sa.segments.to_list())

    def testUint64ToFromArray(self):
        """
        Test conversion to and from numpy array / pdarray using unsigned 64bit integer (uint64)
        """
        npa1 = np.array(
            [18446744073709551500, 18446744073709551501, 18446744073709551502], dtype=np.uint64
        )
        pda1 = ak.array(npa1)
        self.assertEqual(18446744073709551500, pda1[0])
        self.assertListEqual(pda1.to_list(), npa1.tolist())

    def testHdfUnsanitizedNames(self):
        # Test when quotes are part of the dataset name
        my_arrays = {'foo"0"': ak.arange(100), 'bar"': ak.arange(100)}
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            ak.to_hdf(my_arrays, f"{tmp_dirname}/bad_dataset_names")
            ak.read_hdf(f"{tmp_dirname}/bad_dataset_names*")

    def testInternalVersions(self):
        """
        Test loading legacy files to ensure they can still be read.
        Test loading internal arkouda hdf5 structuring by loading v0 and v1 files.
        v1 contains _arkouda_metadata group and attributes, v0 does not.
        Files are located under `test/resources` ... where server-side unit tests are located.
        """
        # Note: pytest unit tests are located under "tests/" vs chapel "test/"
        # The test files are located in the Chapel `test/resources` directory
        # Determine where the test was launched by inspecting our path and update it accordingly
        cwd = os.getcwd()
        if cwd.endswith("tests"):  # IDEs may launch unit tests from this location
            cwd = cwd + "/server/resources"
        else:  # assume arkouda root dir
            cwd += "/tests/server/resources"

        # Now that we've figured out our loading path, load the files and test the lengths
        v0 = ak.load(cwd + "/array_v0.hdf5", file_format="hdf5").popitem()[1]
        v1 = ak.load(cwd + "/array_v1.hdf5", file_format="hdf5").popitem()[1]
        self.assertEqual(50, v0.size)
        self.assertEqual(50, v1.size)

    def test_multi_dim_rdwr(self):
        arr = ak.ArrayView(ak.arange(27), ak.array([3, 3, 3]))
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            arr.to_hdf(tmp_dirname + "/multi_dim_test", dataset="MultiDimObj", mode="append")
            # load data back
            read_arr = ak.read_hdf(tmp_dirname + "/multi_dim_test*", datasets="MultiDimObj")[
                "MultiDimObj"
            ]
            self.assertTrue(np.array_equal(arr.to_ndarray(), read_arr.to_ndarray()))

    def test_legacy_read(self):
        cwd = os.getcwd()
        if cwd.endswith("tests"):  # IDEs may launch unit tests from this location
            cwd = cwd + "/../resources/hdf5-testing"
        else:  # assume arkouda root dir
            cwd += "/resources/hdf5-testing"
        rd_arr = ak.read_hdf(f"{cwd}/Legacy_String.hdf5").popitem()[1]

        self.assertListEqual(["ABC", "DEF", "GHI"], rd_arr.to_list())

    def test_csv_read_write(self):
        # first test that can read csv with no header not written by Arkouda
        cols = ["ColA", "ColB", "ColC"]
        a = ["ABC", "DEF"]
        b = ["123", "345"]
        c = ["3.14", "5.56"]
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            with open(f"{tmp_dirname}/non_ak.csv", "w") as f:
                f.write(",".join(cols) + "\n")
                f.write(f"{a[0]},{b[0]},{c[0]}\n")
                f.write(f"{a[1]},{b[1]},{c[1]}\n")

            data = ak.read_csv(f"{tmp_dirname}/non_ak.csv")
            self.assertListEqual(list(data.keys()), cols)
            self.assertListEqual(data["ColA"].to_list(), a)
            self.assertListEqual(data["ColB"].to_list(), b)
            self.assertListEqual(data["ColC"].to_list(), c)

            data = ak.read_csv(f"{tmp_dirname}/non_ak.csv", datasets="ColB")["ColB"]
            self.assertIsInstance(data, ak.Strings)
            self.assertListEqual(data.to_list(), b)

        # test can read csv with header not written by Arkouda
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            with open(f"{tmp_dirname}/non_ak.csv", "w") as f:
                f.write("**HEADER**\n")
                f.write("str,int64,float64\n")
                f.write("*/HEADER/*\n")
                f.write(",".join(cols) + "\n")
                f.write(f"{a[0]},{b[0]},{c[0]}\n")
                f.write(f"{a[1]},{b[1]},{c[1]}\n")

            data = ak.read_csv(f"{tmp_dirname}/non_ak.csv")
            self.assertListEqual(list(data.keys()), cols)
            self.assertListEqual(data["ColA"].to_list(), a)
            self.assertListEqual(data["ColB"].to_list(), [int(x) for x in b])
            self.assertListEqual(data["ColC"].to_list(), [round(float(x), 2) for x in c])

            # test reading subset of columns
            data = ak.read_csv(f"{tmp_dirname}/non_ak.csv", datasets="ColB")["ColB"]
            self.assertIsInstance(data, ak.pdarray)
            self.assertListEqual(data.to_list(), [int(x) for x in b])

        # test writing file with Arkouda with non-standard delim
        d = {
            cols[0]: ak.array(a),
            cols[1]: ak.array([int(x) for x in b]),
            cols[2]: ak.array([round(float(x), 2) for x in c]),
        }
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            ak.to_csv(d, f"{tmp_dirname}/non_standard_delim.csv", col_delim="|*|")

            # test reading that file with Arkouda
            data = ak.read_csv(f"{tmp_dirname}/non_standard_delim*", column_delim="|*|")
            self.assertListEqual(list(data.keys()), cols)
            self.assertListEqual(data["ColA"].to_list(), a)
            self.assertListEqual(data["ColB"].to_list(), [int(x) for x in b])
            self.assertListEqual(data["ColC"].to_list(), [round(float(x), 2) for x in c])

            # test reading subset of columns
            data = ak.read_csv(
                f"{tmp_dirname}/non_standard_delim*", datasets="ColB", column_delim="|*|"
            )["ColB"]
            self.assertIsInstance(data, ak.pdarray)
            self.assertListEqual(data.to_list(), [int(x) for x in b])

        # larger data set testing
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            d = {
                "ColA": ak.randint(0, 50, 101),
                "ColB": ak.randint(0, 50, 101),
                "ColC": ak.randint(0, 50, 101),
            }

            ak.to_csv(d, f"{tmp_dirname}/non_equal_set.csv")
            data = ak.read_csv(f"{tmp_dirname}/non_equal_set*")
            self.assertListEqual(data["ColA"].to_list(), d["ColA"].to_list())
            self.assertListEqual(data["ColB"].to_list(), d["ColB"].to_list())
            self.assertListEqual(data["ColC"].to_list(), d["ColC"].to_list())

    def test_segarray_hdf(self):
        a = [0, 1, 2, 3]
        b = [4, 0, 5, 6, 0, 7, 8, 0]
        c = [9, 0, 0]

        # int64 test
        flat = a + b + c
        segments = ak.array([0, len(a), len(a) + len(b)])
        dtype = ak.dtypes.int64
        akflat = ak.array(flat, dtype)
        segarr = ak.SegArray(segments, akflat)

        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            segarr.to_hdf(f"{tmp_dirname}/segarray_int")
            # Now load it back in
            seg2 = ak.load(f"{tmp_dirname}/segarray_int", dataset="segarray")["segarray"]
            self.assertListEqual(segarr.segments.to_list(), seg2.segments.to_list())
            self.assertListEqual(segarr.values.to_list(), seg2.values.to_list())

        # uint64 test
        dtype = ak.dtypes.uint64
        akflat = ak.array(flat, dtype)
        segarr = ak.SegArray(segments, akflat)

        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            segarr.to_hdf(f"{tmp_dirname}/segarray_uint")
            # Now load it back in
            seg2 = ak.load(f"{tmp_dirname}/segarray_uint", dataset="segarray")["segarray"]
            self.assertListEqual(segarr.segments.to_list(), seg2.segments.to_list())
            self.assertListEqual(segarr.values.to_list(), seg2.values.to_list())

        # float64 test
        dtype = ak.dtypes.float64
        akflat = ak.array(flat, dtype)
        segarr = ak.SegArray(segments, akflat)

        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            segarr.to_hdf(f"{tmp_dirname}/segarray_float")
            # Now load it back in
            seg2 = ak.load(f"{tmp_dirname}/segarray_float", dataset="segarray")["segarray"]
            self.assertListEqual(segarr.segments.to_list(), seg2.segments.to_list())
            self.assertListEqual(segarr.values.to_list(), seg2.values.to_list())

        # bool test
        dtype = ak.dtypes.bool_
        akflat = ak.array(flat, dtype)
        segarr = ak.SegArray(segments, akflat)

        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            segarr.to_hdf(f"{tmp_dirname}/segarray_bool")
            # Now load it back in
            seg2 = ak.load(f"{tmp_dirname}/segarray_bool", dataset="segarray")["segarray"]
            self.assertListEqual(segarr.segments.to_list(), seg2.segments.to_list())
            self.assertListEqual(segarr.values.to_list(), seg2.values.to_list())

    def test_dataframe_segarr(self):
        a = [0, 1, 2, 3]
        b = [4, 0, 5, 6, 0, 7, 8, 0]
        c = [9, 0, 0]

        # int64 test
        flat = a + b + c
        segments = ak.array([0, len(a), len(a) + len(b)])
        dtype = ak.dtypes.int64
        akflat = ak.array(flat, dtype)
        segarr = ak.SegArray(segments, akflat)

        s = ak.array(["abc", "def", "ghi"])
        df = ak.DataFrame([segarr, s])
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            df.to_hdf(f"{tmp_dirname}/dataframe_segarr")
            df_load = ak.DataFrame.load(f"{tmp_dirname}/dataframe_segarr")
            self.assertTrue(df.to_pandas().equals(df_load.to_pandas()))

    def test_hdf_groupby(self):
        # test for categorical and multiple keys
        string = ak.array(["a", "b", "a", "b", "c"])
        cat = ak.Categorical(string)
        cat_from_codes = ak.Categorical.from_codes(
            codes=ak.array([0, 1, 0, 1, 2]), categories=ak.array(["a", "b", "c"])
        )
        cat_grouping = ak.GroupBy([cat, cat_from_codes])
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            cat_grouping.to_hdf(f"{tmp_dirname}/cat_test")
            cg_load = ak.read(f"{tmp_dirname}/cat_test*").popitem()[1]
            self.assertEqual(len(cg_load.keys), len(cat_grouping.keys))
            self.assertListEqual(cg_load.permutation.to_list(), cat_grouping.permutation.to_list())
            self.assertListEqual(cg_load.segments.to_list(), cat_grouping.segments.to_list())
            self.assertListEqual(cg_load._uki.to_list(), cat_grouping._uki.to_list())
            for k, kload in zip(cat_grouping.keys, cg_load.keys):
                self.assertListEqual(k.to_list(), kload.to_list())

        # test Strings GroupBy
        str_grouping = ak.GroupBy(string)
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            str_grouping.to_hdf(f"{tmp_dirname}/str_test")
            str_load = ak.read(f"{tmp_dirname}/str_test*").popitem()[1]
            self.assertEqual(len(str_load.keys), len(str_grouping.keys))
            self.assertListEqual(str_load.permutation.to_list(), str_grouping.permutation.to_list())
            self.assertListEqual(str_load.segments.to_list(), str_grouping.segments.to_list())
            self.assertListEqual(str_load._uki.to_list(), str_grouping._uki.to_list())
            self.assertListEqual(str_grouping.keys.to_list(), str_load.keys.to_list())

        # test pdarray GroupBy
        pda = ak.array([0, 1, 2, 0, 2])
        g = ak.GroupBy(pda)
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            g.to_hdf(f"{tmp_dirname}/pd_test")
            g_load = ak.read(f"{tmp_dirname}/pd_test*").popitem()[1]
            self.assertEqual(len(g_load.keys), len(g.keys))
            self.assertListEqual(g_load.permutation.to_list(), g.permutation.to_list())
            self.assertListEqual(g_load.segments.to_list(), g.segments.to_list())
            self.assertListEqual(g_load._uki.to_list(), g._uki.to_list())
            self.assertListEqual(g_load.keys.to_list(), g.keys.to_list())

    def test_hdf_overwrite_pdarray(self):
        # test repack with a single object
        a = ak.arange(1000)
        b = ak.randint(0, 100, 1000)
        c = ak.arange(15)
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            a.to_hdf(f"{tmp_dirname}/pda_test")
            b.to_hdf(f"{tmp_dirname}/pda_test", dataset="array_2", mode="append")
            f_list = glob.glob(f"{tmp_dirname}/pda_test_*")
            orig_size = sum(os.path.getsize(f) for f in f_list)
            c.update_hdf(f"{tmp_dirname}/pda_test")

            new_size = sum(os.path.getsize(f) for f in f_list)

            # ensure that the column was actually overwritten
            self.assertLess(new_size, orig_size)
            data = ak.read_hdf(f"{tmp_dirname}/pda_test_*")
            self.assertListEqual(data["array"].to_list(), c.to_list())

        # test with repack off - file should get larger
        b = ak.arange(1000)
        c = ak.arange(15)
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            a.to_hdf(f"{tmp_dirname}/pda_test")
            b.to_hdf(f"{tmp_dirname}/pda_test", dataset="array_2", mode="append")
            f_list = glob.glob(f"{tmp_dirname}/pda_test_*")
            orig_size = sum(os.path.getsize(f) for f in f_list)
            # hdf5 only releases memory if overwritting last dset so overwrite first
            c.update_hdf(f"{tmp_dirname}/pda_test", dataset="array", repack=False)

            new_size = sum(os.path.getsize(f) for f in f_list)

            # ensure that the column was actually overwritten
            self.assertGreaterEqual(
                new_size, orig_size
            )  # ensure that overwritten data mem is not released
            data = ak.read_hdf(f"{tmp_dirname}/pda_test_*")
            self.assertListEqual(data["array"].to_list(), c.to_list())

        # test overwrites with different types
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            a.to_hdf(f"{tmp_dirname}/pda_test")
            b = ak.arange(15, dtype=ak.uint64)
            b.update_hdf(f"{tmp_dirname}/pda_test")
            data = ak.read_hdf(f"{tmp_dirname}/pda_test_*").popitem()[1]
            self.assertListEqual(data.to_list(), b.to_list())

            b = ak.arange(150, dtype=ak.float64)
            b.update_hdf(f"{tmp_dirname}/pda_test")
            data = ak.read_hdf(f"{tmp_dirname}/pda_test_*").popitem()[1]
            self.assertListEqual(data.to_list(), b.to_list())

            b = ak.arange(1000, dtype=ak.bool_)
            b.update_hdf(f"{tmp_dirname}/pda_test")
            data = ak.read_hdf(f"{tmp_dirname}/pda_test_*").popitem()[1]
            self.assertListEqual(data.to_list(), b.to_list())

    def test_hdf_overwrite_strings(self):
        # test repack with a single object
        a = ak.random_strings_uniform(0, 16, 1000)
        b = ak.random_strings_uniform(0, 16, 1000)
        c = ak.random_strings_uniform(0, 16, 10)
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            a.to_hdf(f"{tmp_dirname}/str_test", dataset="test_set")
            b.to_hdf(f"{tmp_dirname}/str_test", mode="append")
            f_list = glob.glob(f"{tmp_dirname}/str_test_*")
            orig_size = sum(os.path.getsize(f) for f in f_list)
            c.update_hdf(f"{tmp_dirname}/str_test", dataset="test_set")

            new_size = sum(os.path.getsize(f) for f in f_list)
            # ensure that the column was actually overwritten
            self.assertLess(new_size, orig_size)
            data = ak.read_hdf(f"{tmp_dirname}/str_test_*")
            self.assertListEqual(data["test_set"].to_list(), c.to_list())

        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            a.to_hdf(f"{tmp_dirname}/str_test", dataset="test_set")
            b.to_hdf(f"{tmp_dirname}/str_test", mode="append")
            f_list = glob.glob(f"{tmp_dirname}/str_test_*")
            orig_size = sum(os.path.getsize(f) for f in f_list)
            # hdf5 only releases memory if overwritting last dset so overwrite first
            c.update_hdf(f"{tmp_dirname}/str_test", dataset="test_set", repack=False)

            new_size = sum(os.path.getsize(f) for f in f_list)
            # ensure that the column was actually overwritten
            self.assertGreaterEqual(new_size, orig_size)
            data = ak.read_hdf(f"{tmp_dirname}/str_test_*")
            self.assertListEqual(data["test_set"].to_list(), c.to_list())

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
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            df.to_hdf(f"{tmp_dirname}/df_test")
            f_list = glob.glob(f"{tmp_dirname}/df_test_*")
            orig_size = sum(os.path.getsize(f) for f in f_list)
            # hdf5 only releases memory if overwritting last dset so overwrite first
            odf.update_hdf(f"{tmp_dirname}/df_test")

            new_size = sum(os.path.getsize(f) for f in f_list)
            # ensure that the column was actually overwritten
            self.assertLessEqual(new_size, orig_size)
            data = ak.read_hdf(f"{tmp_dirname}/df_test_*")
            self.assertListEqual(data["a"].to_list(), df["a"].to_list())
            self.assertListEqual(data["b"].to_list(), odf["b"].to_list())
            self.assertListEqual(data["c"].to_list(), odf["c"].to_list())
            self.assertListEqual(data["d"].to_list(), df["d"].to_list())

        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            df.to_hdf(f"{tmp_dirname}/df_test")
            f_list = glob.glob(f"{tmp_dirname}/df_test*")
            orig_size = sum(os.path.getsize(f) for f in f_list)
            # hdf5 only releases memory if overwritting last dset so overwrite first
            odf.update_hdf(f"{tmp_dirname}/df_test", repack=False)

            new_size = sum(os.path.getsize(f) for f in f_list)
            # ensure that the column was actually overwritten
            self.assertGreaterEqual(new_size, orig_size)
            data = ak.read_hdf(f"{tmp_dirname}/df_test_*")
            self.assertListEqual(data["a"].to_list(), df["a"].to_list())
            self.assertListEqual(data["b"].to_list(), odf["b"].to_list())
            self.assertListEqual(data["c"].to_list(), odf["c"].to_list())
            self.assertListEqual(data["d"].to_list(), df["d"].to_list())

    def test_overwrite_segarray(self):
        sa1 = ak.SegArray(ak.arange(0, 1000, 5), ak.arange(1000))
        sa2 = ak.SegArray(ak.arange(0, 100, 5), ak.arange(100))
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            sa1.to_hdf(f"{tmp_dirname}/segarray_test")
            sa1.to_hdf(f"{tmp_dirname}/segarray_test", dataset="seg2", mode="append")
            f_list = glob.glob(f"{tmp_dirname}/segarray_test_*")
            orig_size = sum(os.path.getsize(f) for f in f_list)

            sa2.update_hdf(f"{tmp_dirname}/segarray_test")

            new_size = sum(os.path.getsize(f) for f in f_list)
            # ensure that the column was actually overwritten
            self.assertLessEqual(new_size, orig_size)
            data = ak.read_hdf(f"{tmp_dirname}/segarray_test_*")
            self.assertListEqual(data["segarray"].values.to_list(), sa2.values.to_list())
            self.assertListEqual(data["segarray"].segments.to_list(), sa2.segments.to_list())

        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            sa1.to_hdf(f"{tmp_dirname}/segarray_test")
            sa1.to_hdf(f"{tmp_dirname}/segarray_test", dataset="seg2", mode="append")
            f_list = glob.glob(f"{tmp_dirname}/segarray_test_*")
            orig_size = sum(os.path.getsize(f) for f in f_list)

            sa2.update_hdf(f"{tmp_dirname}/segarray_test", repack=False)

            new_size = sum(os.path.getsize(f) for f in f_list)
            # ensure that the column was actually overwritten
            self.assertGreaterEqual(new_size, orig_size)
            data = ak.read_hdf(f"{tmp_dirname}/segarray_test_*")
            self.assertListEqual(data["segarray"].values.to_list(), sa2.values.to_list())
            self.assertListEqual(data["segarray"].segments.to_list(), sa2.segments.to_list())

    def test_overwrite_arrayview(self):
        a = ak.arange(27)
        av = a.reshape((3, 3, 3))
        a2 = ak.arange(8)
        av2 = a2.reshape((2, 2, 2))
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            av.to_hdf(f"{tmp_dirname}/array_view_test")
            av2.update_hdf(f"{tmp_dirname}/array_view_test", repack=False)
            data = ak.read_hdf(f"{tmp_dirname}/array_view_test_*").popitem()[1]
            self.assertListEqual(av2.to_list(), data.to_list())

    def test_overwrite(self):
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
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            df.to_hdf(f"{tmp_dirname}/overwrite_test")
            f_list = glob.glob(f"{tmp_dirname}/overwrite_test_*")
            orig_size = sum(os.path.getsize(f) for f in f_list)
            # hdf5 only releases memory if overwritting last dset so overwrite first
            ak.update_hdf(replace, f"{tmp_dirname}/overwrite_test")

            new_size = sum(os.path.getsize(f) for f in f_list)
            # ensure that the column was actually overwritten
            self.assertLess(new_size, orig_size)
            data = ak.read_hdf(f"{tmp_dirname}/overwrite_test_*")
            self.assertListEqual(data["b"].to_list(), replace["b"].to_list())
            self.assertListEqual(data["c"].to_list(), replace["c"].to_list())

        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            df.to_hdf(f"{tmp_dirname}/overwrite_test")
            f_list = glob.glob(f"{tmp_dirname}/overwrite_test_*")
            orig_size = sum(os.path.getsize(f) for f in f_list)
            # hdf5 only releases memory if overwritting last dset so overwrite first
            ak.update_hdf(replace, f"{tmp_dirname}/overwrite_test", repack=False)

            new_size = sum(os.path.getsize(f) for f in f_list)
            # ensure that the column was actually overwritten
            self.assertGreaterEqual(new_size, orig_size)
            data = ak.read_hdf(f"{tmp_dirname}/overwrite_test_*")
            self.assertListEqual(data["b"].to_list(), replace["b"].to_list())
            self.assertListEqual(data["c"].to_list(), replace["c"].to_list())

    def test_overwrite_single_dset(self):
        # we need to test that both repack=False and repack=True generate the same file size here
        a = ak.arange(1000)
        b = ak.arange(15)
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            a.to_hdf(f"{tmp_dirname}/test_file")
            b.update_hdf(f"{tmp_dirname}/test_file")
            f_list = glob.glob(f"{tmp_dirname}/test_file*")
            f1_size = sum(os.path.getsize(f) for f in f_list)

            a.to_hdf(f"{tmp_dirname}/test_file_2")
            b.update_hdf(f"{tmp_dirname}/test_file_2", repack=False)
            f_list = glob.glob(f"{tmp_dirname}/test_file_2_*")
            f2_size = sum(os.path.getsize(f) for f in f_list)

            self.assertEqual(f1_size, f2_size)

    def test_segarray_str_hdf5(self):
        words = ak.array(["one,two,three", "uno,dos,tres"])
        strs, segs = words.split(",", return_segments=True)

        x = ak.SegArray(segs, strs)
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            x.to_hdf(f"{tmp_dirname}/test_file")
            rd = ak.read_hdf(f"{tmp_dirname}/test_file*").popitem()[1]
            self.assertIsInstance(rd, ak.SegArray)
            self.assertListEqual(x.segments.to_list(), rd.segments.to_list())
            self.assertListEqual(x.values.to_list(), rd.values.to_list())

    def test_snapshot(self):
        from pandas.testing import assert_frame_equal

        df = ak.DataFrame(
            {
                "int_col": ak.arange(10),
                "uint_col": ak.array([i + 2**63 for i in range(10)], dtype=ak.uint64),
                "float_col": ak.linspace(-3.5, 3.5, 10),
                "bool_col": ak.randint(0, 2, 10, dtype=ak.bool_),
                "bigint_col": ak.array([i + 2**200 for i in range(10)], dtype=ak.bigint),
                "segarr_col": ak.SegArray(ak.arange(0, 20, 2), ak.randint(0, 3, 20)),
                "str_col": ak.random_strings_uniform(0, 3, 10),
                "ip": ak.IPv4(ak.arange(10)),
                "datetime": ak.Datetime(ak.arange(10)),
                "timedelta": ak.Timedelta(ak.arange(10)),
            }
        )
        df_str_idx = df.copy()
        df_str_idx._set_index(["A" + str(i) for i in range(len(df))])
        col_order = df.columns.values
        df_ref = df.to_pandas()
        df_str_idx_ref = df_str_idx.to_pandas(retain_index=True)
        a = ak.randint(0, 10, 100)
        a_ref = a.to_list()
        s = ak.random_strings_uniform(0, 5, 50)
        s_ref = s.to_list()
        c = ak.Categorical(s)
        c_ref = c.to_list()
        g = ak.GroupBy(a)
        g_ref = {
            "perm": g.permutation.to_list(),
            "keys": g.keys.to_list(),
            "segments": g.segments.to_list(),
        }

        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            ak.snapshot(f"{tmp_dirname}/arkouda_snapshot_test")
            # delete variables
            del df
            del df_str_idx
            del a
            del s
            del c
            del g

            # verify no longer in the namespace
            with self.assertRaises(NameError):
                self.assertTrue(not df)
            with self.assertRaises(NameError):
                self.assertTrue(not df_str_idx)
            with self.assertRaises(NameError):
                self.assertTrue(not a)
            with self.assertRaises(NameError):
                self.assertTrue(not s)
            with self.assertRaises(NameError):
                self.assertTrue(not c)
            with self.assertRaises(NameError):
                self.assertTrue(not g)

            # restore the variables
            data = ak.restore(f"{tmp_dirname}/arkouda_snapshot_test")
            for vn in ["df", "df_str_idx", "a", "s", "c", "g"]:
                # ensure all variable names returned
                self.assertTrue(vn in data.keys())

            # validate that restored variables are correct
            self.assertTrue(
                assert_frame_equal(df_ref[col_order], data["df"].to_pandas(retain_index=True)[col_order])
                is None
            )
            self.assertTrue(
                assert_frame_equal(
                    df_str_idx_ref[col_order], data["df_str_idx"].to_pandas(retain_index=True)[col_order]
                )
                is None
            )
            self.assertListEqual(a_ref, data["a"].to_list())
            self.assertListEqual(s_ref, data["s"].to_list())
            self.assertListEqual(c_ref, data["c"].to_list())
            self.assertListEqual(g_ref["perm"], data["g"].permutation.to_list())
            self.assertListEqual(g_ref["keys"], data["g"].keys.to_list())
            self.assertListEqual(g_ref["segments"], data["g"].segments.to_list())

    def test_segarr_edge(self):
        """
        This test was added specifically for issue #2612.
        Pierce will be adding testing to the new framework for this.
        """
        df = ak.DataFrame(
            {
                "c_11": ak.SegArray(
                    ak.array([0, 2, 3, 3]), ak.array(["a", "b", "", "c", "d", "e", "f", "g", "h", "i"])
                )
            }
        )
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            df.to_hdf(f"{tmp_dirname}/seg_test")

            rd_data = ak.read_hdf(f"{tmp_dirname}/seg_test*").popitem()[1]
            self.assertListEqual(df["c_11"].to_list(), rd_data.to_list())

        df = ak.DataFrame({"c_2": ak.SegArray(ak.array([0, 9, 14]), ak.arange(-10, 10))})
        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            df.to_hdf(f"{tmp_dirname}/seg_test")

            # only verifying the read is successful
            rd_arr = ak.read_hdf(
                filenames=[f"{tmp_dirname}/seg_test_LOCALE0000", f"{tmp_dirname}/seg_test_MISSING"],
                strict_types=False,
                allow_errors=True,
            )

    def test_special_dtypes(self):
        """
        This test is simply to ensure that the dtype is persisted through the io
        operation. It ultimately uses the process of pdarray, but need to ensure
        correct Arkouda Object Type is returned
        """
        ip = ak.IPv4(ak.arange(10))
        dt = ak.Datetime(ak.arange(10))
        td = ak.Timedelta(ak.arange(10))
        df = ak.DataFrame({"ip": ip, "datetime": dt, "timedelta": td})

        with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
            ip.to_hdf(f"{tmp_dirname}/ip_test")
            rd_ip = ak.read_hdf(f"{tmp_dirname}/ip_test*").popitem()[1]
            self.assertIsInstance(rd_ip, ak.IPv4)
            self.assertListEqual(ip.to_list(), rd_ip.to_list())

            dt.to_hdf(f"{tmp_dirname}/dt_test")
            rd_dt = ak.read_hdf(f"{tmp_dirname}/dt_test*").popitem()[1]
            self.assertIsInstance(rd_dt, ak.Datetime)
            self.assertListEqual(dt.to_list(), rd_dt.to_list())

            td.to_hdf(f"{tmp_dirname}/td_test")
            rd_td = ak.read_hdf(f"{tmp_dirname}/td_test*").popitem()[1]
            self.assertIsInstance(rd_td, ak.Timedelta)
            self.assertListEqual(td.to_list(), rd_td.to_list())

            df.to_hdf(f"{tmp_dirname}/df_test")
            rd_df = ak.read_hdf(f"{tmp_dirname}/df_test*")

            self.assertIsInstance(rd_df["ip"], ak.IPv4)
            self.assertIsInstance(rd_df["datetime"], ak.Datetime)
            self.assertIsInstance(rd_df["timedelta"], ak.Timedelta)
            self.assertListEqual(df["ip"].to_list(), rd_df["ip"].to_list())
            self.assertListEqual(df["datetime"].to_list(), rd_df["datetime"].to_list())
            self.assertListEqual(df["timedelta"].to_list(), rd_df["timedelta"].to_list())

    def test_index(self):
        tests = [
            ak.arange(10),
            ak.linspace(-2.5, 2.5, 10),
            ak.random_strings_uniform(1, 2, 10),
            ak.Categorical(ak.random_strings_uniform(1, 2, 10)),
        ]

        for t in tests:
            idx = ak.Index(t)
            with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
                idx.to_hdf(f"{tmp_dirname}/idx_test")
                rd_idx = ak.read_hdf(f"{tmp_dirname}/idx_test*").popitem()[1]

                self.assertIsInstance(rd_idx, ak.Index)
                self.assertEqual(type(rd_idx.values), type(idx.values))
                self.assertListEqual(idx.to_list(), rd_idx.to_list())

    def test_multi_index(self):
        tests = [
            ak.arange(10),
            ak.linspace(-2.5, 2.5, 10),
            ak.random_strings_uniform(1, 2, 10),
            ak.Categorical(ak.random_strings_uniform(1, 2, 10)),
        ]
        for t1 in tests:
            for t2 in tests:
                idx = ak.Index.factory([t1, t2])
                with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
                    idx.to_hdf(f"{tmp_dirname}/idx_test")
                    rd_idx = ak.read_hdf(f"{tmp_dirname}/idx_test*").popitem()[1]

                    self.assertIsInstance(rd_idx, ak.MultiIndex)
                    self.assertListEqual(idx.to_list(), rd_idx.to_list())

    def test_zarr_read_write(self):
        pytest.skip()
        shapes = [(10,), (20,)]
        chunk_shapes = [(2,), (3,)]
        dtypes = [ak.int64, ak.float64]
        for shape,chunk_shape in zip(shapes,chunk_shapes):
            for dtype in dtypes:
                a = Array.full(shape, 7, dtype=dtype)
                with tempfile.TemporaryDirectory(dir=IOTest.io_test_dir) as tmp_dirname:
                    to_zarr(f"{tmp_dirname}", a._array, chunk_shape)
                    b = read_zarr(f"{tmp_dirname}", len(shape), dtype)
                    self.assertTrue(np.allclose(a.to_ndarray(), b.to_ndarray()))


    def tearDown(self):
        super(IOTest, self).tearDown()
        for f in glob.glob("{}/*".format(IOTest.io_test_dir)):
            os.remove(f)

    @classmethod
    def tearDownClass(cls):
        super(IOTest, cls).tearDownClass()
        shutil.rmtree(IOTest.io_test_dir)
