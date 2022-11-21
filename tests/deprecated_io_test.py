import glob
import os
import shutil
import tempfile
import pytest

import h5py
import numpy as np
import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda import io_util, io

TYPES = ("int64", "uint64", "bool", "float64", "str")
SIZE = 100
NUMFILES = 5
verbose = True


class DeprecatedIOTest(ArkoudaTest):
    """
        This test class is added to maintain coverage for deprecated save/read functionality.
        Once the functions covered here are deprecated this will file can be removed.
    """

    @classmethod
    def setUpClass(cls):
        super(DeprecatedIOTest, cls).setUpClass()
        DeprecatedIOTest.dep_test_base_tmp = "{}/dep_io_test".format(os.getcwd())
        io_util.get_directory(DeprecatedIOTest.dep_test_base_tmp)

    def _getCategorical(self, prefix: str = "string", size: int = 11) -> ak.Categorical:
        return ak.Categorical(ak.array(["{} {}".format(prefix, i) for i in range(1, size)]))

    def build_pandas_dataframe(self):
        df = pd.DataFrame(
            data={
                "Random_A": np.random.randint(0, 5, 5),
                "Random_B": np.random.randint(0, 5, 5),
                "Random_C": np.random.randint(0, 5, 5),
            },
            index=np.arange(5),
        )
        return df

    def build_arkouda_dataframe(self):
        pddf = self.build_pandas_dataframe()
        return ak.DataFrame(pddf)

    def testSaveAndLoadCategorical(self):
        """
        Test to save categorical to hdf5 and read it back successfully
        """
        num_elems = 51  # _getCategorical starts counting at 1, so the size is really off by one
        cat = self._getCategorical(size=num_elems)
        with self.assertRaises(ValueError):
            # Expect error for mode not being append or truncate
            cat.save("foo", dataset="bar", mode="not_allowed")

        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            dset_name = "categorical_array"  # name of categorical array

            # Test the save functionality & confirm via h5py
            cat.save(f"{tmp_dirname}/cat-save-test", dataset=dset_name)

            import h5py

            f = h5py.File(tmp_dirname + "/cat-save-test_LOCALE0000", mode="r")
            keys = set(f.keys())
            if (
                io.ARKOUDA_HDF5_FILE_METADATA_GROUP in keys
            ):  # Ignore the metadata group if it exists
                keys.remove(io.ARKOUDA_HDF5_FILE_METADATA_GROUP)
            self.assertEqual(len(keys), 5, "Expected 5 keys")
            self.assertSetEqual(
                set(f"categorical_array.{k}" for k in cat._get_components_dict().keys()), keys
            )
            f.close()

            # Now try to read them back with load_all
            x = ak.load_all(path_prefix=f"{tmp_dirname}/cat-save-test")
            self.assertTrue(dset_name in x)
            cat_from_hdf = x[dset_name]

            expected_categories = [f"string {i}" for i in range(1, num_elems)] + ["N/A"]

            # Note assertCountEqual asserts a and b have the same elements
            # in the same amount regardless of order
            self.assertCountEqual(cat_from_hdf.categories.to_list(), expected_categories)

            # Asserting the optional components and sizes are correct
            # for both constructors should be sufficient
            self.assertTrue(cat_from_hdf.segments is not None)
            self.assertTrue(cat_from_hdf.permutation is not None)
            print(f"==> cat_from_hdf.size:{cat_from_hdf.size}")
            self.assertEqual(cat_from_hdf.size, num_elems - 1)

    def testSaveAndLoadCategoricalMulti(self):
        """
        Test to build a pseudo dataframe with multiple
        categoricals, pdarrays, strings objects and successfully
        write/read it from HDF5
        """
        c1 = self._getCategorical(prefix="c1", size=51)
        c2 = self._getCategorical(prefix="c2", size=52)
        pda1 = ak.zeros(51)
        strings1 = ak.random_strings_uniform(9, 10, 52)

        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            df = {"cat1": c1, "cat2": c2, "pda1": pda1, "strings1": strings1}
            with pytest.deprecated_call():
                ak.save_all(df, f"{tmp_dirname}/cat-save-test")
            x = ak.load_all(path_prefix=f"{tmp_dirname}/cat-save-test")
            self.assertEqual(len(x.items()), 4)
            # Note assertCountEqual asserts a and b have the same
            # elements in the same amount regardless of order
            self.assertCountEqual(x["cat1"].categories.to_list(), c1.categories.to_list())
            self.assertCountEqual(x["cat2"].categories.to_list(), c2.categories.to_list())
            self.assertCountEqual(x["pda1"].to_list(), pda1.to_list())
            self.assertCountEqual(x["strings1"].to_list(), strings1.to_list())

    def test_df_save(self):
        i = list(range(3))
        c1 = [9, 7, 17]
        c2 = [2, 4, 6]
        df_dict = {"i": ak.array(i), "c_1": ak.array(c1), "c_2": ak.array(c2)}

        akdf = ak.DataFrame(df_dict)

        validation_df = pd.DataFrame(
            {
                "i": i,
                "c_1": c1,
                "c_2": c2,
            }
        )
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                akdf.save(f"{tmp_dirname}/testName", file_format="Parquet")

            ak_loaded = ak.DataFrame.load(f"{tmp_dirname}/testName")
            self.assertTrue(validation_df.equals(ak_loaded.to_pandas()))

            # test save with index true
            with pytest.deprecated_call():
                akdf.save(f"{tmp_dirname}/testName_with_index.pq", file_format="Parquet", index=True)
            self.assertEqual(len(glob.glob(f"{tmp_dirname}/testName_with_index*.pq")), ak.get_config()["numLocales"])

            # Test for df having seg array col
            df = ak.DataFrame({"a": ak.arange(10), "b": ak.segarray(ak.arange(10), ak.arange(10))})
            with pytest.deprecated_call():
                df.save(f"{tmp_dirname}/seg_test.h5")
            self.assertEqual(len(glob.glob(f"{tmp_dirname}/seg_test*.h5")), ak.get_config()["numLocales"])
            ak_loaded = ak.DataFrame.load(f"{tmp_dirname}/seg_test.h5")
            self.assertTrue(df.to_pandas().equals(ak_loaded.to_pandas()))

    def test_export_hdf(self):
        akdf = self.build_arkouda_dataframe()
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                akdf.save(f"{tmp_dirname}/ak_write")

            pddf = ak.export(f"{tmp_dirname}/ak_write", write_file=f"{tmp_dirname}/pd_from_ak.h5", index=True)
            self.assertEqual(len(glob.glob(f"{tmp_dirname}/pd_from_ak.h5")), 1)
            self.assertTrue(pddf.equals(akdf.to_pandas()))

            with self.assertRaises(RuntimeError):
                pddf = ak.export(f"{tmp_dirname}/foo.h5", write_file=f"{tmp_dirname}/pd_from_ak.h5", index=True)

    def test_export_parquet(self):
        akdf = self.build_arkouda_dataframe()
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            akdf.save(f"{tmp_dirname}/ak_write", file_format="Parquet")
            print(akdf.__repr__())

            pddf = ak.export(f"{tmp_dirname}/ak_write", write_file=f"{tmp_dirname}/pd_from_ak.parquet", index=True)
            print(pddf)
            self.assertEqual(len(glob.glob(f"{tmp_dirname}/pd_from_ak.parquet")), 1)
            self.assertTrue(pddf.equals(akdf.to_pandas()))

            with self.assertRaises(RuntimeError):
                pddf = ak.export(f"{tmp_dirname}/foo.h5", write_file=f"{tmp_dirname}/pd_from_ak.h5", index=True)

    def test_index_save(self):
        locale_count = ak.get_config()["numLocales"]
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            idx = ak.Index(ak.arange(5))
            with pytest.deprecated_call():
                idx.save(f"{tmp_dirname}/idx_file.h5")
            self.assertEqual(len(glob.glob(f"{tmp_dirname}/idx_file_*.h5")), locale_count)

    def testSaveStringsDataset(self):
        # Create, save, and load Strings dataset
        strings_array = ak.array(["testing string{}".format(num) for num in list(range(0, 25))])
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                strings_array.save("{}/strings-test".format(tmp_dirname), dataset="strings")
            r_strings_array = ak.load("{}/strings-test".format(tmp_dirname), dataset="strings")

            strings = strings_array.to_ndarray()
            strings.sort()
            r_strings = r_strings_array.to_ndarray()
            r_strings.sort()
            self.assertListEqual(strings.tolist(), r_strings.tolist())

            # Read a part of a saved Strings dataset from one hdf5 file
            r_strings_subset = ak.read(filenames="{}/strings-test_LOCALE0000".format(tmp_dirname))
            self.assertIsNotNone(r_strings_subset)
            self.assertTrue(isinstance(r_strings_subset[0], str))
            self.assertIsNotNone(
                ak.read(
                    filenames="{}/strings-test_LOCALE0000".format(tmp_dirname),
                    datasets="strings/values",
                )
            )
            with pytest.deprecated_call():
                self.assertIsNotNone(
                    ak.read(
                        filenames="{}/strings-test_LOCALE0000".format(tmp_dirname),
                        datasets="strings/segments",
                    )
                )

            # Repeat the test using the calc_string_offsets=True option to
            # have server calculate offsets array
            with pytest.deprecated_call():
                r_strings_subset = ak.read(
                    filenames=f"{tmp_dirname}/strings-test_LOCALE0000", calc_string_offsets=True
                )
            self.assertIsNotNone(r_strings_subset)
            self.assertTrue(isinstance(r_strings_subset[0], str))
            with pytest.deprecated_call():
                self.assertIsNotNone(
                    ak.read(
                        filenames=f"{tmp_dirname}/strings-test_LOCALE0000",
                        datasets="strings/values",
                        calc_string_offsets=True,
                    )
                )
            with pytest.deprecated_call():
                self.assertIsNotNone(
                    ak.read(
                        filenames=f"{tmp_dirname}/strings-test_LOCALE0000",
                        datasets="strings/segments",
                        calc_string_offsets=True,
                    )
                )

    def testStringsWithoutOffsets(self):
        """
        This tests both saving & reading a strings array without saving and reading the offsets to HDF5.
        Instead the offsets array will be derived from the values/bytes area by looking for null-byte
        terminator strings
        """
        strings_array = ak.array(["testing string{}".format(num) for num in list(range(0, 25))])
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                strings_array.save(
                    "{}/strings-test".format(tmp_dirname), dataset="strings", save_offsets=False
                )
            r_strings_array = ak.load(
                "{}/strings-test".format(tmp_dirname), dataset="strings", calc_string_offsets=True
            )
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
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                strings.save("{}/strings-test".format(tmp_dirname), dataset="strings")

            n_strings = strings.to_ndarray()
            n_strings.sort()
            r_strings = ak.load("{}/strings-test".format(tmp_dirname), dataset="strings").to_ndarray()
            r_strings.sort()

            self.assertListEqual(n_strings.tolist(), r_strings.tolist())

    def testSaveMixedStringsDataset(self):
        strings_array = ak.array(["string {}".format(num) for num in list(range(0, 25))])
        m_floats = ak.array([x / 10.0 for x in range(0, 10)])
        m_ints = ak.array(list(range(0, 10)))
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                ak.save_all(
                    {"m_strings": strings_array, "m_floats": m_floats, "m_ints": m_ints},
                    "{}/multi-type-test".format(tmp_dirname),
                )
            r_mixed = ak.load_all("{}/multi-type-test".format(tmp_dirname))

            self.assertListEqual(
                np.sort(strings_array.to_ndarray()).tolist(),
                np.sort(r_mixed["m_strings"].to_ndarray()).tolist(),
            )
            self.assertIsNotNone(r_mixed["m_floats"])
            self.assertIsNotNone(r_mixed["m_ints"])

            r_floats = ak.sort(ak.load("{}/multi-type-test".format(tmp_dirname), dataset="m_floats"))
            self.assertListEqual(m_floats.to_list(), r_floats.to_list())

            r_ints = ak.sort(ak.load("{}/multi-type-test".format(tmp_dirname), dataset="m_ints"))
            self.assertListEqual(m_ints.to_list(), r_ints.to_list())

            strings = strings_array.to_ndarray()
            strings.sort()
            r_strings = ak.load(
                "{}/multi-type-test".format(tmp_dirname), dataset="m_strings"
            ).to_ndarray()
            r_strings.sort()

            self.assertListEqual(strings.tolist(), r_strings.tolist())

    def testAppendStringsDataset(self):
        strings_array = ak.array(["string {}".format(num) for num in list(range(0, 25))])
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                strings_array.save("{}/append-strings-test".format(tmp_dirname), dataset="strings")
            with pytest.deprecated_call():
                strings_array.save(
                    "{}/append-strings-test".format(tmp_dirname), dataset="strings-dupe", mode="append"
                )

            r_strings = ak.load("{}/append-strings-test".format(tmp_dirname), dataset="strings")
            r_strings_dupe = ak.load(
                "{}/append-strings-test".format(tmp_dirname), dataset="strings-dupe"
            )
            self.assertListEqual(r_strings.to_list(), r_strings_dupe.to_list())

    def testAppendMixedStringsDataset(self):
        strings_array = ak.array(["string {}".format(num) for num in list(range(0, 25))])
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                strings_array.save("{}/append-multi-type-test".format(tmp_dirname), dataset="m_strings")
            m_floats = ak.array([x / 10.0 for x in range(0, 10)])
            m_ints = ak.array(list(range(0, 10)))
            with pytest.deprecated_call():
                ak.save_all(
                    {"m_floats": m_floats, "m_ints": m_ints},
                    "{}/append-multi-type-test".format(tmp_dirname),
                    mode="append",
                )
            r_mixed = ak.load_all("{}/append-multi-type-test".format(tmp_dirname))

            self.assertIsNotNone(r_mixed["m_floats"])
            self.assertIsNotNone(r_mixed["m_ints"])

            r_floats = ak.sort(
                ak.load("{}/append-multi-type-test".format(tmp_dirname), dataset="m_floats")
            )
            r_ints = ak.sort(
                ak.load("{}/append-multi-type-test".format(tmp_dirname), dataset="m_ints")
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
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            prefix = "{}/strict-type-test".format(tmp_dirname)
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
                with pytest.deprecated_call():
                    ak.read(prefix + "*")

            with pytest.deprecated_call():
                a = ak.read(prefix + "*", strictTypes=False)
            self.assertListEqual(a["integers"].to_list(), np.arange(len(inttypes) * N).tolist())
            self.assertTrue(
                np.allclose(a["floats"].to_ndarray(), np.arange(len(floattypes) * N, dtype=np.float64))
            )

    def testSmallArrayToHDF5(self):
        a1 = ak.array([1])
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                a1.save(f"{tmp_dirname}/small_numeric", dataset="a1")
            # Now load it back in
            a2 = ak.load(f"{tmp_dirname}/small_numeric", dataset="a1")
            self.assertEqual(str(a1), str(a2))

    # This tests small array corner cases on multi-locale environments
    def testSmallStringArrayToHDF5(self):
        a1 = ak.array(["ab", "cd"])
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                a1.save(f"{tmp_dirname}/small_string_array", dataset="a1")
            # Now load it back in
            a2 = ak.load(f"{tmp_dirname}/small_string_array", dataset="a1")
            self.assertEqual(str(a1), str(a2))

        # Test a single string
        b1 = ak.array(["123456789"])
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                b1.save(f"{tmp_dirname}/single_string", dataset="b1")
            # Now load it back in
            b2 = ak.load(f"{tmp_dirname}/single_string", dataset="b1")
            self.assertEqual(str(b1), str(b2))

    def testUint64ToFromHDF5(self):
        """
        Test our ability to read/write uint64 to HDF5
        """
        npa1 = np.array(
            [18446744073709551500, 18446744073709551501, 18446744073709551502], dtype=np.uint64
        )
        pda1 = ak.array(npa1)
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                pda1.save(f"{tmp_dirname}/small_numeric", dataset="pda1")
            # Now load it back in
            pda2 = ak.load(f"{tmp_dirname}/small_numeric", dataset="pda1")
            self.assertEqual(str(pda1), str(pda2))
            self.assertEqual(18446744073709551500, pda2[0])
            self.assertListEqual(pda2.to_list(), npa1.tolist())

    def testHdfUnsanitizedNames(self):
        # Test when quotes are part of the dataset name
        my_arrays = {'foo"0"': ak.arange(100), 'bar"': ak.arange(100)}
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                ak.save_all(my_arrays, f"{tmp_dirname}/bad_dataset_names")
            with pytest.deprecated_call():
                ak.read(f"{tmp_dirname}/bad_dataset_names*")

    def test_multi_dim_rdwr(self):
        arr = ak.ArrayView(ak.arange(27), ak.array([3, 3, 3]))
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                ak.write_hdf5_multi_dim(
                    arr, tmp_dirname + "/multi_dim_test", "MultiDimObj", mode="append"
                )
            # load data back
            with pytest.deprecated_call():
                read_arr = ak.read(tmp_dirname + "/multi_dim_test*", "MultiDimObj")
            self.assertTrue(np.array_equal(arr.to_ndarray(), read_arr.to_ndarray()))

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

            with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
                with pytest.deprecated_call():
                    ak_arr.save_parquet(f"{tmp_dirname}/pq_testcorrect", "my-dset")
                with pytest.deprecated_call():
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

            with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
                per_arr = int(adjusted_size / NUMFILES)
                for i in range(NUMFILES):
                    test_arrs.append(elems[(i * per_arr): (i * per_arr) + per_arr])
                    with pytest.deprecated_call():
                        test_arrs[i].save_parquet(f"{tmp_dirname}/pq_test{i:04d}", "test-dset")
                with pytest.deprecated_call():
                    pq_arr = ak.read(f"{tmp_dirname}/pq_test*", "test-dset")
                self.assertListEqual(elems.to_list(), pq_arr.to_list())

    def test_wrong_dset_name(self):
        ak_arr = ak.randint(0, 2**32, SIZE)
        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                ak_arr.save_parquet(f"{tmp_dirname}/pq_test", "test-dset-name")

            with self.assertRaises(RuntimeError):
                with pytest.deprecated_call():
                    ak.read(f"{tmp_dirname}/pq_test*", "wrong-dset-name")

            with self.assertRaises(ValueError):
                with pytest.deprecated_call():
                    ak.read(f"{tmp_dirname}/pq_test*", ["test-dset-name", "wrong-dset-name"])

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
            with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
                with pytest.deprecated_call():
                    a.save_parquet(f"{tmp_dirname}/pq_test", "test-dset")
                with pytest.deprecated_call():
                    ak_res = ak.read(f"{tmp_dirname}/pq_test*", "test-dset")
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

            with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
                with pytest.deprecated_call():
                    ak_arr.save_parquet(f"{tmp_dirname}/pq_testdset", "TEST_DSET")

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

        with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
            with pytest.deprecated_call():
                base_dset.save_parquet(f"{tmp_dirname}/pq_testcorrect", "base-dset")
            for key in ak_dict:
                ak_dict[key].save_parquet(f"{tmp_dirname}/pq_testcorrect", key, mode="append")

            with pytest.deprecated_call():
                ak_vals = ak.read(f"{tmp_dirname}/pq_testcorrect*")

            for key in ak_dict:
                self.assertListEqual(ak_vals[key].to_list(), ak_dict[key].to_list())

    def test_null_strings(self):
        datadir = "resources/parquet-testing"
        basename = "null-strings.parquet"
        expected = ["first-string", "", "string2", "", "third", "", ""]

        filename = os.path.join(datadir, basename)
        with pytest.deprecated_call():
            res = ak.read(filename)

        self.assertListEqual(expected, res.to_list())

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

            with tempfile.TemporaryDirectory(dir=DeprecatedIOTest.dep_test_base_tmp) as tmp_dirname:
                with pytest.deprecated_call():
                    ak_arr.save(f"{tmp_dirname}/pq_testcorrect", "my-dset", mode="append", file_format="parquet")
                with pytest.deprecated_call():
                    pq_arr = ak.read(f"{tmp_dirname}/pq_testcorrect*", "my-dset")

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
                with pytest.deprecated_call():
                    data = ak.read(filename, datasets=columns)
            else:
                # Since delta encoding is not supported, the columns in
                # this file should raise an error and not crash the server
                with self.assertRaises(RuntimeError):
                    with pytest.deprecated_call():
                        data = ak.read(filename, datasets=columns)