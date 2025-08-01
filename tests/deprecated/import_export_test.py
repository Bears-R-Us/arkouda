import glob
import os
import tempfile
from shutil import rmtree

import numpy as np
import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda.pandas import io, io_util


class ImportExportTest(ArkoudaTest):
    @classmethod
    def setUpClass(cls):
        super(ImportExportTest, cls).setUpClass()
        ImportExportTest.ie_test_base_tmp = "{}/import_export_test".format(os.getcwd())
        io_util.get_directory(ImportExportTest.ie_test_base_tmp)

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

    def get_locale_count(self):
        return ak.get_config()["numLocales"]

    def test_import_hdf(self):
        locales = self.get_locale_count()
        f_base = f"{os.getcwd()}/import_export_test"
        # make directory to save to so pandas read works
        if not os.path.exists(f_base):
            os.mkdir(f_base)

        pddf = self.build_pandas_dataframe()
        pddf.to_hdf(f"{f_base}/table.h5", "dataframe", format="Table", mode="w")
        akdf = ak.import_data(f"{f_base}/table.h5", write_file=f"{f_base}/ak_table.h5")
        self.assertEqual(len(glob.glob(f"{f_base}/ak_table_*.h5")), locales)
        self.assertTrue(pddf.equals(akdf.to_pandas()))

        pddf.to_hdf(
            f"{f_base}/table_columns.h5", "dataframe", format="Table", data_columns=True, mode="w"
        )
        akdf = ak.import_data(f"{f_base}/table_columns.h5", write_file=f"{f_base}/ak_table_columns.h5")
        self.assertEqual(len(glob.glob(f"{f_base}/ak_table_columns_*.h5")), locales)
        self.assertTrue(pddf.equals(akdf.to_pandas()))

        pddf.to_hdf(f"{f_base}/fixed.h5", "dataframe", format="fixed", data_columns=True, mode="w")
        akdf = ak.import_data(f"{f_base}/fixed.h5", write_file=f"{f_base}/ak_fixed.h5")
        self.assertEqual(len(glob.glob(f"{f_base}/ak_fixed_*.h5")), locales)
        self.assertTrue(pddf.equals(akdf.to_pandas()))

        with self.assertRaises(FileNotFoundError):
            akdf = ak.import_data(f"{f_base}/foo.h5", write_file=f"{f_base}/ak_fixed.h5")
        with self.assertRaises(RuntimeError):
            akdf = ak.import_data(f"{f_base}/*.h5", write_file=f"{f_base}/ak_fixed.h5")

        # clean up test files
        rmtree(f_base)

    def test_export_hdf(self):
        akdf = self.build_arkouda_dataframe()
        with tempfile.TemporaryDirectory(dir=ImportExportTest.ie_test_base_tmp) as tmp_dirname:
            akdf.to_hdf(f"{tmp_dirname}/ak_write")

            pddf = ak.export(f"{tmp_dirname}/ak_write", write_file=f"{tmp_dirname}/pd_from_ak.h5", index=True)
            self.assertEqual(len(glob.glob(f"{tmp_dirname}/pd_from_ak.h5")), 1)
            self.assertTrue(pddf.equals(akdf.to_pandas()))

            with self.assertRaises(RuntimeError):
                pddf = ak.export(f"{tmp_dirname}/foo.h5", write_file=f"{tmp_dirname}/pd_from_ak.h5", index=True)

    def test_import_parquet(self):
        locales = self.get_locale_count()
        f_base = f"{os.getcwd()}/import_export_test"
        # make directory to save to so pandas read works
        if not os.path.exists(f_base):
            os.mkdir(f_base)

        pddf = self.build_pandas_dataframe()
        pddf.to_parquet(f"{f_base}/table.parquet")
        akdf = ak.import_data(f"{f_base}/table.parquet", write_file=f"{f_base}/ak_table.parquet")
        self.assertEqual(len(glob.glob(f"{f_base}/ak_table_*.parquet")), locales)
        self.assertTrue(pddf.equals(akdf.to_pandas()))

        # clean up test files
        rmtree(f_base)

    def test_export_parquet(self):
        akdf = self.build_arkouda_dataframe()
        with tempfile.TemporaryDirectory(dir=ImportExportTest.ie_test_base_tmp) as tmp_dirname:
            akdf.to_parquet(f"{tmp_dirname}/ak_write")

            pddf = ak.export(f"{tmp_dirname}/ak_write", write_file=f"{tmp_dirname}/pd_from_ak.parquet", index=True)
            self.assertEqual(len(glob.glob(f"{tmp_dirname}/pd_from_ak.parquet")), 1)
            self.assertTrue(pddf[akdf.columns.values].equals(akdf.to_pandas()))

            with self.assertRaises(RuntimeError):
                pddf = ak.export(f"{tmp_dirname}/foo.h5", write_file=f"{tmp_dirname}/pd_from_ak.h5", index=True)
