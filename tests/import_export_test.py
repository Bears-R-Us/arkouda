import numpy as np
import pandas as pd
import os
from shutil import rmtree
import glob

import pytest

from base_test import ArkoudaTest
from context import arkouda as ak

class DataFrameTest(ArkoudaTest):
    def build_pandas_dataframe(self):
        df = pd.DataFrame(data={
            'Random_A': np.random.randint(0, 5, 5),
            'Random_B': np.random.randint(0, 5, 5),
            'Random_C': np.random.randint(0, 5, 5)
        }, index=np.arange(5))
        return df

    def build_arkouda_dataframe(self):
        pddf = self.build_pandas_dataframe()
        return ak.DataFrame(pddf)

    def test_import_hdf(self):
        f_base = f"{os.getcwd()}/import_export_test"
        # make directory to save to so pandas read works
        os.mkdir(f_base)

        pddf = self.build_pandas_dataframe()
        pddf.to_hdf(f"{f_base}/table.h5", "dataframe", format="Table", mode="w")
        akdf = ak.import_data(f"{f_base}/table.h5", write_file=f"{f_base}/ak_table.h5",
                              return_obj=True)
        self.assertTrue(len(glob.glob(f"{f_base}/ak_table_*.h5")) == 1)
        self.assertTrue(pddf.equals(akdf.to_pandas()))

        pddf.to_hdf(f"{f_base}/table_columns.h5", "dataframe", format="Table", data_columns=True, mode="w")
        akdf = ak.import_data(f"{f_base}/table_columns.h5", write_file=f"{f_base}/ak_table_columns.h5",
                              return_obj=True)
        self.assertTrue(len(glob.glob(f"{f_base}/ak_table_columns_*.h5")) == 1)
        self.assertTrue(pddf.equals(akdf.to_pandas()))

        pddf.to_hdf(f"{f_base}/fixed.h5", "dataframe", format="fixed", data_columns=True, mode="w")
        akdf = ak.import_data(f"{f_base}/fixed.h5", write_file=f"{f_base}/ak_fixed.h5",
                              return_obj=True)
        self.assertTrue(len(glob.glob(f"{f_base}/ak_fixed_*.h5")) == 1)
        self.assertTrue(pddf.equals(akdf.to_pandas()))

        with self.assertRaises(FileNotFoundError):
            akdf = ak.import_data(f"{f_base}/foo.h5", write_file=f"{f_base}/ak_fixed.h5",
                                  return_obj=True)
        with self.assertRaises(RuntimeError):
            akdf = ak.import_data(f"{f_base}/*.h5", write_file=f"{f_base}/ak_fixed.h5",
                                  return_obj=True)

        # clean up test files
        rmtree(f_base)

    def test_export_hdf(self):
        f_base = f"{os.getcwd()}/import_export_test"
        # make directory to save to so pandas read works
        os.mkdir(f_base)

        akdf = self.build_arkouda_dataframe()
        akdf.save_table(f"{f_base}/ak_write")

        pddf = ak.export(f"{f_base}/ak_write", write_file=f"{f_base}/pd_from_ak.h5",
                         return_obj=True, index=True)
        self.assertTrue(len(glob.glob(f"{f_base}/pd_from_ak.h5")) == 1)
        self.assertTrue(pddf.equals(akdf.to_pandas()))

        with self.assertRaises(RuntimeError):
            pddf = ak.export(f"{f_base}/foo.h5", write_file=f"{f_base}/pd_from_ak.h5",
                             return_obj=True, index=True)

        # clean up test files
        rmtree(f_base)

    @pytest.mark.skipif(not os.getenv('ARKOUDA_SERVER_PARQUET_SUPPORT'), reason="No parquet support")
    def test_import_parquet(self):
        f_base = f"{os.getcwd()}/import_export_test"
        # make directory to save to so pandas read works
        os.mkdir(f_base)

        pddf = self.build_pandas_dataframe()
        pddf.to_parquet(f"{f_base}/table.parquet")
        akdf = ak.import_data(f"{f_base}/table.parquet", write_file=f"{f_base}/ak_table.parquet",
                              return_obj=True)
        self.assertTrue(len(glob.glob(f"{f_base}/ak_table_LOCALE0000.parquet")) == 1)
        self.assertTrue(pddf.equals(akdf.to_pandas()))

        # clean up test files
        rmtree(f_base)

    @pytest.mark.skipif(not os.getenv('ARKOUDA_SERVER_PARQUET_SUPPORT'), reason="No parquet support")
    def test_export_parquet(self):
        f_base = f"{os.getcwd()}/import_export_test"
        # make directory to save to so pandas read works
        os.mkdir(f_base)

        akdf = self.build_arkouda_dataframe()
        akdf.save_table(f"{f_base}/ak_write", file_format="Parquet")
        print(akdf.__repr__())

        pddf = ak.export(f"{f_base}/ak_write", write_file=f"{f_base}/pd_from_ak.parquet",
                         return_obj=True, index=True)
        print(pddf)
        self.assertTrue(len(glob.glob(f"{f_base}/pd_from_ak.parquet")) == 1)
        self.assertTrue(pddf.equals(akdf.to_pandas()))

        with self.assertRaises(RuntimeError):
            pddf = ak.export(f"{f_base}/foo.h5", write_file=f"{f_base}/pd_from_ak.h5",
                             return_obj=True, index=True)

        # clean up test files
        rmtree(f_base)
