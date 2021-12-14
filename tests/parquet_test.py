import glob, os
from context import arkouda as ak
from base_test import ArkoudaTest
import numpy as np
import pytest

SIZE = 100
NUMFILES = 5
verbose = True

@pytest.mark.skipif(not os.getenv('ARKOUDA_SERVER_PARQUET_SUPPORT'), reason="No parquet support")
class ParquetTest(ArkoudaTest):
    def test_parquet(self):
        ak_arr = ak.randint(0, 2**32, SIZE)
        ak_arr.save_parquet("pq_testcorrect", "my-dset")
        pq_arr = ak.read_parquet("pq_testcorrect*", "my-dset")
        self.assertTrue((ak_arr ==  pq_arr).all())
        
        for f in glob.glob('pq_test*'):
            os.remove(f)

    def test_multi_file(self):
        adjusted_size = int(SIZE/NUMFILES)*NUMFILES
        test_arrs = []
        elems = ak.randint(0, 2**32, adjusted_size)
        per_arr = int(adjusted_size/NUMFILES)
        for i in range(NUMFILES):
            test_arrs.append(elems[(i*per_arr):(i*per_arr)+per_arr])
            test_arrs[i].save_parquet("pq_test" + str(i), "test-dset")

        pq_arr = ak.read_parquet("pq_test*", "test-dset")

        self.assertTrue((elems == pq_arr).all())

        for f in glob.glob('pq_test*'):
            os.remove(f)

    def test_wrong_dset_name(self):
        ak_arr = ak.randint(0, 2**32, SIZE)
        ak_arr.save_parquet("pq_test", "test-dset-name")
        
        with self.assertRaises(RuntimeError) as cm:
            ak.read_parquet("pq_test*", "wrong-dset-name")
        self.assertIn("wrong-dset-name does not exist in file", cm.exception.args[0])

        for f in glob.glob("pq_test*"):
            os.remove(f)
        
