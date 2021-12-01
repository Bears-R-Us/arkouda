import glob, os
from context import arkouda as ak
from base_test import ArkoudaTest
import numpy as np
import pytest

SIZE = 50
NUMFILES = 5
verbose = True

@pytest.mark.skipif(not os.getenv('ARKOUDA_SERVER_PARQUET_SUPPORT'), reason="No parquet support")
class ParquetTest(ArkoudaTest):
    def test_parquet(self):
        ak_arr = ak.randint(0, 2**32, SIZE)
        ak_arr.save_parquet("pq_testcorrect", "my-dset")
        pq_arr = ak.read_parquet("pq_testcorrect*", "my-dset")
        # get the dset from the dictionary in multi-locale cases
        for f in glob.glob('pq_test*'):
            os.remove(f)
            a = ak_arr.to_ndarray().sort()
            b = pq_arr.to_ndarray().sort()
            self.assertTrue(a ==  b)

    def test_multi_file(self):
        adjusted_size = int(SIZE/NUMFILES)*NUMFILES
        test_arrs = []
        elems = ak.randint(0, 2**32, adjusted_size)
        per_arr = int(adjusted_size/NUMFILES)
        for i in range(NUMFILES):
            test_arrs.append(elems[(i*per_arr):(i*per_arr)+per_arr])
            test_arrs[i].save_parquet("pq_test" + str(i), "test-dset")

        a = elems.to_ndarray()
        a.sort()
        pq_arr = ak.read_parquet("pq_test*", "test-dset")
        b = pq_arr.to_ndarray()
        b.sort()

        self.assertTrue((a == b).all())

        for f in glob.glob('pq_test*'):
            os.remove(f)
