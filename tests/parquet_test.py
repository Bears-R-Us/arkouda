import glob, os
from context import arkouda as ak
from base_test import ArkoudaTest
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
            self.assertTrue((ak_arr ==  pq_arr).all())

    def test_multi_file(self):
        adjusted_size = int(SIZE/NUMFILES)*NUMFILES
        test_arrs = []
        for i in range(NUMFILES):
            test_arrs.append(ak.randint(0, 2**32, int(adjusted_size/NUMFILES)))
            test_arrs[i].save_parquet("pq_test" + str(i), "test-dset")

        pq_arr = ak.read_parquet("pq_test*", "test-dset")
        self.assertTrue(len(pq_arr) == adjusted_size)

        for i in range(NUMFILES):
            sz = len(test_arrs[i])
            self.assertTrue((test_arrs[i] == pq_arr[(i*sz):(i*sz)+sz]).all())
        for f in glob.glob('pq_test*'):
            os.remove(f)
