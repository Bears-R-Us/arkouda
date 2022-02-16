import glob, os
from context import arkouda as ak
from base_test import ArkoudaTest
import numpy as np
import pytest

TYPES = ('int64', 'uint64')
SIZE = 100
NUMFILES = 5
verbose = True

def read_write_test(dtype):
    if dtype == 'int64':
        ak_arr = ak.randint(0, 2**32, SIZE)
    elif dtype =='uint64':
        ak_arr = ak.randint(0, 2**32, SIZE, dtype=ak.uint64)
        
    ak_arr.save_parquet("pq_testcorrect", "my-dset")
    pq_arr = ak.read_parquet("pq_testcorrect*", "my-dset")
    
    for f in glob.glob('pq_test*'):
        os.remove(f)

    return ak_arr, pq_arr

def read_write_multi_test(dtype):
    adjusted_size = int(SIZE/NUMFILES)*NUMFILES
    test_arrs = []
    if dtype == 'int64':
        elems = ak.randint(0, 2**32, adjusted_size)
    elif dtype == 'uint64':
        elems = ak.randint(0, 2**32, adjusted_size, dtype=ak.uint64)
        
    per_arr = int(adjusted_size/NUMFILES)
    for i in range(NUMFILES):
        test_arrs.append(elems[(i*per_arr):(i*per_arr)+per_arr])
        test_arrs[i].save_parquet(f"pq_test{i:04d}", "test-dset")

    pq_arr = ak.read_parquet("pq_test*", "test-dset")
    
    for f in glob.glob('pq_test*'):
        os.remove(f)

    return elems, pq_arr
    

@pytest.mark.skipif(not os.getenv('ARKOUDA_SERVER_PARQUET_SUPPORT'), reason="No parquet support")
class ParquetTest(ArkoudaTest):
    def test_parquet(self):
        for dtype in TYPES:
            (ak_arr, pq_arr) = read_write_test(dtype)
            self.assertTrue((ak_arr ==  pq_arr).all())

    def test_multi_file(self):
        for dtype in TYPES:
            (ak_arr, pq_arr) = read_write_multi_test(dtype)
            self.assertTrue((ak_arr ==  pq_arr).all())

    def test_wrong_dset_name(self):
        ak_arr = ak.randint(0, 2**32, SIZE)
        ak_arr.save_parquet("pq_test", "test-dset-name")
        
        with self.assertRaises(RuntimeError) as cm:
            ak.read_parquet("pq_test*", "wrong-dset-name")

        for f in glob.glob("pq_test*"):
            os.remove(f)

    def test_max_read_write(self):
        for dtype in TYPES:
            if dtype == 'int64':
                val = np.iinfo(np.int64).max
            elif dtype == 'uint64':
                val = np.iinfo(np.uint64).max
            a = ak.array([val])
            a.save_parquet("pq_test", 'test-dset')
            ak_res = ak.read_parquet("pq_test*", 'test-dset')
            self.assertTrue(ak_res[0] == val)

            for f in glob.glob('pq_test*'):
                os.remove(f)

