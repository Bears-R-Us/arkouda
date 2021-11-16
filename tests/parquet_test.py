import glob, os
from context import arkouda as ak
from base_test import ArkoudaTest
import pytest

SIZE = 50
NUMFILES = 5
verbose = True

def compare_values(arr1, arr2):
    if (arr1 != arr2).all():
        print("Arrays do not match")
        return 1
    return 0

def run_parquet_test(verbose=True):
    ak_arr = ak.randint(0, 2**32, SIZE)
    ak_arr.save_parquet("pq_testcorrect", "my-dset")
    pq_arr = ak.read_parquet("pq_testcorrect*", "my-dset")
    # get the dset from the dictionary in multi-locale cases
    for f in glob.glob('pq_test*'):
        os.remove(f)
    return compare_values(ak_arr, pq_arr)

def run_parquet_multi_file_test(verbose=True):
    adjusted_size = int(SIZE/NUMFILES)*NUMFILES
    failures = 0
    test_arrs = []
    for i in range(NUMFILES):
        test_arrs.append(ak.randint(0, 2**32, int(adjusted_size/NUMFILES)))
        test_arrs[i].save_parquet("pq_test" + str(i), "test-dset")

    pq_arr = ak.read_parquet("pq_test*", "test-dset")
    if len(pq_arr) != adjusted_size:
        print('Size of array read in was', str(len(pq_arr)), 'but should be', adjusted_size)
        failures += 1

    for i in range(NUMFILES):
        sz = len(test_arrs[i])
        failures += compare_values(test_arrs[i], pq_arr[(i*sz):(i*sz)+sz])
    for f in glob.glob('pq_test*'):
        os.remove(f)
    return failures



@pytest.mark.skipif(not os.getenv('ARKOUDA_SERVER_PARQUET_SUPPORT'), reason="No parquet support")
class ParquetTest(ArkoudaTest):
    def test_parquet(self):
        self.assertEqual(run_parquet_test(), 0)

    def test_multi_file(self):
        self.assertEqual(run_parquet_multi_file_test(), 0)
