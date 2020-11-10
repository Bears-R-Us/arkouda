from context import arkouda as ak
from base_test import ArkoudaTest

SIZE = 10

def run_test() -> int:
    a = ak.ones(SIZE,dtype=ak.float64)
    b = ak.ones(SIZE,dtype=ak.int64)
    d = a.register('test_float64')
    e = b.register('test_int64')
    ak.clear()

    # will get decremented to 0 if test is successful
    result = 2

    try:
        a+b
    except:
        result -= 1

    try:
        d+e
        result -= 1
    except:
        result = 100

    d.unregister()
    e.unregister()
    return result

class ClearTest(ArkoudaTest):
    def test_clear(self):
        '''
        Executes run_test and asserts whether there are any errors
        
        :return: None
        :raise: AssertionError if there are any errors encountered in run_test for set operations
        '''
        self.assertEqual(0, run_test())
