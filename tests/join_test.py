import numpy as np

from base_test import ArkoudaTest
from context import arkouda as ak

'''
Encapsulates a variety of arkouda join_on_eq_with_dt test cases.
'''
class JoinTest(ArkoudaTest):

    def setUp(self):
        ArkoudaTest.setUp(self)
        self.N = 1000
        self.a1 = ak.ones(self.N,dtype=np.int64)
        self.a2 = ak.arange(0,self.N,1)
        self.t1 = self.a1
        self.t2 = self.a1 * 10
        self.dt = 10
        ak.verbose = False

    def test_join_on_eq_with_true_dt(self):
        I,J = ak.join_on_eq_with_dt(self.a2,self.a1,self.t1,self.t2,self.dt,"true_dt")
        nl = ak.get_config()['numLocales']
        self.assertEqual(self.N//nl, I.size)
        self.assertEqual(self.N//nl, J.size)
               
    def test_join_on_eq_with_true_dt_with_result_limit(self):
        nl = ak.get_config()['numLocales']
        lim = (self.N + nl) * self.N
        res_size = self.N * self.N
        I,J = ak.join_on_eq_with_dt(self.a1,self.a1,self.a1,self.a1,self.dt,"true_dt",result_limit=lim)
        self.assertEqual(res_size, I.size)
        self.assertEqual(res_size, J.size)

    def test_join_on_eq_with_abs_dt(self):
        I,J = ak.join_on_eq_with_dt(self.a2,self.a1,self.t1,self.t2,self.dt,"abs_dt")
        nl = ak.get_config()['numLocales']
        self.assertEqual(self.N//nl, I.size)
        self.assertEqual(self.N//nl, J.size)

    def test_join_on_eq_with_pos_dt(self):
        I,J = ak.join_on_eq_with_dt(self.a2,self.a1,self.t1,self.t2,self.dt,"pos_dt")
        nl = ak.get_config()['numLocales']
        self.assertEqual(self.N//nl, I.size)
        self.assertEqual(self.N//nl, J.size)

    def test_join_on_eq_with_abs_dt_outside_window(self):
        '''
        Should get 0 answers because N^2 matches but 0 within dt window 
        '''
        dt = 8
        I,J = ak.join_on_eq_with_dt(self.a1,self.a1,self.t1,self.t1*10,dt,"abs_dt")
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)

        I,J = ak.join_on_eq_with_dt(self.a2,self.a1,self.t1,self.t2,dt,"abs_dt")
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)

    def test_join_on_eq_with_pos_dt_outside_window(self):
        '''
        Should get 0 answers because N matches but 0 within dt window
        '''
        dt = 8
        I,J = ak.join_on_eq_with_dt(self.a2,self.a1,self.t1,self.t2,dt,"pos_dt")
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)
        
        dt = np.int64(8)
        I,J = ak.join_on_eq_with_dt(self.a2,self.a1,self.t1,self.t2,dt,"pos_dt")
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)
        
        I,J = ak.join_on_eq_with_dt(self.a2,self.a1,self.t1,self.t2,dt,"pos_dt", int(0))
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)

    def test_gen_ranges(self):
        start = ak.array([0, 10, 20])
        end = ak.array([10, 20, 30])

        segs, ranges = ak.join.gen_ranges(start, end)
        self.assertListEqual(segs.to_ndarray().tolist(), [0, 10, 20])
        self.assertListEqual(ranges.to_ndarray().tolist(), list(range(30)))

        with self.assertRaises(ValueError):
            segs, ranges = ak.join.gen_ranges(ak.array([11, 12, 41]), end)

    def test_inner_join(self):
        left = ak.arange(10)
        right = ak.array([0, 5, 3, 3, 4, 6, 7, 9, 8, 1])

        l, r = ak.join.inner_join(left, right)
        self.assertTrue((left[l] == right[r]).all())

        with self.assertRaises(ValueError):
            l, r = ak.join.inner_join(left, right, wherefunc=ak.unique)

        with self.assertRaises(ValueError):
            l, r = ak.join.inner_join(left, right, wherefunc=ak.intersect1d)

        with self.assertRaises(ValueError):
            l, r = ak.join.inner_join(left, right, wherefunc=ak.intersect1d, whereargs=(ak.arange(5), ak.arange(10)))

        with self.assertRaises(ValueError):
            l, r = ak.join.inner_join(left, right, wherefunc=ak.intersect1d, whereargs=(ak.arange(10), ak.arange(5)))

    def test_lookup(self):
        keys = ak.arange(5)
        values = 10*keys
        args = ak.array([5, 3, 1, 4, 2, 3, 1, 0])
        ans = np.array([-1, 30, 10, 40, 20, 30, 10, 0])
        # Simple lookup with int keys
        # Also test shortcut for unique-ordered keys
        res = ak.lookup(keys, values, args, fillvalue=-1, keys_from_unique=True)
        self.assertTrue((res.to_ndarray() == ans).all())
        # Compound lookup with (str, int) keys
        res2 = ak.lookup((ak.cast(keys, ak.str_), keys), values, (ak.cast(args, ak.str_), args), fillvalue=-1)
        self.assertTrue((res2.to_ndarray() == ans).all())
        # Keys not in uniqued order
        res3 = ak.lookup(keys[::-1], values[::-1], args, fillvalue=-1)
        self.assertTrue((res3.to_ndarray() == ans).all())
        # Non-unique keys should raise error
        with self.assertRaises(ak.NonUniqueError):
            keys = ak.arange(10) % 5
            values = 10 * keys
            ak.lookup(keys, values, args)

    def test_error_handling(self):
        """
        Tests error TypeError and ValueError handling
        """
        with self.assertRaises(TypeError):
            ak.join_on_eq_with_dt([list(range(0,11))],
                                  self.a1,self.t1,self.t2,8,"pos_dt")
        with self.assertRaises(TypeError):
            ak.join_on_eq_with_dt([self.a1, list(range(0,11))],
                                  self.t1,self.t2,8,"pos_dt")
        with self.assertRaises(TypeError):
            ak.join_on_eq_with_dt([self.a1, self.a1, list(range(0,11))],
                                  self.t2,8,"pos_dt")
        with self.assertRaises(TypeError):
            ak.join_on_eq_with_dt([self.a1, self.a1, self.t1,
                                  list(range(0,11))],8,"pos_dt")
        with self.assertRaises(TypeError):
            ak.join_on_eq_with_dt(self.a1,
                                  self.a1,self.t1,self.t2,'8',"pos_dt")
        with self.assertRaises(ValueError):
            ak.join_on_eq_with_dt(self.a1,self.a1,self.t1,self.t1*10,8,"ab_dt")
        with self.assertRaises(ValueError):
            ak.join_on_eq_with_dt(self.a1,self.a1,self.t1,self.t1*10,8,"abs_dt",-1)            
