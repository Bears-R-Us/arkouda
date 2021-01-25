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
        
        I,J = ak.join_on_eq_with_dt(self.a2,self.a1,self.t1,self.t2,dt,"pos_dt", np.int(0))
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)
        
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
