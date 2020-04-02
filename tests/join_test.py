import importlib
import numpy as np
import math
import gc
import sys

from base_test import ArkoudaTest
from context import arkouda as ak

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
        self.assertEqual(self.N, I.size)
        self.assertEqual(self.N, J.size)
               
    def test_join_on_eq_with_true_dt_with_result_limit(self):
        I,J = ak.join_on_eq_with_dt(self.a1,self.a1,self.a1,self.a1,self.dt,"true_dt",result_limit=self.N*self.N)
        self.assertEqual(self.N*self.N, I.size)
        self.assertEqual(self.N*self.N, J.size)

    def test_join_on_eq_with_abs_dt(self):
        I,J = ak.join_on_eq_with_dt(self.a2,self.a1,self.t1,self.t2,self.dt,"abs_dt")
        self.assertEqual(self.N, I.size)
        self.assertEqual(self.N, J.size)

    def test_join_on_eq_with_pos_dt(self):
        I,J = ak.join_on_eq_with_dt(self.a2,self.a1,self.t1,self.t2,self.dt,"pos_dt")
        self.assertEqual(self.N, I.size)
        self.assertEqual(self.N, J.size)

    def test_join_on_eq_with_abs_dt_outside_window(self):
        # should get 0 answers
        # N^2 matches but 0 within dt window
        dt = 8
        I,J = ak.join_on_eq_with_dt(self.a1,self.a1,self.t1,self.t1*10,dt,"abs_dt")
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)

        # should get 0 answers
        # N matches but 0 within dt window
        dt = 8
        I,J = ak.join_on_eq_with_dt(self.a2,self.a1,self.t1,self.t2,dt,"abs_dt")
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)

    def test_join_on_eq_with_pos_dt_outside_window(self):
        # should get 0 answers
        # N matches but 0 within dt window
        dt = 8
        I,J = ak.join_on_eq_with_dt(self.a2,self.a1,self.t1,self.t2,dt,"pos_dt")
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)
