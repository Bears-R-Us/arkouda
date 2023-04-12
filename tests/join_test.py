import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

"""
Encapsulates a variety of arkouda join_on_eq_with_dt test cases.
"""


class JoinTest(ArkoudaTest):
    def setUp(self):
        ArkoudaTest.setUp(self)
        self.N = 1000
        self.a1 = ak.ones(self.N, dtype=np.int64)
        self.a2 = ak.arange(0, self.N, 1)
        self.t1 = self.a1
        self.t2 = self.a1 * 10
        self.dt = 10
        ak.verbose = False

    def test_join_on_eq_with_true_dt(self):
        I, J = ak.join_on_eq_with_dt(self.a2, self.a1, self.t1, self.t2, self.dt, "true_dt")
        nl = ak.get_config()["numLocales"]
        self.assertEqual(self.N // nl, I.size)
        self.assertEqual(self.N // nl, J.size)

    def test_join_on_eq_with_true_dt_with_result_limit(self):
        nl = ak.get_config()["numLocales"]
        lim = (self.N + nl) * self.N
        res_size = self.N * self.N
        I, J = ak.join_on_eq_with_dt(
            self.a1, self.a1, self.a1, self.a1, self.dt, "true_dt", result_limit=lim
        )
        self.assertEqual(res_size, I.size)
        self.assertEqual(res_size, J.size)

    def test_join_on_eq_with_abs_dt(self):
        I, J = ak.join_on_eq_with_dt(self.a2, self.a1, self.t1, self.t2, self.dt, "abs_dt")
        nl = ak.get_config()["numLocales"]
        self.assertEqual(self.N // nl, I.size)
        self.assertEqual(self.N // nl, J.size)

    def test_join_on_eq_with_pos_dt(self):
        I, J = ak.join_on_eq_with_dt(self.a2, self.a1, self.t1, self.t2, self.dt, "pos_dt")
        nl = ak.get_config()["numLocales"]
        self.assertEqual(self.N // nl, I.size)
        self.assertEqual(self.N // nl, J.size)

    def test_join_on_eq_with_abs_dt_outside_window(self):
        """
        Should get 0 answers because N^2 matches but 0 within dt window
        """
        dt = 8
        I, J = ak.join_on_eq_with_dt(self.a1, self.a1, self.t1, self.t1 * 10, dt, "abs_dt")
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)

        I, J = ak.join_on_eq_with_dt(self.a2, self.a1, self.t1, self.t2, dt, "abs_dt")
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)

    def test_join_on_eq_with_pos_dt_outside_window(self):
        """
        Should get 0 answers because N matches but 0 within dt window
        """
        dt = 8
        I, J = ak.join_on_eq_with_dt(self.a2, self.a1, self.t1, self.t2, dt, "pos_dt")
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)

        dt = np.int64(8)
        I, J = ak.join_on_eq_with_dt(self.a2, self.a1, self.t1, self.t2, dt, "pos_dt")
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)

        I, J = ak.join_on_eq_with_dt(self.a2, self.a1, self.t1, self.t2, dt, "pos_dt", int(0))
        self.assertEqual(0, I.size)
        self.assertEqual(0, J.size)

    def test_gen_ranges(self):
        start = ak.array([0, 10, 20])
        end = ak.array([10, 20, 30])

        segs, ranges = ak.join.gen_ranges(start, end)
        self.assertListEqual(segs.to_list(), [0, 10, 20])
        self.assertListEqual(ranges.to_list(), list(range(30)))

        with self.assertRaises(ValueError):
            segs, ranges = ak.join.gen_ranges(ak.array([11, 12, 41]), end)

    def test_inner_join(self):
        left = ak.arange(10)
        right = ak.array([0, 5, 3, 3, 4, 6, 7, 9, 8, 1])

        l, r = ak.join.inner_join(left, right)
        self.assertListEqual(left[l].to_list(), right[r].to_list())

        l, r = ak.join.inner_join(
            left, right, wherefunc=num_join_where, whereargs=(left, right)
        )
        self.assertListEqual(left[l].to_list(), right[r].to_list())

        with self.assertRaises(ValueError):
            l, r = ak.join.inner_join(left, right, wherefunc=ak.unique)

        with self.assertRaises(ValueError):
            l, r = ak.join.inner_join(left, right, wherefunc=ak.intersect1d)

        with self.assertRaises(ValueError):
            l, r = ak.join.inner_join(
                left, right, wherefunc=ak.intersect1d, whereargs=(ak.arange(5), ak.arange(10))
            )

        with self.assertRaises(ValueError):
            l, r = ak.join.inner_join(
                left, right, wherefunc=ak.intersect1d, whereargs=(ak.arange(10), ak.arange(5))
            )

    def test_str_cat_inner_join(self):
        strLeft = ak.array(["a", "c", "c", "d", "a", "b", "a", "e"])
        strRight = ak.array(["c", "b", "a", "d", "a", "c", "b", "d"])

        strL, strR = ak.join.inner_join(strLeft, strRight)
        self.assertListEqual(strLeft[strL].to_list(), strRight[strR].to_list())

        strLWhere, strRWhere = ak.join.inner_join(
            strLeft, strRight, wherefunc=string_join_where, whereargs=(strLeft, strRight)
        )
        self.assertListEqual(strLeft[strLWhere].to_list(), strRight[strRWhere].to_list())

        catLeft = ak.Categorical(strLeft)
        catRight = ak.Categorical(strRight)

        catL, catR = ak.join.inner_join(catLeft, catRight)
        self.assertListEqual(
            catLeft[catL].to_list(),
            catRight[catR].to_list(),
        )

        catLWhere, catRWhere = ak.join.inner_join(
            catLeft, catRight, wherefunc=string_join_where, whereargs=(strLeft, strRight)
        )
        self.assertListEqual(
            catLeft[catLWhere].to_list(),
            catRight[catRWhere].to_list(),
        )

    def test_lookup(self):
        keys = ak.arange(5)
        values = 10 * keys
        args = ak.array([5, 3, 1, 4, 2, 3, 1, 0])
        ans = [-1, 30, 10, 40, 20, 30, 10, 0]
        # Simple lookup with int keys
        # Also test shortcut for unique-ordered keys
        res = ak.lookup(keys, values, args, fillvalue=-1)
        self.assertListEqual(res.to_list(), ans)
        # Compound lookup with (str, int) keys
        res2 = ak.lookup(
            (ak.cast(keys, ak.str_), keys), values, (ak.cast(args, ak.str_), args), fillvalue=-1
        )
        self.assertListEqual(res2.to_list(), ans)
        # Keys not in uniqued order
        res3 = ak.lookup(keys[::-1], values[::-1], args, fillvalue=-1)
        self.assertListEqual(res3.to_list(), ans)
        # Non-unique keys should raise error
        with self.assertWarns(UserWarning):
            keys = ak.arange(10) % 5
            values = 10 * keys
            ak.lookup(keys, values, args)

    def test_error_handling(self):
        """
        Tests error TypeError and ValueError handling
        """
        with self.assertRaises(TypeError):
            ak.join_on_eq_with_dt([list(range(0, 11))], self.a1, self.t1, self.t2, 8, "pos_dt")
        with self.assertRaises(TypeError):
            ak.join_on_eq_with_dt([self.a1, list(range(0, 11))], self.t1, self.t2, 8, "pos_dt")
        with self.assertRaises(TypeError):
            ak.join_on_eq_with_dt([self.a1, self.a1, list(range(0, 11))], self.t2, 8, "pos_dt")
        with self.assertRaises(TypeError):
            ak.join_on_eq_with_dt([self.a1, self.a1, self.t1, list(range(0, 11))], 8, "pos_dt")
        with self.assertRaises(TypeError):
            ak.join_on_eq_with_dt(self.a1, self.a1, self.t1, self.t2, "8", "pos_dt")
        with self.assertRaises(ValueError):
            ak.join_on_eq_with_dt(self.a1, self.a1, self.t1, self.t1 * 10, 8, "ab_dt")
        with self.assertRaises(ValueError):
            ak.join_on_eq_with_dt(self.a1, self.a1, self.t1, self.t1 * 10, 8, "abs_dt", -1)


def string_join_where(L, R):
    idx = []
    for i in range(L.size):
        if L[i] in ["a", "c", "e"] and R[i] in ["a", "c"]:
            idx.append(True)
        else:
            idx.append(False)

    return ak.array(idx)


def num_join_where(L, R):
    idx = []
    for i in range(L.size):
        if L[i] in [0, 2, 3] and R[i] in [0, 2]:
            idx.append(True)
        else:
            idx.append(False)

    return ak.array(idx)
