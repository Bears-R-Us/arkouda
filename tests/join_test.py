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

        l, r = ak.join.inner_join(left, right, wherefunc=join_where, whereargs=(left, right))
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

    def test_multi_array_inner_join(self):
        size = 1000
        seed = 1
        a = ak.randint(-size // 10, size // 10, size, seed=seed)
        b = ak.randint(-size // 10, size // 10, size, seed=seed + 1)
        ones = ak.ones(size, int)
        altr = ak.cast(ak.arange(size) % 2 == 0, int)
        left_lists = [
            [a, ones],
            [pda_to_str_helper(a), ones],
            [a, pda_to_str_helper(ones)],
            [pda_to_str_helper(a), pda_to_str_helper(ones)],
        ]
        right_list = [
            [b, altr],
            [pda_to_str_helper(b), altr],
            [b, pda_to_str_helper(altr)],
            [pda_to_str_helper(b), pda_to_str_helper(altr)],
        ]
        for left, right in zip(left_lists, right_list):
            # test with no where args
            l_ind, r_ind = ak.join.inner_join(left, right)
            for lf, rt in zip(left, right):
                self.assertTrue((lf[l_ind] == rt[r_ind]).all())

            # test with where args
            def where_func(x, y):
                x_bool = (
                    (x[0] % 2 == 0) if isinstance(x[0], ak.pdarray) else (x[0].get_lengths() % 2 == 0)
                )
                y_bool = (
                    (x[0] % 2 == 0) if isinstance(y[0], ak.pdarray) else (y[0].get_lengths() % 2 == 0)
                )
                return x_bool | y_bool

            l_ind, r_ind = ak.join.inner_join(left, right, where_func, (left, right))
            self.assertTrue(where_func([lf[l_ind] for lf in left], [rt[r_ind] for rt in right]).all())

    def test_str_inner_join(self):
        intLeft = ak.arange(50)
        intRight = ak.randint(0, 50, 50)
        strLeft = ak.array([f"str {i}" for i in intLeft.to_list()])
        strRight = ak.array([f"str {i}" for i in intRight.to_list()])

        strL, strR = ak.join.inner_join(strLeft, strRight)
        self.assertListEqual(strLeft[strL].to_list(), strRight[strR].to_list())

        strLWhere, strRWhere = ak.join.inner_join(
            strLeft, strRight, wherefunc=join_where, whereargs=(strLeft, strRight)
        )
        self.assertListEqual(strLeft[strLWhere].to_list(), strRight[strRWhere].to_list())

        # reproducer from PR
        int_left = ak.arange(10)
        int_right = ak.array([0, 5, 3, 3, 4, 6, 7, 9, 8, 1])
        str_left = ak.array([f"str {i}" for i in int_left.to_list()])
        str_right = ak.array([f"str {i}" for i in int_right.to_list()])

        sl, sr = ak.join.inner_join(str_left, str_right)
        self.assertListEqual(str_left[sl].to_list(), str_right[sr].to_list())

        def where_func(x, y):
            return x % 2 == 0

        il, ir = ak.join.inner_join(
            int_left, int_right, wherefunc=where_func, whereargs=(int_left, int_right)
        )
        sl, sr = ak.join.inner_join(
            str_left, str_right, wherefunc=where_func, whereargs=(int_left, int_right)
        )
        self.assertListEqual(sl.to_list(), il.to_list())
        self.assertListEqual(sr.to_list(), ir.to_list())

    def test_cat_inner_join(self):
        intLeft = ak.arange(50)
        intRight = ak.randint(0, 50, 50)
        strLeft = ak.array([f"str {i}" for i in intLeft.to_list()])
        strRight = ak.array([f"str {i}" for i in intRight.to_list()])
        catLeft = ak.Categorical(strLeft)
        catRight = ak.Categorical(strRight)

        # Base Case
        catL, catR = ak.join.inner_join(catLeft, catRight)
        self.assertListEqual(catLeft[catL].to_list(), catRight[catR].to_list())

        catLWhere, catRWhere = ak.join.inner_join(
            catLeft, catRight, wherefunc=join_where, whereargs=(catLeft, catRight)
        )
        self.assertListEqual(catLeft[catLWhere].to_list(), catRight[catRWhere].to_list())

    def test_mixed_inner_join_where(self):
        intLeft = ak.arange(50)
        intRight = ak.randint(0, 50, 50)
        strLeft = ak.array([f"str {i}" for i in intLeft.to_list()])
        strRight = ak.array([f"str {i}" for i in intRight.to_list()])
        catLeft = ak.Categorical(strLeft)
        catRight = ak.Categorical(strRight)

        L, R = ak.join.inner_join(intLeft, intRight, wherefunc=join_where, whereargs=(catLeft, strRight))
        self.assertListEqual(catLeft[L].to_list(), catRight[R].to_list())

        L, R = ak.join.inner_join(strLeft, strRight, wherefunc=join_where, whereargs=(catLeft, intRight))
        self.assertListEqual(catLeft[L].to_list(), catRight[R].to_list())

        L, R = ak.join.inner_join(catLeft, catRight, wherefunc=join_where, whereargs=(strLeft, intRight))
        self.assertListEqual(catLeft[L].to_list(), catRight[R].to_list())

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


def join_where(L, R):
    return ak.arange(L.size) % 2 == 0


def pda_to_str_helper(pda):
    return ak.array([f"str {i}" for i in pda.to_list()])
