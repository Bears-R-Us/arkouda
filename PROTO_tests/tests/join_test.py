import numpy as np
import pytest

import arkouda as ak

"""
Encapsulates a variety of arkouda join_on_eq_with_dt test cases.
"""


class TestJoin:
    @classmethod
    def setup_class(cls):
        cls.size = 1000
        cls.a1 = ak.ones(cls.size, dtype=np.int64)
        cls.a2 = ak.arange(0, cls.size, 1)
        cls.t1 = cls.a1
        cls.t2 = cls.a1 * 10
        cls.dt = 10
        ak.verbose = False

    @pytest.mark.parametrize("dt_type", ["true_dt", "abs_dt", "pos_dt"])
    def test_join_on_eq_by_dt(self, dt_type):
        x, y = ak.join_on_eq_with_dt(self.a2, self.a1, self.t1, self.t2, self.dt, dt_type)
        assert self.size // pytest.nl, x.size
        assert self.size // pytest.nl, y.size

    def test_join_on_eq_with_true_dt_with_result_limit(self):
        lim = (self.size + pytest.nl) * self.size
        res_size = self.size * self.size
        x, y = ak.join_on_eq_with_dt(
            self.a1, self.a1, self.a1, self.a1, self.dt, "true_dt", result_limit=lim
        )
        assert res_size, x.size
        assert res_size, y.size

    def test_join_on_eq_with_abs_dt_outside_window(self):
        """
        Should get 0 answers because N^2 matches but 0 within dt window
        """
        dt = 8
        x, y = ak.join_on_eq_with_dt(self.a1, self.a1, self.t1, self.t1 * 10, dt, "abs_dt")
        assert 0 == x.size
        assert 0 == y.size

        x, y = ak.join_on_eq_with_dt(self.a2, self.a1, self.t1, self.t2, dt, "abs_dt")
        assert 0 == x.size
        assert 0 == y.size

    def test_join_on_eq_with_pos_dt_outside_window(self):
        """
        Should get 0 answers because N matches but 0 within dt window
        """
        for dt in 8, np.int64(8):
            x, y = ak.join_on_eq_with_dt(self.a2, self.a1, self.t1, self.t2, dt, "pos_dt")
            assert 0 == x.size
            assert 0 == y.size

        dt = np.int64(8)
        x, y = ak.join_on_eq_with_dt(self.a2, self.a1, self.t1, self.t2, dt, "pos_dt", int(0))
        assert 0 == x.size
        assert 0 == y.size

    def test_gen_ranges(self):
        start = ak.array([0, 10, 20])
        end = ak.array([10, 20, 30])

        segs, ranges = ak.join.gen_ranges(start, end)
        assert segs.to_list() == [0, 10, 20]
        assert ranges.to_list() == list(range(30))

        with pytest.raises(ValueError):
            segs, ranges = ak.join.gen_ranges(ak.array([11, 12, 41]), end)

    def test_inner_join(self):
        left = ak.arange(10)
        right = ak.array([0, 5, 3, 3, 4, 6, 7, 9, 8, 1])

        l, r = ak.join.inner_join(left, right)
        assert left[l].to_list() == right[r].to_list()

        l, r = ak.join.inner_join(left, right, wherefunc=join_where, whereargs=(left, right))
        assert left[l].to_list() == right[r].to_list()

        for where_func in ak.unique, ak.intersect1d:
            with pytest.raises(ValueError):
                l, r = ak.join.inner_join(left, right, wherefunc=where_func)

        for where_args in ((ak.arange(5), ak.arange(10)), (ak.arange(10), ak.arange(5))):
            with pytest.raises(ValueError):
                l, r = ak.join.inner_join(
                    left, right, wherefunc=ak.intersect1d, whereargs=where_args
                )

    def test_str_inner_join(self):
        int_left = ak.arange(50)
        int_right = ak.randint(0, 50, 50)
        str_left = ak.array([f"str {i}" for i in int_left.to_list()])
        str_right = ak.array([f"str {i}" for i in int_right.to_list()])

        str_l, str_r = ak.join.inner_join(str_left, str_right)
        assert str_left[str_l].to_list() == str_right[str_r].to_list()

        str_l_where, str_r_where = ak.join.inner_join(
            str_left, str_right, wherefunc=join_where, whereargs=(str_left, str_right)
        )
        assert str_left[str_l_where].to_list() == str_right[str_r_where].to_list()

        # reproducer from PR
        int_left = ak.arange(10)
        int_right = ak.array([0, 5, 3, 3, 4, 6, 7, 9, 8, 1])
        str_left = ak.array([f"str {i}" for i in int_left.to_list()])
        str_right = ak.array([f"str {i}" for i in int_right.to_list()])

        sl, sr = ak.join.inner_join(str_left, str_right)
        assert str_left[sl].to_list() == str_right[sr].to_list()

        def where_func(x, y):
            return x % 2 == 0

        il, ir = ak.join.inner_join(
            int_left, int_right, wherefunc=where_func, whereargs=(int_left, int_right)
        )
        sl, sr = ak.join.inner_join(
            str_left, str_right, wherefunc=where_func, whereargs=(int_left, int_right)
        )
        assert sl.to_list() == il.to_list()
        assert sr.to_list() == ir.to_list()

    def test_cat_inner_join(self):
        int_left = ak.arange(50)
        int_right = ak.randint(0, 50, 50)
        str_left = ak.array([f"str {i}" for i in int_left.to_list()])
        str_right = ak.array([f"str {i}" for i in int_right.to_list()])
        cat_left = ak.Categorical(str_left)
        cat_right = ak.Categorical(str_right)

        # Base Case
        cat_l, cat_r = ak.join.inner_join(cat_left, cat_right)
        assert cat_left[cat_l].to_list() == cat_right[cat_r].to_list()

        cat_l_where, cat_r_where = ak.join.inner_join(
            cat_left, cat_right, wherefunc=join_where, whereargs=(cat_left, cat_right)
        )
        assert cat_left[cat_l_where].to_list() == cat_right[cat_r_where].to_list()

    def test_mixed_inner_join_where(self):
        int_left = ak.arange(50)
        int_right = ak.randint(0, 50, 50)
        str_left = ak.array([f"str {i}" for i in int_left.to_list()])
        str_right = ak.array([f"str {i}" for i in int_right.to_list()])
        cat_left = ak.Categorical(str_left)
        cat_right = ak.Categorical(str_right)

        left, right = ak.join.inner_join(
            int_left, int_right, wherefunc=join_where, whereargs=(cat_left, str_right)
        )
        assert cat_left[left].to_list() == cat_right[right].to_list()

        left, right = ak.join.inner_join(
            str_left, str_right, wherefunc=join_where, whereargs=(cat_left, int_right)
        )
        assert cat_left[left].to_list() == cat_right[right].to_list()

        left, right = ak.join.inner_join(
            cat_left, cat_right, wherefunc=join_where, whereargs=(str_left, int_right)
        )
        assert cat_left[left].to_list() == cat_right[right].to_list()

    def test_lookup(self):
        keys = ak.arange(5)
        values = 10 * keys
        args = ak.array([5, 3, 1, 4, 2, 3, 1, 0])
        ans = [-1, 30, 10, 40, 20, 30, 10, 0]
        # Simple lookup with int keys
        # Also test shortcut for unique-ordered keys
        res = ak.lookup(keys, values, args, fillvalue=-1)
        assert res.to_list() == ans
        # Compound lookup with (str, int) keys
        res2 = ak.lookup(
            (ak.cast(keys, ak.str_), keys), values, (ak.cast(args, ak.str_), args), fillvalue=-1
        )
        assert res2.to_list() == ans
        # Keys not in uniqued order
        res3 = ak.lookup(keys[::-1], values[::-1], args, fillvalue=-1)
        assert res3.to_list() == ans
        # Non-unique keys should raise error
        with pytest.warns(UserWarning):
            keys = ak.arange(10) % 5
            values = 10 * keys
            ak.lookup(keys, values, args)

    def test_error_handling(self):
        """
        Tests error TypeError and ValueError handling
        """
        with pytest.raises(TypeError):
            ak.join_on_eq_with_dt([list(range(0, 11))], self.a1, self.t1, self.t2, 8, "pos_dt")
        with pytest.raises(TypeError):
            ak.join_on_eq_with_dt([self.a1, list(range(0, 11))], self.t1, self.t2, 8, "pos_dt")
        with pytest.raises(TypeError):
            ak.join_on_eq_with_dt([self.a1, self.a1, list(range(0, 11))], self.t2, 8, "pos_dt")
        with pytest.raises(TypeError):
            ak.join_on_eq_with_dt([self.a1, self.a1, self.t1, list(range(0, 11))], 8, "pos_dt")
        with pytest.raises(TypeError):
            ak.join_on_eq_with_dt(self.a1, self.a1, self.t1, self.t2, "8", "pos_dt")
        with pytest.raises(ValueError):
            ak.join_on_eq_with_dt(self.a1, self.a1, self.t1, self.t1 * 10, 8, "ab_dt")
        with pytest.raises(ValueError):
            ak.join_on_eq_with_dt(self.a1, self.a1, self.t1, self.t1 * 10, 8, "abs_dt", -1)


def join_where(left, right):
    return ak.arange(left.size) % 2 == 0
