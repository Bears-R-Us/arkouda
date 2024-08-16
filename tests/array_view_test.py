import random
from itertools import product

import numpy as np
import pytest

import arkouda as ak

SHAPE = {
    56: (7, 8),
    30: (5, 3, 2),
    27: (3, 3, 3),
    48: (2, 3, 4, 2),
}

SIZE = [56, 30, 27, 48]
NO_BOOL = [ak.int64, ak.float64, ak.uint64]


class TestArrayView:
    @pytest.mark.parametrize("size", SIZE)
    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_mulitdimensional_array_creation(self, size, num_type):
        n = np.array([[0, 0], [0, 1], [1, 1]])
        a = ak.array([[0, 0], [0, 1], [1, 1]])
        assert np.array_equal(n.tolist(), a.to_list())
        n = np.arange(size).reshape(SHAPE[size])
        a = ak.arange(size).reshape(SHAPE[size])
        assert np.array_equal(n.tolist(), a.to_list())
        n = np.arange(size, dtype=num_type).reshape(SHAPE[size])
        a = ak.arange(size, dtype=num_type).reshape(SHAPE[size])
        assert np.array_equal(n.tolist(), a.to_list())

    def test_arrayview_int_indexing(self):
        nd = np.arange(9).reshape(3, 3)
        pd_reshape = ak.arange(9).reshape(3, 3)
        pd_array = ak.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

        nd_ind = [nd[i, j] for (i, j) in product(range(3), repeat=2)]
        reshape_ind = [pd_reshape[i, j] for (i, j) in product(range(3), repeat=2)]
        array_ind = [pd_array[i, j] for (i, j) in product(range(3), repeat=2)]
        assert nd_ind == reshape_ind
        assert nd_ind == array_ind

        with pytest.raises(IndexError):
            # index out bounds (>= dimension)
            # index 3 is out of bounds for axis 0 with size 3
            pd_reshape[3, 1]
        with pytest.raises(IndexError):
            # index -4 is out of bounds for axis 1 with size 3
            pd_reshape[2, -4]
        with pytest.raises(IndexError):
            # too many indicies for array: array is 2-dimensional, but 3 were indexed
            pd_reshape[0, 1, 1]
        with pytest.raises(ValueError):

            # cannot reshape array of size 9 into shape (4,3)
            ak.arange(9).reshape(4, 3)

    @pytest.mark.parametrize("size", SIZE)
    def test_int_list_indexing(self, size):
        iav = ak.arange(size).reshape(SHAPE[size])
        uav = ak.arange(size, dtype=np.uint64).reshape(SHAPE[size])

        iind = ak.ones(len(SHAPE[size]), dtype=ak.uint64)
        uind = ak.cast(iind, np.uint64)
        assert np.array_equal(iav[iind], iav[uind])
        assert np.array_equal(uav[iind], uav[uind])

    @pytest.mark.parametrize("size", SIZE)
    def test_set_index(self, size):
        inav = np.arange(size).reshape(SHAPE[size])
        unav = np.arange(size, dtype=np.uint64).reshape(SHAPE[size])
        iav = ak.arange(size).reshape(SHAPE[size])
        uav = ak.arange(size, dtype=ak.uint64).reshape(SHAPE[size])

        nind = tuple(random.randint(0, y - 1) for y in SHAPE[size])
        iind = ak.array(nind)
        uind = ak.cast(iind, ak.uint64)

        inav[nind] = -9999
        unav[nind] = 2**64 - 9999
        iav[uind] = -9999
        uav[iind] = -9999
        assert np.array_equal(iav[uind], inav[nind])
        assert np.array_equal(iav[iind], iav[uind])
        assert np.array_equal(uav[uind], unav[nind])
        assert np.array_equal(uav[iind], uav[uind])

    @pytest.mark.parametrize("size", SIZE)
    def test_get_bool_pdarray(self, size):
        N = len(SHAPE[size])

        n = np.arange(size).reshape(SHAPE[size])
        a = ak.arange(size).reshape(SHAPE[size])
        for i in range(0, N + 1):
            truth = tuple(j != i - 1 if i > 0 else True for j in range(N))
            n_bool_list = n[truth].tolist()
            a_bool_list = a[truth].to_list()

            assert n_bool_list == a_bool_list

    @pytest.mark.parametrize("size", SIZE)
    def test_set_bool_pdarray(self, size):
        N = len(SHAPE[size])

        for i in range(0, N + 1):
            n = np.arange(size).reshape(SHAPE[size])
            a = ak.arange(size).reshape(SHAPE[size])
            truth = tuple(j != i - 1 if i > 0 else True for j in range(N))

            val = random.randint(1, 10)
            n[truth] = val
            a[truth] = val

            assert n.tolist() == a.to_list()

    @pytest.mark.parametrize("size", SIZE)
    def test_reshape_order(self, size):
        # Keep 'C'/'F' (C/Fortran) order to be consistent with numpy
        # But also accept more descriptive 'row_major' and 'column_major'
        nd = np.arange(size).reshape(SHAPE[size], order="C")
        ak_C = ak.arange(size).reshape(SHAPE[size], order="C")
        ak_row = ak.arange(size).reshape(SHAPE[size], order="row_major")

        nd_ind = [nd[t] for t in product(*(range(s) for s in SHAPE[size]))]
        C_order = [ak_C[t] for t in product(*(range(s) for s in SHAPE[size]))]
        row_order = [ak_row[t] for t in product(*(range(s) for s in SHAPE[size]))]

        assert nd_ind == C_order
        assert nd_ind == row_order

        nd = np.arange(size).reshape(SHAPE[size], order="F")
        ak_F = ak.arange(size).reshape(SHAPE[size], order="F")
        ak_column = ak.arange(size).reshape(SHAPE[size], order="column_major")

        nd_ind = [nd[t] for t in product(*(range(s) for s in SHAPE[size]))]
        F_order = [ak_F[t] for t in product(*(range(s) for s in SHAPE[size]))]
        column_order = [ak_column[t] for t in product(*(range(s) for s in SHAPE[size]))]

        assert np.array_equal(nd_ind, F_order)
        assert np.array_equal(nd_ind, column_order)

    def test_basic_indexing(self):
        # verify functionality is consistent with numpy basic indexing tutorial
        # https://numpy.org/doc/stable/user/basics.indexing.html
        n = np.arange(10).reshape(2, 5)
        a = ak.arange(10).reshape(2, 5)

        assert list(n.shape) == a.shape.to_list()
        # n.tolist() = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        assert n.tolist() == a.to_list()
        assert n.__str__() == a.__str__()

        assert n[1, 3] == a[1, 3]
        assert n[1, -1] == a[1, -1]

        assert n[0].tolist() == a[0].to_list()
        assert n[0][2] == a[0][2]

        n = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        a = ak.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        # n[0, 1:7:2].tolist() = [1, 3, 5]
        assert n[0, 1:7:2].tolist() == a[0, 1:7:2].to_list()
        # n[0, -2:10].tolist() = [8, 9]
        assert n[0, -2:10].tolist() == a[0, -2:10].to_list()
        # n[0, -3:3:-1].tolist() = [7, 6, 5, 4]
        assert n[0, -3:3:-1].tolist() == a[0, -3:3:-1].to_list()
        # n[0, 5:].tolist() = [5, 6, 7, 8, 9]
        assert n[0, 5:].tolist() == a[0, 5:].to_list()

        n = np.array([[[1], [2], [3]], [[4], [5], [6]]])
        a = ak.array([[[1], [2], [3]], [[4], [5], [6]]])
        # list(n.shape) = [2, 3, 1]
        assert list(n.shape) == a.shape.to_list()
        # n.tolist() = [[[1], [2], [3]], [[4], [5], [6]]]
        assert n.tolist() == a.to_list()
        assert n.__str__() == a.__str__()

        # n[1:2].tolist() = [[[4], [5], [6]]]
        assert n[1:2].tolist() == a[1:2].to_list()

    def test_slicing(self):
        a = ak.arange(30).reshape(2, 3, 5)
        n = np.arange(30).reshape(2, 3, 5)
        assert n.tolist() == a.to_list()

        # n[:, ::-1, 1:5:2].tolist() = [[[11, 13], [6, 8], [1, 3]], [[26, 28], [21, 23], [16, 18]]]
        assert n[:, ::-1, 1:5:2].tolist() == a[:, ::-1, 1:5:2].to_list()

        # n[:, 5:8, 1:5:2].tolist() = [[], []]
        assert n[:, 5:8, 1:5:2].tolist() == a[:, 5:8, 1:5:2].to_list()
        # n[:, 5:8, 1:5:2][1].tolist() = []
        assert n[:, 5:8, 1:5:2][1].tolist() == a[:, 5:8, 1:5:2][1].to_list()

        a = ak.arange(30).reshape(2, 3, 5, order="F")
        n = np.arange(30).reshape(2, 3, 5, order="F")
        assert n.tolist() == a.to_list()

        # n[:, ::-1, 1:5:2].tolist() = [[10, 22], [8, 20], [6, 18]]
        assert n[:, ::-1, 1:5:2].tolist() == a[:, ::-1, 1:5:2].to_list()
