from itertools import product

import numpy as np

import arkouda as ak
import pytest


def test_mulitdimensional_array_creation():
    n = np.array([[0, 0], [0, 1], [1, 1]])
    a = ak.array([[0, 0], [0, 1], [1, 1]])
    assert (n.tolist() == a.to_list())
    n = np.arange(27).reshape((3, 3, 3))
    a = ak.arange(27).reshape((3, 3, 3))
    assert (n.tolist() == a.to_list())
    n = np.arange(27).reshape(3, 3, 3)
    a = ak.arange(27).reshape(3, 3, 3)
    assert (n.tolist() == a.to_list())
    n = np.arange(27, dtype=np.uint64).reshape(3, 3, 3)
    a = ak.arange(27, dtype=ak.uint64).reshape(3, 3, 3)
    assert (n.tolist() == a.to_list())


def test_arrayview_int_indexing():
    nd = np.arange(9).reshape(3, 3)
    pd_reshape = ak.arange(9).reshape(3, 3)
    pd_array = ak.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    nd_ind = [nd[i, j] for (i, j) in product(range(3), repeat=2)]
    reshape_ind = [pd_reshape[i, j] for (i, j) in product(range(3), repeat=2)]
    array_ind = [pd_array[i, j] for (i, j) in product(range(3), repeat=2)]
    assert (nd_ind == reshape_ind)
    assert (nd_ind == array_ind)

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


def test_int_list_indexing():
    iav = ak.arange(30).reshape((5, 3, 2))
    uav = ak.arange(30, dtype=ak.uint64).reshape((5, 3, 2))

    iind = ak.array([3, 0, 1])
    uind = ak.cast(iind, ak.uint64)
    assert (iav[iind] == iav[uind])
    assert (uav[iind] == uav[uind])


def test_set_index():
    inav = np.arange(30).reshape((5, 3, 2))
    unav = np.arange(30, dtype=np.uint64).reshape((5, 3, 2))
    iav = ak.arange(30).reshape((5, 3, 2))
    uav = ak.arange(30, dtype=ak.uint64).reshape((5, 3, 2))

    nind = (3, 0, 1)
    iind = ak.array(nind)
    uind = ak.cast(iind, ak.uint64)

    inav[nind] = -9999
    unav[nind] = 2**64 - 9999
    iav[uind] = -9999
    uav[iind] = -9999
    assert (iav[uind] == inav[nind])
    assert (iav[iind] == iav[uind])
    assert (uav[uind] == unav[nind])
    assert (uav[iind] == uav[uind])


def test_get_bool_pdarray():
    n = np.arange(30).reshape(5, 3, 2)
    a = ak.arange(30).reshape(5, 3, 2)

    n_bool_list = n[True, True, True].tolist()
    a_bool_list = a[True, True, True].to_list()
    assert (n_bool_list == a_bool_list)
    n_bool_list = n[False, True, True].tolist()
    a_bool_list = a[False, True, True].to_list()
    assert (n_bool_list == a_bool_list)
    n_bool_list = n[True, False, True].tolist()
    a_bool_list = a[True, False, True].to_list()
    assert (n_bool_list == a_bool_list)
    n_bool_list = n[True, True, False].tolist()
    a_bool_list = a[True, True, False].to_list()
    assert (n_bool_list == a_bool_list)


def test_set_bool_pdarray():
    n = np.arange(30).reshape(5, 3, 2)
    a = ak.arange(30).reshape(5, 3, 2)

    n[True, True, True] = 9
    a[True, True, True] = 9
    assert (n.tolist() == a.to_list())
    n[False, True, True] = 5
    a[False, True, True] = 5
    assert (n.tolist() == a.to_list())
    n[True, False, True] = 6
    a[True, False, True] = 6
    assert (n.tolist() == a.to_list())
    n[True, True, False] = 13
    a[True, True, False] = 13
    assert (n.tolist() == a.to_list())


def test_reshape_order():
    # Keep 'C'/'F' (C/Fortran) order to be consistent with numpy
    # But also accept more descriptive 'row_major' and 'column_major'
    nd = np.arange(30).reshape((5, 3, 2), order="C")
    ak_C = ak.arange(30).reshape((5, 3, 2), order="C")
    ak_row = ak.arange(30).reshape((5, 3, 2), order="row_major")

    nd_ind = [nd[i, j, k] for (i, j, k) in product(range(5), range(3), range(2))]
    C_order = [ak_C[i, j, k] for (i, j, k) in product(range(5), range(3), range(2))]
    row_order = [ak_row[i, j, k] for (i, j, k) in product(range(5), range(3), range(2))]
    assert (nd_ind == C_order)
    assert (nd_ind == row_order)

    nd = np.arange(30).reshape((5, 3, 2), order="F")
    ak_F = ak.arange(30).reshape((5, 3, 2), order="F")
    ak_column = ak.arange(30).reshape((5, 3, 2), order="column_major")

    nd_ind = [nd[i, j, k] for (i, j, k) in product(range(5), range(3), range(2))]
    F_order = [ak_F[i, j, k] for (i, j, k) in product(range(5), range(3), range(2))]
    column_order = [ak_column[i, j, k] for (i, j, k) in product(range(5), range(3), range(2))]
    assert (nd_ind == F_order)
    assert (nd_ind == column_order)


def test_basic_indexing():
    # verify functionality is consistent with numpy basic indexing tutorial
    # https://numpy.org/doc/stable/user/basics.indexing.html
    n = np.arange(10).reshape(2, 5)
    a = ak.arange(10).reshape(2, 5)

    assert (list(n.shape) == a.shape.to_list())
    # n.tolist() = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert (n.tolist() == a.to_list())
    assert (n.__str__() == a.__str__())

    assert (n[1, 3] == a[1, 3])
    assert (n[1, -1] == a[1, -1])

    assert (n[0].tolist() == a[0].to_list())
    assert (n[0][2] == a[0][2])

    n = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    a = ak.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    # n[0, 1:7:2].tolist() = [1, 3, 5]
    assert (n[0, 1:7:2].tolist() == a[0, 1:7:2].to_list())
    # n[0, -2:10].tolist() = [8, 9]
    assert (n[0, -2:10].tolist() == a[0, -2:10].to_list())
    # n[0, -3:3:-1].tolist() = [7, 6, 5, 4]
    assert (n[0, -3:3:-1].tolist() == a[0, -3:3:-1].to_list())
    # n[0, 5:].tolist() = [5, 6, 7, 8, 9]
    assert (n[0, 5:].tolist() == a[0, 5:].to_list())

    n = np.array([[[1], [2], [3]], [[4], [5], [6]]])
    a = ak.array([[[1], [2], [3]], [[4], [5], [6]]])
    # list(n.shape) = [2, 3, 1]
    assert (list(n.shape) == a.shape.to_list())
    # n.tolist() = [[[1], [2], [3]], [[4], [5], [6]]]
    assert (n.tolist() == a.to_list())
    assert (n.__str__() == a.__str__())

    # n[1:2].tolist() = [[[4], [5], [6]]]
    assert (n[1:2].tolist() == a[1:2].to_list())


def test_slicing():
    a = ak.arange(30).reshape(2, 3, 5)
    n = np.arange(30).reshape(2, 3, 5)
    assert (n.tolist() == a.to_list())

    # n[:, ::-1, 1:5:2].tolist() = [[[11, 13], [6, 8], [1, 3]], [[26, 28], [21, 23], [16, 18]]]
    assert (n[:, ::-1, 1:5:2].tolist() == a[:, ::-1, 1:5:2].to_list())

    # n[:, 5:8, 1:5:2].tolist() = [[], []]
    assert (n[:, 5:8, 1:5:2].tolist() == a[:, 5:8, 1:5:2].to_list())
    # n[:, 5:8, 1:5:2][1].tolist() = []
    assert (n[:, 5:8, 1:5:2][1].tolist() == a[:, 5:8, 1:5:2][1].to_list())

    a = ak.arange(30).reshape(2, 3, 5, order="F")
    n = np.arange(30).reshape(2, 3, 5, order="F")
    assert (n.tolist() == a.to_list())

    # n[:, ::-1, 1:5:2].tolist() = [[10, 22], [8, 20], [6, 18]]
    assert (n[:, ::-1, 1:5:2].tolist() == a[:, ::-1, 1:5:2].to_list())
