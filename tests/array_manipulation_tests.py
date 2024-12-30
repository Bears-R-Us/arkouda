import json

import numpy as np
import pytest

import arkouda as ak


class TestManipulationFunctions:

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_vstack(self):
        a = [ak.random.randint(0, 10, 25) for _ in range(4)]
        n = [x.to_ndarray() for x in a]

        n_vstack = np.vstack(n)
        a_vstack = ak.vstack(a)

        assert n_vstack.tolist() == a_vstack.to_list()

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_delete(self):
        a = ak.randint(0, 100, (10, 10))
        n = a.to_ndarray()

        n_delete = np.delete(n, 5, axis=1)
        a_delete = ak.delete(a, 5, axis=1)

        assert n_delete.tolist() == a_delete.to_list()

        n_delete = np.delete(n, np.array([1, 3, 5]), axis=0)
        a_delete = ak.delete(a, ak.array([1, 3, 5]), axis=0)

        assert n_delete.tolist() == a_delete.to_list()

        n_delete = np.delete(n, np.array([1, 3, 5]))
        a_delete = ak.delete(a, ak.array([1, 3, 5]))

        assert n_delete.tolist() == a_delete.to_list()

        n_delete = np.delete(n, slice(3, 5), axis=1)
        a_delete = ak.delete(a, slice(3, 5), axis=1)

        assert n_delete.tolist() == a_delete.to_list()
