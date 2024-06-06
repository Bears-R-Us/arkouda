import arkouda as ak
import numpy as np

import pytest
import json


def get_server_max_array_dims():
    try:
        return json.load(open('serverConfig.json', 'r'))['max_array_dims']
    except (ValueError, FileNotFoundError, TypeError, KeyError):
        return 1


class TestManipulationFunctions:
    @pytest.mark.skipif(get_server_max_array_dims() < 2, reason="vstack requires server with 'max_array_dims' >= 2")
    def test_vstack(self):
        a = [ak.random.randint(0, 10, 25) for _ in range(4)]
        n = [x.to_ndarray() for x in a]

        n_vstack = np.vstack(n)
        a_vstack = ak.vstack(a)

        assert n_vstack.tolist() == a_vstack.to_list()
