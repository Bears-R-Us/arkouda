import numpy as np

import arkouda as ak
import pytest


@pytest.fixture(scope="function", autouse=True)
def bitops_data():
    pytest.bitops_data = {
        'A': ak.arange(10),
        'B': ak.cast(ak.arange(10), ak.uint64),
        'EDGE_CASES': ak.array([-(2**63), -1, 2**63 - 1]),
        'EDGE_CASES_UINT': ak.cast(ak.array([-(2**63), -1, 2**63 - 1]), ak.uint64)
    }


def test_popcount():
    # Method invocation
    # Toy input
    ans = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2]
    assert pytest.bitops_data["A"].popcount().to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert pytest.bitops_data["B"].popcount().to_list() == ans.tolist()

    # Function invocation
    # Edge case input
    ans = [1, 64, 63]
    assert ak.popcount(pytest.bitops_data["EDGE_CASES"]).to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert ak.popcount(pytest.bitops_data["EDGE_CASES_UINT"]).to_list() == ans.tolist()


def test_parity():
    ans = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
    assert pytest.bitops_data["A"].parity().to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert pytest.bitops_data["B"].parity().to_list() == ans.tolist()

    ans = [1, 0, 1]
    assert ak.parity(pytest.bitops_data["EDGE_CASES"]).to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert ak.parity(pytest.bitops_data["EDGE_CASES_UINT"]).to_list() == ans.tolist()


def test_clz():
    ans = [64, 63, 62, 62, 61, 61, 61, 61, 60, 60]
    assert pytest.bitops_data["A"].clz().to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert pytest.bitops_data["B"].clz().to_list() == ans.tolist()

    ans = [0, 0, 1]
    assert ak.clz(pytest.bitops_data["EDGE_CASES"]).to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert ak.clz(pytest.bitops_data["EDGE_CASES_UINT"]).to_list() == ans.tolist()


def test_ctz():
    ans = [0, 0, 1, 0, 2, 0, 1, 0, 3, 0]
    assert pytest.bitops_data["A"].ctz().to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert pytest.bitops_data["B"].ctz().to_list() == ans.tolist()

    ans = [63, 0, 0]
    assert ak.ctz(pytest.bitops_data["EDGE_CASES"]).to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert ak.ctz(pytest.bitops_data["EDGE_CASES_UINT"]).to_list() == ans.tolist()


def test_dtypes():
    f = ak.zeros(10, dtype=ak.float64)
    with pytest.raises(TypeError):
        f.popcount()

    with pytest.raises(TypeError):
        ak.popcount(f)


def test_rotl():
    # vector <<< scalar
    rotated = pytest.bitops_data["A"].rotl(5)
    shifted = pytest.bitops_data["A"] << 5
    # No wraparound, so these should be equal
    assert rotated.to_list() == shifted.to_list()

    r = ak.rotl(pytest.bitops_data["EDGE_CASES"], 1)
    assert r.to_list() == [1, -1, -2]

    # vector <<< vector
    rotated = pytest.bitops_data["A"].rotl(pytest.bitops_data["A"])
    shifted = pytest.bitops_data["A"] << pytest.bitops_data["A"]
    # No wraparound, so these should be equal
    assert rotated.to_list() == shifted.to_list()

    r = ak.rotl(pytest.bitops_data["EDGE_CASES"], ak.array([1, 1, 1]))
    assert r.to_list() == [1, -1, -2]

    # scalar <<< vector
    rotated = ak.rotl(-(2**63), pytest.bitops_data["A"])
    ans = [-(2**63), 1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert rotated.to_list() == ans


def test_rotr():
    # vector <<< scalar
    rotated = (1024 * pytest.bitops_data["A"]).rotr(5)
    shifted = (1024 * pytest.bitops_data["A"]) >> 5
    # No wraparound, so these should be equal
    assert rotated.to_list() == shifted.to_list()

    r = ak.rotr(pytest.bitops_data["EDGE_CASES"], 1)
    assert r.to_list() == [2**62, -1, -(2**62) - 1]

    # vector <<< vector
    rotated = (1024 * pytest.bitops_data["A"]).rotr(pytest.bitops_data["A"])
    shifted = (1024 * pytest.bitops_data["A"]) >> pytest.bitops_data["A"]
    # No wraparound, so these should be equal
    assert rotated.to_list() == shifted.to_list()

    r = ak.rotr(pytest.bitops_data["EDGE_CASES"], ak.array([1, 1, 1]))
    assert r.to_list() == [2**62, -1, -(2**62) - 1]

    # scalar <<< vector
    rotated = ak.rotr(1, pytest.bitops_data["A"])
    ans = [1, -(2**63), 2**62, 2**61, 2**60, 2**59, 2**58, 2**57, 2**56, 2**55]
    assert rotated.to_list() == ans
