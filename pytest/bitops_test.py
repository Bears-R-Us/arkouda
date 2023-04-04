import numpy as np

import arkouda as ak
import pytest

A = B = EDGE_CASES = EDGE_CASES_UINT = None


def set_globals():
    global A, B, EDGE_CASES, EDGE_CASES_UINT

    A = ak.arange(10)
    B = ak.cast(A, ak.uint64)
    EDGE_CASES = ak.array([-(2**63), -1, 2**63 - 1])
    EDGE_CASES_UINT = ak.cast(ak.array([-(2**63), -1, 2**63 - 1]), ak.uint64)


def test_popcount():
    if A is None:
        set_globals()

    # Method invocation
    # Toy input
    ans = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2]
    assert A.popcount().to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert B.popcount().to_list() == ans.tolist()

    # Function invocation
    # Edge case input
    ans = [1, 64, 63]
    assert ak.popcount(EDGE_CASES).to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert ak.popcount(EDGE_CASES_UINT).to_list() == ans.tolist()


def test_parity():
    if A is None:
        set_globals()

    ans = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
    assert A.parity().to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert B.parity().to_list() == ans.tolist()

    ans = [1, 0, 1]
    assert ak.parity(EDGE_CASES).to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert ak.parity(EDGE_CASES_UINT).to_list() == ans.tolist()


def test_clz():
    if A is None:
        set_globals()

    ans = [64, 63, 62, 62, 61, 61, 61, 61, 60, 60]
    assert A.clz().to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert B.clz().to_list() == ans.tolist()

    ans = [0, 0, 1]
    assert ak.clz(EDGE_CASES).to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert ak.clz(EDGE_CASES_UINT).to_list() == ans.tolist()


def test_ctz():
    if A is None:
        set_globals()

    ans = [0, 0, 1, 0, 2, 0, 1, 0, 3, 0]
    assert A.ctz().to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert B.ctz().to_list() == ans.tolist()

    ans = [63, 0, 0]
    assert ak.ctz(EDGE_CASES).to_list() == ans

    # Test uint case
    ans = np.array(ans, ak.uint64)
    assert ak.ctz(EDGE_CASES_UINT).to_list() == ans.tolist()


def test_dtypes():
    if A is None:
        set_globals()

    f = ak.zeros(10, dtype=ak.float64)
    with pytest.raises(TypeError):
        f.popcount()

    with pytest.raises(TypeError):
        ak.popcount(f)


def test_rotl():
    if A is None:
        set_globals()

    # vector <<< scalar
    rotated = A.rotl(5)
    shifted = A << 5
    # No wraparound, so these should be equal
    assert rotated.to_list() == shifted.to_list()

    r = ak.rotl(EDGE_CASES, 1)
    assert r.to_list() == [1, -1, -2]

    # vector <<< vector
    rotated = A.rotl(A)
    shifted = A << A
    # No wraparound, so these should be equal
    assert rotated.to_list() == shifted.to_list()

    r = ak.rotl(EDGE_CASES, ak.array([1, 1, 1]))
    assert r.to_list() == [1, -1, -2]

    # scalar <<< vector
    rotated = ak.rotl(-(2**63), A)
    ans = [-(2**63), 1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert rotated.to_list() == ans


def test_rotr():
    if A is None:
        set_globals()

    # vector <<< scalar
    rotated = (1024 * A).rotr(5)
    shifted = (1024 * A) >> 5
    # No wraparound, so these should be equal
    assert rotated.to_list() == shifted.to_list()

    r = ak.rotr(EDGE_CASES, 1)
    assert r.to_list() == [2**62, -1, -(2**62) - 1]

    # vector <<< vector
    rotated = (1024 * A).rotr(A)
    shifted = (1024 * A) >> A
    # No wraparound, so these should be equal
    assert rotated.to_list() == shifted.to_list()

    r = ak.rotr(EDGE_CASES, ak.array([1, 1, 1]))
    assert r.to_list() == [2**62, -1, -(2**62) - 1]

    # scalar <<< vector
    rotated = ak.rotr(1, A)
    ans = [1, -(2**63), 2**62, 2**61, 2**60, 2**59, 2**58, 2**57, 2**56, 2**55]
    assert rotated.to_list() == ans
