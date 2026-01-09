import numpy as np
import pytest

import arkouda as ak


def _np_struct_from_cols(cols: list[np.ndarray]) -> np.ndarray:
    """
    Build a NumPy structured array representing "rows" from multiple 1D columns.
    This lets us use np.union1d/intersect1d/setdiff1d/setxor1d on rows.
    """
    assert len(cols) >= 1
    n = len(cols[0])
    for c in cols[1:]:
        assert len(c) == n

    dtype = [(f"f{i}", cols[i].dtype) for i in range(len(cols))]
    out = np.empty(n, dtype=dtype)
    for i, c in enumerate(cols):
        out[f"f{i}"] = c
    return out


def _np_setop_rows(op, A_cols, B_cols):
    """
    Compute numpy reference for multi-column setops by treating rows as structured scalars.
    op: one of np.union1d, np.intersect1d, np.setdiff1d, np.setxor1d
    Returns list of numpy arrays (one per column), sorted lexicographically by row.
    """
    A_rows = _np_struct_from_cols(A_cols)
    B_rows = _np_struct_from_cols(B_cols)
    rows = op(A_rows, B_rows)

    # Sort rows to match arkouda's "sorted unique" intent (and stable comparisons)
    rows = np.sort(rows)

    # De-structure back into columns
    out_cols = [rows[f"f{i}"] for i in range(len(A_cols))]
    return out_cols


@pytest.mark.requires_chapel_module("In1dMsg")
@pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
@pytest.mark.parametrize("n", [0, 1, 2, 10, 100])
def test_in1d_matches_numpy(dtype, n):
    rng = np.random.default_rng(12345)
    a_np = rng.integers(0, 20, size=n, dtype=np.int64)
    b_np = rng.integers(0, 20, size=max(n // 2, 1), dtype=np.int64)

    # Cast for uint64 cases
    if dtype == ak.uint64:
        a_np = a_np.astype(np.uint64, copy=False)
        b_np = b_np.astype(np.uint64, copy=False)

    a = ak.array(a_np)
    b = ak.array(b_np)

    got = ak.in1d(a, b)
    exp = np.in1d(a_np, b_np, assume_unique=False, invert=False)

    assert np.array_equal(got.to_ndarray(), exp)

    got_inv = ak.in1d(a, b, invert=True)
    exp_inv = np.in1d(a_np, b_np, assume_unique=False, invert=True)
    assert np.array_equal(got_inv.to_ndarray(), exp_inv)


@pytest.mark.requires_chapel_module("In1dMsg")
@pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
def test_in1d_symmetric_matches_numpy(dtype):
    rng = np.random.default_rng(2468)
    a_np = rng.integers(0, 30, size=50, dtype=np.int64)
    b_np = rng.integers(0, 30, size=40, dtype=np.int64)
    if dtype == ak.uint64:
        a_np = a_np.astype(np.uint64, copy=False)
        b_np = b_np.astype(np.uint64, copy=False)

    a = ak.array(a_np)
    b = ak.array(b_np)

    got_a, got_b = ak.in1d(a, b, symmetric=True)
    exp_a = np.in1d(a_np, b_np)
    exp_b = np.in1d(b_np, a_np)
    assert np.array_equal(got_a.to_ndarray(), exp_a)
    assert np.array_equal(got_b.to_ndarray(), exp_b)


@pytest.mark.requires_chapel_module("In1dMsg")
@pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
def test_in1d_assume_unique_raises_when_not_unique(dtype):
    # Arkouda explicitly validates uniqueness when assume_unique=True for multi-array path,
    # and raises NonUniqueError. This test targets that behavior.
    from arkouda.numpy.alignment import NonUniqueError

    a_np = np.array([1, 1, 2, 3], dtype=np.int64)
    b_np = np.array([1, 2, 4], dtype=np.int64)
    if dtype == ak.uint64:
        a_np = a_np.astype(np.uint64, copy=False)
        b_np = b_np.astype(np.uint64, copy=False)

    a = ak.array(a_np)
    b = ak.array(b_np)

    # For scalar pdarray path, arkouda routes through _in1d_single,
    # which does not validate uniqueness; so this test uses multi-array
    # mode (sequence-of-arrays) which does validate.
    A = [a]
    B = [b]
    with pytest.raises(NonUniqueError):
        ak.in1d(A, B, assume_unique=True)


@pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
@pytest.mark.parametrize("n1,n2", [(0, 0), (0, 10), (10, 0), (10, 10), (50, 40)])
def test_union1d_matches_numpy(dtype, n1, n2):
    if (n1 == 0 and n2 > 0) or (n2 == 0 and n1 > 0):
        pytest.xfail(
            "Known bug: ak.union1d returns non-unique/unsorted when one input is empty; "
            "should match np.union1d (sorted unique). Issue #5273."
        )

    rng = np.random.default_rng(999)
    a_np = rng.integers(0, 25, size=n1, dtype=np.int64)
    b_np = rng.integers(0, 25, size=n2, dtype=np.int64)
    if dtype == ak.uint64:
        a_np = a_np.astype(np.uint64, copy=False)
        b_np = b_np.astype(np.uint64, copy=False)

    a = ak.array(a_np)
    b = ak.array(b_np)

    got = ak.union1d(a, b)
    exp = np.union1d(a_np, b_np)
    assert np.array_equal(got.to_ndarray(), exp)


@pytest.mark.xfail(
    reason="Known bug: ak.union1d returns non-unique/unsorted when one input is empty; "
    "should match np.union1d (sorted unique).. Issue #5273.",
    strict=False,
)
def test_union1d_empty_left_matches_numpy():
    b_np = np.array([20, 19, 4, 4, 4, 17, 2, 18, 3, 4], dtype=np.int64)
    got = ak.union1d(ak.array(np.array([], dtype=np.int64)), ak.array(b_np))
    assert np.array_equal(got.to_ndarray(), np.union1d(np.array([], dtype=np.int64), b_np))


@pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
@pytest.mark.parametrize("assume_unique", [False, True])
def test_intersect1d_matches_numpy(dtype, assume_unique):
    rng = np.random.default_rng(2024)
    a_np = rng.integers(0, 40, size=100, dtype=np.int64)
    b_np = rng.integers(0, 40, size=80, dtype=np.int64)
    if dtype == ak.uint64:
        a_np = a_np.astype(np.uint64, copy=False)
        b_np = b_np.astype(np.uint64, copy=False)

    if assume_unique:
        a_ref = np.unique(a_np)
        b_ref = np.unique(b_np)

        a_ak = ak.array(a_ref)
        b_ak = ak.array(b_ref)

        got = ak.intersect1d(a_ak, b_ak, assume_unique=True)
        exp = np.intersect1d(a_ref, b_ref, assume_unique=True)
    else:
        a_ak = ak.array(a_np)
        b_ak = ak.array(b_np)

        got = ak.intersect1d(a_ak, b_ak, assume_unique=False)
        exp = np.intersect1d(a_np, b_np, assume_unique=False)

    assert np.array_equal(got.to_ndarray(), exp)


@pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
@pytest.mark.parametrize("assume_unique", [False, True])
def test_setdiff1d_matches_numpy(dtype, assume_unique):
    rng = np.random.default_rng(777)
    a_np = rng.integers(0, 50, size=120, dtype=np.int64)
    b_np = rng.integers(0, 50, size=70, dtype=np.int64)

    if dtype == ak.uint64:
        a_np = a_np.astype(np.uint64, copy=False)
        b_np = b_np.astype(np.uint64, copy=False)

    if assume_unique:
        a_ref = np.unique(a_np)
        b_ref = np.unique(b_np)

        got = ak.setdiff1d(ak.array(a_ref), ak.array(b_ref), assume_unique=True)
        exp = np.setdiff1d(a_ref, b_ref, assume_unique=True)
    else:
        got = ak.setdiff1d(ak.array(a_np), ak.array(b_np), assume_unique=False)
        exp = np.setdiff1d(a_np, b_np, assume_unique=False)

    assert np.array_equal(got.to_ndarray(), exp)


@pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
@pytest.mark.parametrize("assume_unique", [False, True])
def test_setxor1d_matches_numpy(dtype, assume_unique):
    rng = np.random.default_rng(31415)
    a_np = rng.integers(0, 60, size=100, dtype=np.int64)
    b_np = rng.integers(0, 60, size=90, dtype=np.int64)
    if dtype == ak.uint64:
        a_np = a_np.astype(np.uint64, copy=False)
        b_np = b_np.astype(np.uint64, copy=False)

    if assume_unique:
        a_ref = np.unique(a_np)
        b_ref = np.unique(b_np)

        got = ak.setxor1d(ak.array(a_ref), ak.array(b_ref), assume_unique=True)
        exp = np.setxor1d(a_ref, b_ref, assume_unique=True)
    else:
        got = ak.setxor1d(ak.array(a_np), ak.array(b_np), assume_unique=False)
        exp = np.setxor1d(a_np, b_np, assume_unique=False)

    assert np.array_equal(got.to_ndarray(), exp)


@pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
def test_concatenate_ordered_matches_numpy(dtype):
    rng = np.random.default_rng(123)
    parts = [rng.integers(0, 100, size=s, dtype=np.int64) for s in [0, 5, 1, 10]]
    if dtype == ak.uint64:
        parts = [p.astype(np.uint64, copy=False) for p in parts]

    ak_parts = [ak.array(p) for p in parts]
    got = ak.concatenate(ak_parts, ordered=True)
    exp = np.concatenate(parts, axis=0)

    assert np.array_equal(got.to_ndarray(), exp)


@pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
def test_concatenate_unordered_is_multiset_equal(dtype):
    rng = np.random.default_rng(456)
    parts = [rng.integers(0, 50, size=s, dtype=np.int64) for s in [3, 7, 0, 9]]
    if dtype == ak.uint64:
        parts = [p.astype(np.uint64, copy=False) for p in parts]

    ak_parts = [ak.array(p) for p in parts]
    got = ak.concatenate(ak_parts, ordered=False)
    exp = np.concatenate(parts, axis=0)

    # unordered concatenate may interleave; compare as multisets
    assert np.array_equal(np.sort(got.to_ndarray()), np.sort(exp))


@pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
def test_multiarray_union_intersect_setdiff_setxor_align(dtype):
    rng = np.random.default_rng(8888)

    # 2-column "rows"
    n1, n2 = 60, 55
    a1 = rng.integers(0, 20, size=n1, dtype=np.int64)
    a2 = rng.integers(0, 20, size=n1, dtype=np.int64)
    b1 = rng.integers(0, 20, size=n2, dtype=np.int64)
    b2 = rng.integers(0, 20, size=n2, dtype=np.int64)

    if dtype == ak.uint64:
        a1, a2, b1, b2 = [x.astype(np.uint64, copy=False) for x in (a1, a2, b1, b2)]

    A = [ak.array(a1), ak.array(a2)]
    B = [ak.array(b1), ak.array(b2)]

    # union1d (multi)
    got_u = ak.union1d(A, B)
    exp_u = _np_setop_rows(np.union1d, [a1, a2], [b1, b2])
    assert np.array_equal(got_u[0].to_ndarray(), exp_u[0])
    assert np.array_equal(got_u[1].to_ndarray(), exp_u[1])

    # intersect1d (multi)
    got_i = ak.intersect1d(A, B, assume_unique=False)
    exp_i = _np_setop_rows(np.intersect1d, [a1, a2], [b1, b2])
    assert np.array_equal(got_i[0].to_ndarray(), exp_i[0])
    assert np.array_equal(got_i[1].to_ndarray(), exp_i[1])

    # setdiff1d (multi): A - B
    got_d = ak.setdiff1d(A, B, assume_unique=False)
    exp_d = _np_setop_rows(np.setdiff1d, [a1, a2], [b1, b2])
    assert np.array_equal(got_d[0].to_ndarray(), exp_d[0])
    assert np.array_equal(got_d[1].to_ndarray(), exp_d[1])

    # setxor1d (multi)
    got_x = ak.setxor1d(A, B, assume_unique=False)
    exp_x = _np_setop_rows(np.setxor1d, [a1, a2], [b1, b2])
    assert np.array_equal(got_x[0].to_ndarray(), exp_x[0])
    assert np.array_equal(got_x[1].to_ndarray(), exp_x[1])


@pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
def test_indexof1d_all_occurrences_remove_missing(dtype):
    rng = np.random.default_rng(13579)
    space_np = rng.integers(0, 10, size=50, dtype=np.int64)
    query_np = rng.integers(0, 10, size=20, dtype=np.int64)

    # Force some missing values by shifting query range
    query_np = (query_np + 50).astype(np.int64)

    # Insert some present values as well
    query_np[:5] = space_np[:5]

    if dtype == ak.uint64:
        space_np = space_np.astype(np.uint64, copy=False)
        query_np = query_np.astype(np.uint64, copy=False)

    space = ak.array(space_np)
    query = ak.array(query_np)

    got = ak.indexof1d(query, space)
    got_np = got.to_ndarray()

    # Reference: for each query value, emit indices of all matches in space; skip if none.
    exp_list = []
    for q in query_np:
        hits = np.nonzero(space_np == q)[0]
        if hits.size:
            exp_list.extend(hits.tolist())
    exp = np.array(exp_list, dtype=np.int64)

    assert np.array_equal(got_np, exp)
