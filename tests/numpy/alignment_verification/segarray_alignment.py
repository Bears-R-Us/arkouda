from __future__ import annotations

import math
import random

from typing import TypeVar

import numpy as np
import pytest

import arkouda as ak


T = TypeVar("T")


# ----------------------------
# Helpers: build / convert
# ----------------------------


def _lists_to_segarray(py_segs: list[list], *, dtype: str = "int64") -> ak.SegArray:
    """
    Build ak.SegArray from python list-of-lists.
    dtype: "int64", "float64", "bool", "str"
    """
    offsets = []
    flat = []

    ctr = 0
    for row in py_segs:
        offsets.append(ctr)
        flat.extend(row)
        ctr += len(row)

    segments = ak.array(np.array(offsets, dtype=np.int64))

    if dtype == "str":
        # ak.array(list[str]) -> Strings
        values = ak.array([str(x) for x in flat])
    elif dtype == "bool":
        values = ak.array(np.array(flat, dtype=np.bool_))
    elif dtype == "float64":
        values = ak.array(np.array(flat, dtype=np.float64))
    else:
        values = ak.array(np.array(flat, dtype=np.int64))

    return ak.SegArray(segments, values)


def _normalize_index(i: int, n: int) -> int:
    return i if i >= 0 else n + i


def _ref_getitem(py: list[list], idx):
    if isinstance(idx, slice):
        return py[idx]
    if isinstance(idx, (list, np.ndarray)):
        # boolean mask or integer index list
        if len(idx) == 0:
            return []
        if isinstance(idx[0], (bool, np.bool_)):
            return [row for row, keep in zip(py, idx) if bool(keep)]
        return [py[int(i)] for i in idx]
    if isinstance(idx, (int, np.integer)):
        return py[int(idx)]
    raise TypeError(type(idx))


# ----------------------------
# Reference ops
# ----------------------------


def _ref_concat_axis0(xs: list[list[list]]) -> list[list]:
    out = []
    for x in xs:
        out.extend(x)
    return out


def _ref_concat_axis1(xs: list[list[list]]) -> list[list]:
    if not xs:
        raise ValueError("empty")
    n = len(xs[0])
    if any(len(x) != n for x in xs):
        raise ValueError("sizes differ")
    out = []
    for i in range(n):
        row = []
        for x in xs:
            row.extend(x[i])
        out.append(row)
    return out


def _ref_suffixes(py: list[list], n: int, *, proper: bool):
    if proper:
        mask = [len(r) > n for r in py]
    else:
        mask = [len(r) >= n for r in py]
    rows = [r[-n:] for r, m in zip(py, mask) if m]
    # return columnar form (list of columns), like SegArray.get_suffixes does
    cols = [[] for _ in range(n)]
    for r in rows:
        for j in range(n):
            cols[j].append(r[j])
    return cols, mask


def _ref_prefixes(py: list[list], n: int, *, proper: bool):
    if proper:
        mask = [len(r) > n for r in py]
    else:
        mask = [len(r) >= n for r in py]
    rows = [r[:n] for r, m in zip(py, mask) if m]
    cols = [[] for _ in range(n)]
    for r in rows:
        for j in range(n):
            cols[j].append(r[j])
    return cols, mask


def _ref_ngrams(py: list[list], n: int):
    # returns columnar (n columns), plus origins
    cols = [[] for _ in range(n)]
    origins = []
    for i, row in enumerate(py):
        if len(row) < n:
            continue
        for start in range(len(row) - n + 1):
            for j in range(n):
                cols[j].append(row[start + j])
            origins.append(i)
    return cols, origins


def _ref_get_jth(py: list[list], j: int, *, compressed: bool, default=0):
    out = []
    mask = []
    for row in py:
        jj = j
        if jj < 0:
            jj = len(row) + jj
        ok = 0 <= jj < len(row)
        mask.append(ok)
        if compressed:
            if ok:
                out.append(row[jj])
        else:
            out.append(row[jj] if ok else default)
    return out, mask


def _ref_set_jth(py: list[list], idxs: list[int], j: int, vals):
    # vals may be scalar or list aligned with idxs
    out = [r[:] for r in py]
    is_scalar = not isinstance(vals, (list, tuple, np.ndarray))
    for k, i in enumerate(idxs):
        row = out[i]
        jj = j
        if jj < 0:
            jj = len(row) + jj
        if not (0 <= jj < len(row)):
            raise ValueError("Not all (i, j) in bounds")
        row[jj] = vals if is_scalar else vals[k]
    return out


def _ref_remove_repeats(py: list[list]):
    out = []
    multiplicity = []
    for row in py:
        if not row:
            out.append([])
            multiplicity.append([])
            continue
        nr = [row[0]]
        mult = [1]
        for x in row[1:]:
            if x == nr[-1]:
                mult[-1] += 1
            else:
                nr.append(x)
                mult.append(1)
        out.append(nr)
        multiplicity.append(mult)
    return out, multiplicity


def _ref_unique_sorted(py: list[list]):
    return [sorted(set(r)) for r in py]


def _ref_setop(a, b, op):
    out = []
    for ai, bi in zip(a, b, strict=True):
        aa = np.asarray(ai)
        bb = np.asarray(bi)

        if op == "union":
            rr = np.union1d(aa, bb)
        elif op == "intersect":
            rr = np.intersect1d(aa, bb)
        elif op == "setdiff":
            rr = np.setdiff1d(aa, bb)
        elif op == "setxor":
            rr = np.setxor1d(aa, bb)
        else:
            raise ValueError(op)

        out.append(rr.tolist())
    return out


def _ref_agg(py: list[list], op: str):
    out = []
    for row in py:
        if op == "sum":
            out.append(sum(row) if row else 0)
        elif op == "prod":
            p = 1
            for x in row:
                p *= x
            out.append(p if row else 1)
        elif op == "min":
            out.append(min(row) if row else 0)
        elif op == "max":
            out.append(max(row) if row else 0)
        elif op == "any":
            out.append(bool(any(row)))
        elif op == "all":
            out.append(bool(all(row)))
        elif op == "mean":
            out.append((sum(row) / len(row)) if row else math.nan)
        elif op == "nunique":
            out.append(len(set(row)))
        else:
            raise ValueError(op)
    return out


# ----------------------------
# Fixtures / generators
# ----------------------------


def _rand_segments(rng: random.Random, nseg: int, maxlen: int, *, dtype: str):
    py = []
    for _ in range(nseg):
        L = rng.randrange(0, maxlen + 1)
        if dtype == "str":
            row = [rng.choice(["a", "b", "c", "d", "aa", "bb"]) for _ in range(L)]
        elif dtype == "bool":
            row = [rng.choice([False, True]) for _ in range(L)]
        elif dtype == "float64":
            row = [rng.uniform(-5, 5) for _ in range(L)]
        else:
            row = [rng.randrange(-10, 11) for _ in range(L)]
        py.append(row)
    return py


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool", "str"])
def test_segarray_roundtrip_tolist(dtype):
    rng = random.Random(0)
    py = _rand_segments(rng, nseg=12, maxlen=7, dtype=dtype)
    seg = _lists_to_segarray(py, dtype=dtype)
    assert seg.tolist() == py


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool", "str"])
def test_segarray_getitem_int(dtype):
    rng = random.Random(1)
    py = _rand_segments(rng, nseg=15, maxlen=8, dtype=dtype)
    seg = _lists_to_segarray(py, dtype=dtype)

    for i in [0, 3, 7, len(py) - 1]:
        got = seg[i].to_ndarray().tolist()
        exp = _ref_getitem(py, i)
        assert got == exp


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool", "str"])
def test_segarray_getitem_slice(dtype):
    rng = random.Random(2)
    py = _rand_segments(rng, nseg=20, maxlen=6, dtype=dtype)
    seg = _lists_to_segarray(py, dtype=dtype)

    sl = slice(2, 15, 3)
    got = seg[sl].tolist()
    exp = _ref_getitem(py, sl)
    assert got == exp


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool", "str"])
def test_segarray_getitem_boolmask(dtype):
    rng = random.Random(3)
    py = _rand_segments(rng, nseg=17, maxlen=5, dtype=dtype)
    seg = _lists_to_segarray(py, dtype=dtype)

    mask = [rng.choice([False, True]) for _ in range(len(py))]
    akmask = ak.array(np.array(mask, dtype=np.bool_))
    got = seg[akmask].tolist()
    exp = _ref_getitem(py, mask)
    assert got == exp


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool", "str"])
def test_segarray_concat_axis0(dtype):
    rng = random.Random(4)
    pys = [_rand_segments(rng, nseg=6, maxlen=5, dtype=dtype) for _ in range(3)]
    segs = [_lists_to_segarray(p, dtype=dtype) for p in pys]

    got = ak.SegArray.concat(segs, axis=0).tolist()
    exp = _ref_concat_axis0(pys)
    assert got == exp


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool", "str"])
def test_segarray_concat_axis1(dtype):
    rng = random.Random(5)
    pys = [_rand_segments(rng, nseg=10, maxlen=4, dtype=dtype) for _ in range(3)]
    segs = [_lists_to_segarray(p, dtype=dtype) for p in pys]

    if dtype == "str":
        pytest.xfail("SegArray.concat(axis=1) does not yet support Strings. Issue #5279.")

    got = ak.SegArray.concat(segs, axis=1).tolist()
    exp = _ref_concat_axis1(pys)
    assert got == exp


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool", "str"])
def test_segarray_append_matches_concat(dtype):
    rng = random.Random(6)
    a = _rand_segments(rng, nseg=9, maxlen=5, dtype=dtype)
    b = _rand_segments(rng, nseg=9, maxlen=5, dtype=dtype)
    sa = _lists_to_segarray(a, dtype=dtype)
    sb = _lists_to_segarray(b, dtype=dtype)

    got0 = sa.append(sb, axis=0).tolist()
    exp0 = _ref_concat_axis0([a, b])
    assert got0 == exp0

    if dtype == "str":
        pytest.xfail("SegArray.append(axis=1) does not yet support Strings. Issue #5279.")

    got1 = sa.append(sb, axis=1).tolist()
    exp1 = _ref_concat_axis1([a, b])
    assert got1 == exp1


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool"])
def test_segarray_append_single_and_prepend_single(dtype):
    rng = random.Random(7)
    py = _rand_segments(rng, nseg=11, maxlen=6, dtype=dtype)
    seg = _lists_to_segarray(py, dtype=dtype)

    x = (
        [rng.randrange(-3, 4) for _ in range(len(py))]
        if dtype == "int64"
        else (
            [rng.uniform(-1, 1) for _ in range(len(py))]
            if dtype == "float64"
            else [rng.choice([False, True]) for _ in range(len(py))]
        )
    )
    akx = ak.array(np.array(x))

    got_app = seg.append_single(akx).tolist()
    exp_app = [row + [x[i]] for i, row in enumerate(py)]
    assert got_app == exp_app

    got_pre = seg.prepend_single(akx).tolist()
    exp_pre = [[x[i]] + row for i, row in enumerate(py)]
    assert got_pre == exp_pre


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool", "str"])
@pytest.mark.parametrize("proper", [True, False])
def test_segarray_prefixes_suffixes(dtype, proper):
    rng = random.Random(8)
    py = _rand_segments(rng, nseg=14, maxlen=7, dtype=dtype)
    seg = _lists_to_segarray(py, dtype=dtype)

    n = 3
    got_cols, got_mask = seg.get_prefixes(n, return_origins=True, proper=proper)
    exp_cols, exp_mask = _ref_prefixes(py, n, proper=proper)

    got_cols = [c.to_ndarray().tolist() for c in got_cols]
    assert got_cols == exp_cols
    assert got_mask.to_ndarray().tolist() == exp_mask

    got_cols, got_mask = seg.get_suffixes(n, return_origins=True, proper=proper)
    exp_cols, exp_mask = _ref_suffixes(py, n, proper=proper)

    got_cols = [c.to_ndarray().tolist() for c in got_cols]
    assert got_cols == exp_cols
    assert got_mask.to_ndarray().tolist() == exp_mask


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool", "str"])
def test_segarray_ngrams(dtype):
    rng = random.Random(9)
    py = _rand_segments(rng, nseg=12, maxlen=7, dtype=dtype)
    seg = _lists_to_segarray(py, dtype=dtype)

    n = 2
    got_cols, got_orig = seg.get_ngrams(n, return_origins=True)
    exp_cols, exp_orig = _ref_ngrams(py, n)

    got_cols = [c.to_ndarray().tolist() for c in got_cols]
    assert got_cols == exp_cols
    assert got_orig.to_ndarray().tolist() == exp_orig


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool", "str"])
@pytest.mark.parametrize("compressed", [True, False])
def test_segarray_get_jth(dtype, compressed):
    if dtype == "str" and not compressed:
        pytest.skip("SegArray only supports compressed=False for non-Strings")

    rng = random.Random(10)
    py = _rand_segments(rng, nseg=16, maxlen=6, dtype=dtype)
    seg = _lists_to_segarray(py, dtype=dtype)

    j = -1
    default = 0
    got, got_mask = seg.get_jth(j, return_origins=True, compressed=compressed, default=default)
    exp, exp_mask = _ref_get_jth(py, j, compressed=compressed, default=default)

    got_list = got.to_ndarray().tolist() if hasattr(got, "to_ndarray") else got.to_list()
    assert got_list == exp
    assert got_mask.to_ndarray().tolist() == exp_mask


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool"])
def test_segarray_set_jth(dtype):
    rng = random.Random(11)
    py = _rand_segments(rng, nseg=18, maxlen=6, dtype=dtype)
    seg = _lists_to_segarray(py, dtype=dtype)

    idxs = [i for i, row in enumerate(py) if len(row) >= 2][:6]  # ensure j=1 in-bounds
    assert idxs, "need some non-trivial segments for test"

    j = 1
    vals = (
        [rng.randrange(-9, 10) for _ in idxs]
        if dtype == "int64"
        else (
            [rng.uniform(-3, 3) for _ in idxs]
            if dtype == "float64"
            else [rng.choice([False, True]) for _ in idxs]
        )
    )

    akidx = ak.array(np.array(idxs, dtype=np.int64))
    akvals = ak.array(np.array(vals))

    seg.set_jth(akidx, j, akvals)
    got = seg.tolist()
    exp = _ref_set_jth(py, idxs, j, vals)
    assert got == exp


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool", "str"])
def test_segarray_remove_repeats(dtype):
    rng = random.Random(12)
    py = _rand_segments(rng, nseg=14, maxlen=10, dtype=dtype)
    # force some repeats
    for r in py:
        if len(r) >= 4:
            r[2] = r[1]
            r[3] = r[1]

    seg = _lists_to_segarray(py, dtype=dtype)
    got_nr = seg.remove_repeats(return_multiplicity=False)
    exp_nr, _ = _ref_remove_repeats(py)
    assert got_nr.tolist() == exp_nr

    got_nr, got_mult = seg.remove_repeats(return_multiplicity=True)
    exp_nr, exp_mult = _ref_remove_repeats(py)
    assert got_nr.tolist() == exp_nr
    assert got_mult.tolist() == exp_mult


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool", "str"])
def test_segarray_unique(dtype):
    rng = random.Random(13)
    py = _rand_segments(rng, nseg=15, maxlen=9, dtype=dtype)

    # Known bug:
    # SegArray.unique() fails when SegArray contains empty segments.
    # Empty segments are not represented in the underlying GroupBy, so
    # GroupBy.broadcast receives a values array of incorrect size.
    if any(len(row) == 0 for row in py):
        pytest.xfail("SegArray.unique() fails when SegArray contains empty segments. Issue #5280.")

    seg = _lists_to_segarray(py, dtype=dtype)

    got = seg.unique().tolist()
    exp = _ref_unique_sorted(py)
    assert got == exp


@pytest.mark.xfail(reason="SegArray.unique() fails when SegArray contains empty segments. Issue #5280.")
def test_segarray_unique_with_empty_segments_xfail():
    from arkouda import SegArray, array

    segments = array([0, 2, 2])
    values = array([1.0, 2.0, 2.0, 2.0, 3.0])

    seg = SegArray(segments, values)

    # Conceptually expected:
    # [[1.0, 2.0], [], [2.0, 3.0]]
    seg.unique()


@pytest.mark.parametrize("op", ["intersect", "union", "setdiff", "setxor"])
@pytest.mark.parametrize("dtype", ["int64", "str"])
def test_segarray_setops(dtype, op):
    rng = random.Random(14)
    a = _rand_segments(rng, nseg=12, maxlen=7, dtype=dtype)
    b = _rand_segments(rng, nseg=12, maxlen=7, dtype=dtype)

    sa = _lists_to_segarray(a, dtype=dtype)
    sb = _lists_to_segarray(b, dtype=dtype)

    try:
        if op == "intersect":
            result = sa.intersect(sb)
        elif op == "union":
            result = sa.union(sb)
        elif op == "setdiff":
            result = sa.setdiff(sb)
        else:
            result = sa.setxor(sb)

        got = result.tolist()

    except ValueError as e:
        # Known bug:
        # SegArray set-ops can construct an invalid `segments` array
        # (segment labels instead of offsets), causing SegArray.__init__
        # to raise "Segments must be unique and in sorted order".
        if "Segments must be unique and in sorted order" in str(e):
            pytest.xfail(f"SegArray.{op}() can construct invalid segments for some inputs. Issue #5281.")
        raise

    exp = _ref_setop(a, b, op)

    # Known mismatch: SegArray set-ops preserve stable input order,
    # while NumPy set-ops return sorted unique results.
    def _sorted_lists(x):
        return [sorted(seg) for seg in x]

    if _sorted_lists(got) == _sorted_lists(exp) and got != exp:
        pytest.xfail(f"SegArray.{op}() ordering differs from NumPy for dtype={dtype}")

    assert got == exp


@pytest.mark.xfail(
    reason="Known segarray aggregation mismatch vs NumPy (see issue #5283)",
    strict=False,
)
@pytest.mark.parametrize("dtype", ["int64", "bool"])
def test_segarray_aggregations(dtype):
    rng = random.Random(15)
    py = _rand_segments(rng, nseg=20, maxlen=7, dtype=dtype)
    seg = _lists_to_segarray(py, dtype=dtype)

    # numeric / bool alignment: these match the per-segment GroupBy aggregations
    if dtype == "int64":
        got_sum = seg.sum().to_ndarray().tolist()
        exp_sum = _ref_agg(py, "sum")
        assert got_sum == exp_sum

        got_prod = seg.prod().to_ndarray().tolist()
        exp_prod = _ref_agg(py, "prod")
        assert got_prod == exp_prod

        got_min = seg.min().to_ndarray().tolist()
        exp_min = _ref_agg(py, "min")
        assert got_min == exp_min

        got_max = seg.max().to_ndarray().tolist()
        exp_max = _ref_agg(py, "max")
        assert got_max == exp_max

        got_nuniq = seg.nunique().to_ndarray().tolist()
        exp_nuniq = _ref_agg(py, "nunique")
        assert got_nuniq == exp_nuniq

        got_mean = seg.mean().to_ndarray()
        exp_mean = np.array(_ref_agg(py, "mean"), dtype=np.float64)
        # mean on empty segments can be NaN depending on backend; compare with NaN-safe equality
        assert np.allclose(got_mean, exp_mean, equal_nan=True)

    # bool aggregations
    got_any = seg.any().to_ndarray().tolist()
    exp_any = _ref_agg(py, "any")
    assert got_any == exp_any

    got_all = seg.all().to_ndarray().tolist()
    exp_all = _ref_agg(py, "all")
    assert got_all == exp_all
