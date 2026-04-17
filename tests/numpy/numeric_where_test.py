import warnings

from typing import Any, Tuple, Union

import numpy as np
import pytest

import arkouda as ak

from arkouda import Categorical, Strings, pdarray


warnings.simplefilter("always", UserWarning)

DTYPES = ("uint64", "int64", "float64", "bool")
FORMS = ("array", "scalar")

# Representative multidim shapes (rank 2 and rank 3)
MD_SHAPES: Tuple[Tuple[int, ...], ...] = (
    (2, 3),
    (3, 2),
    (4, 1),
    (2, 2, 3),
)

# Broadcasting-focused shapes
BC_SHAPES: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]], ...] = (
    # (cond_shape, a_shape, b_shape) -> all broadcast to (2, 3)
    ((2, 1), (1, 3), (2, 3)),
    ((1, 3), (2, 1), (2, 3)),
    ((2, 1), (2, 3), (1, 3)),
    # 3D broadcast to (2, 2, 3)
    ((2, 1, 3), (1, 2, 1), (2, 2, 3)),
)

ShapeLike = Union[int, Tuple[int, ...]]


def _normalize_shape(shape: ShapeLike) -> Tuple[int, ...]:
    return (shape,) if isinstance(shape, int) else shape


def _numel(shape: ShapeLike) -> int:
    shp = _normalize_shape(shape)
    return int(np.prod(shp))


def _make_np_value(dtype: str, shape: ShapeLike, which: str) -> np.ndarray:
    """
    Deterministic-ish test values:
      - A in low range, B in higher range (for ints)
      - float B shifted by +10
    """
    shp = _normalize_shape(shape)
    n = _numel(shp)

    if dtype == "uint64":
        low, high = (0, 10) if which == "A" else (10, 20)
        return np.random.randint(low, high, n, dtype=np.uint64).reshape(shp)

    if dtype == "int64":
        low, high = (0, 10) if which == "A" else (10, 20)
        return np.random.randint(low, high, n, dtype=np.int64).reshape(shp)

    if dtype == "float64":
        base = np.random.standard_normal(n).astype(np.float64).reshape(shp)
        return base if which == "A" else (base + 10.0)

    if dtype == "bool":
        return np.random.randint(0, 2, n).astype(bool).reshape(shp)

    raise ValueError(dtype)


def _make_np_cond(shape: ShapeLike) -> np.ndarray:
    shp = _normalize_shape(shape)
    n = _numel(shp)
    return np.random.randint(0, 2, n).astype(bool).reshape(shp)


def _scalar_from(arr: np.ndarray) -> Any:
    """Return a Python scalar representative from an ndarray."""
    return arr.flat[0].item()


def _as_form(arr: np.ndarray, form: str) -> Any:
    """Return ndarray as-is or a Python scalar."""
    if form == "array":
        return arr
    return _scalar_from(arr)


def _to_ak_or_scalar(x: Any) -> Any:
    """Ndarray -> ak.array, scalar stays scalar."""
    return ak.array(x) if isinstance(x, np.ndarray) else x


def _ak_to_np(x: Any) -> Any:
    """pdarray/Strings/Categorical -> ndarray (or scalar stays scalar)."""
    if isinstance(x, (pdarray, Strings)):
        return x.to_ndarray()
    if isinstance(x, Categorical):
        # compare categoricals via strings to avoid code/category-order differences
        return x.to_strings().to_ndarray()
    return x


def _is_numpy_arraylike(x: Any) -> bool:
    """True for numpy ndarrays that represent array outputs (not 0-d scalars)."""
    return isinstance(x, np.ndarray) and x.ndim != 0


def _numpy_scalarize(x: Any) -> Any:
    """NumPy returns 0-d arrays for scalar/scalar where; normalize to Python scalar."""
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    return x


def _assert_equal(akres: Any, npres: Any, *, shape: Tuple[int, ...] | None = None) -> None:
    """
    Compare Arkouda result to NumPy result.
    - If array-like, compare shape and contents.
    - If scalar, compare scalar value.
    """
    akv = _ak_to_np(akres)
    npv = _numpy_scalarize(npres)

    if _is_numpy_arraylike(npv):
        assert isinstance(akv, np.ndarray)
        if shape is not None:
            assert akv.shape == npv.shape == shape
        if npv.dtype == bool:
            assert np.array_equal(akv, npv)
        else:
            assert np.allclose(akv, npv, equal_nan=True)
    else:
        assert not isinstance(akv, np.ndarray)
        assert akv == npv


@pytest.mark.requires_chapel_module("EfuncMsg")
class TestWhereNumeric:
    """
    Numeric/bool where tests.
    Focus: dispatch + scalar broadcasting semantics across:
      - condition: array/scalar
      - A/B: array/scalar
      - dtype_x/dtype_y combinations
    """

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype_x", DTYPES)
    @pytest.mark.parametrize("dtype_y", DTYPES)
    @pytest.mark.parametrize("cond_form", FORMS)
    @pytest.mark.parametrize("x_form", FORMS)
    @pytest.mark.parametrize("y_form", FORMS)
    def test_where_1d(self, size, dtype_x, dtype_y, cond_form, x_form, y_form):
        npX = _make_np_value(dtype_x, size, which="A")
        npY = _make_np_value(dtype_y, size, which="B")
        npCond = _make_np_cond(size)

        cond = _as_form(npCond, cond_form)
        x = _as_form(npX, x_form)
        y = _as_form(npY, y_form)

        akCond = _to_ak_or_scalar(cond)
        akX = _to_ak_or_scalar(x)
        akY = _to_ak_or_scalar(y)

        akres = ak.where(akCond, akX, akY)
        npres = np.where(cond, x, y)

        shape = _normalize_shape(size) if _is_numpy_arraylike(npres) else None
        _assert_equal(akres, npres, shape=shape)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("shape", MD_SHAPES)
    @pytest.mark.parametrize("dtype_x", DTYPES)
    @pytest.mark.parametrize("dtype_y", DTYPES)
    @pytest.mark.parametrize("cond_form", FORMS)
    @pytest.mark.parametrize("x_form", FORMS)
    @pytest.mark.parametrize("y_form", FORMS)
    def test_where_multidim(self, shape, dtype_x, dtype_y, cond_form, x_form, y_form):
        npX = _make_np_value(dtype_x, shape, which="A")
        npY = _make_np_value(dtype_y, shape, which="B")

        # Avoid the pure scalar path (doesn't test multidim semantics)
        if cond_form == "scalar" and x_form == "scalar" and y_form == "scalar":
            pytest.skip("All-scalar inputs do not exercise multidimensional where semantics")

        if cond_form == "array":
            cond = _make_np_cond(shape)
        else:
            cond = bool(np.random.randint(0, 2))

        x = npX if x_form == "array" else _scalar_from(npX)
        y = npY if y_form == "array" else _scalar_from(npY)

        akCond = _to_ak_or_scalar(cond)
        akX = _to_ak_or_scalar(x)
        akY = _to_ak_or_scalar(y)

        akres = ak.where(akCond, akX, akY)
        npres = np.where(cond, x, y)

        if _is_numpy_arraylike(npres):
            _assert_equal(akres, npres, shape=shape)
        else:
            _assert_equal(akres, npres)

    def test_type_errors(self):
        with pytest.raises(TypeError):
            ak.where([0], ak.linspace(1, 10, 10), ak.linspace(1, 10, 10))

        with pytest.raises(TypeError):
            ak.where(ak.linspace(1, 10, 10) > 5, [0], ak.linspace(1, 10, 10))

        with pytest.raises(TypeError):
            ak.where(ak.linspace(1, 10, 10) > 5, ak.linspace(1, 10, 10), [0])

    def test_non_broadcastable_shapes_raise(self):
        # Not broadcastable: (10,) vs (11,)
        cond = (ak.arange(10) % 2) == 0
        a = ak.ones(11, dtype=ak.int64)
        b = ak.ones(10, dtype=ak.int64)
        with pytest.raises(ValueError):
            ak.where(cond, a, b)


@pytest.mark.requires_chapel_module("EfuncMsg")
class TestWhereBroadcasting:
    """
    Explicit broadcasting tests (NumPy semantics):
      - condition, A, B broadcast to a mutual shape when possible.
    These will fail until arkouda.where implements A/B/condition broadcasting.
    """

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("dtype_x", DTYPES)
    @pytest.mark.parametrize("dtype_y", DTYPES)
    @pytest.mark.parametrize("cond_shape,a_shape,b_shape", BC_SHAPES)
    def test_where_broadcasts_arrays(self, dtype_x, dtype_y, cond_shape, a_shape, b_shape):
        npCond = _make_np_cond(cond_shape)
        npA = _make_np_value(dtype_x, a_shape, which="A")
        npB = _make_np_value(dtype_y, b_shape, which="B")

        akCond = ak.array(npCond)
        akA = ak.array(npA)
        akB = ak.array(npB)

        akres = ak.where(akCond, akA, akB)
        npres = np.where(npCond, npA, npB)

        # Determine expected broadcasted shape from numpy result
        assert _is_numpy_arraylike(npres)
        _assert_equal(akres, npres, shape=npres.shape)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_where_broadcasts_with_scalar_operand(self):
        # condition (2,1), A scalar, B (1,3) => result (2,3)
        npCond = _make_np_cond((2, 1))
        npB = _make_np_value("int64", (1, 3), which="B")
        a_scalar = int(_scalar_from(_make_np_value("int64", (2, 1), which="A")))

        akCond = ak.array(npCond)
        akB = ak.array(npB)

        akres = ak.where(akCond, a_scalar, akB)
        npres = np.where(npCond, a_scalar, npB)

        assert _is_numpy_arraylike(npres)
        _assert_equal(akres, npres, shape=npres.shape)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_where_broadcast_failure_raises(self):
        # Not broadcastable: (2,3) and (2,2)
        npCond = _make_np_cond((2, 3))
        npA = _make_np_value("int64", (2, 2), which="A")
        npB = _make_np_value("int64", (2, 3), which="B")

        akCond = ak.array(npCond)
        akA = ak.array(npA)
        akB = ak.array(npB)

        # NumPy raises ValueError; Arkouda should too if matching NumPy semantics
        with pytest.raises(ValueError):
            ak.where(akCond, akA, akB)


@pytest.mark.requires_chapel_module("EfuncMsg")
class TestWhereClauseExamples:
    def test_comparison_clause_array_array(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        npres = np.where(n1 < 5, n1, n2)
        akres = ak.where(a1 < 5, a1, a2).to_ndarray()
        assert np.array_equal(akres, npres)

    def test_comparison_clause_array_scalar(self):
        n1 = np.arange(1, 10)
        a1 = ak.array(n1)

        npres = np.where(n1 > 5, n1, 1)
        akres = ak.where(a1 > 5, a1, 1).to_ndarray()
        assert np.array_equal(akres, npres)

        npres = np.where(n1 > 5, 1, n1)
        akres = ak.where(a1 > 5, 1, a1).to_ndarray()
        assert np.array_equal(akres, npres)

    def test_where_filter_matches_numpy(self):
        n1 = np.arange(1, 10)
        a1 = ak.array(n1)
        assert np.array_equal(n1[n1 > 5], a1[a1 > 5].to_ndarray())

    def test_multiple_where_clauses_not_supported(self):
        a1 = ak.arange(1, 10)
        a2 = ak.ones(9, dtype=ak.int64)

        cond = (a1 > 5, a1 < 8)
        with pytest.raises(TypeError):
            ak.where(cond, a1, a2)


@pytest.mark.requires_chapel_module("EfuncMsg")
class TestWhereCategorical:
    """
    Minimal, high-value categorical coverage without exploding combinations.
    """

    def test_categorical_with_array_condition(self):
        n = 10
        cond = (ak.arange(n) % 2) == 0

        c1 = ak.Categorical(ak.array([f"a{i}" for i in range(n)]))
        c2 = ak.Categorical(ak.array([f"b{i}" for i in range(n)]))

        out = ak.where(cond, c1, c2)
        assert isinstance(out, Categorical)

        np_cond = cond.to_ndarray()
        np1 = np.array([f"a{i}" for i in range(n)], dtype=object)
        np2 = np.array([f"b{i}" for i in range(n)], dtype=object)
        npres = np.where(np_cond, np1, np2)

        assert np.array_equal(out.to_strings().to_ndarray(), npres.astype(str))

    def test_categorical_scalar_condition_broadcasts(self):
        n = 8
        c1 = ak.Categorical(ak.array([f"x{i}" for i in range(n)]))
        c2 = ak.Categorical(ak.array([f"y{i}" for i in range(n)]))

        out_true = ak.where(True, c1, c2)
        out_false = ak.where(False, c1, c2)

        assert isinstance(out_true, Categorical)
        assert isinstance(out_false, Categorical)
        assert out_true.size == n
        assert out_false.size == n

        assert np.array_equal(out_true.to_strings().to_ndarray(), c1.to_strings().to_ndarray())
        assert np.array_equal(out_false.to_strings().to_ndarray(), c2.to_strings().to_ndarray())
