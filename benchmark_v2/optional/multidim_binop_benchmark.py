import math
import operator

import pytest
import functools
import arkouda as ak

from benchmark_v2.benchmark_utils import calc_num_bytes


DTYPES = ("uint64", "bigint")
NDIMS = (1, 2, 3)
OPS = ("+", "-", "*", "/", "//", "&", "|", "^")


@functools.cache
def choose_shape(n: int, ndim: int) -> tuple[int, ...]:
    """
    Choose an ``ndim``-dimensional shape whose element count is as close as possible
    to ``n`` **without exceeding it**, while keeping dimensions as even as possible.

    The returned shape has:
      - ``prod(shape) <= n`` (unless ``n < 1``, in which case a minimal shape is used)
      - minimal dimension spread (``max(shape) - min(shape)``), with ties broken by
        maximizing ``prod(shape)`` (i.e., minimizing ``n - prod(shape)``).

    Parameters
    ----------
    n : int
        Target maximum number of elements. The resulting shape will satisfy
        ``prod(shape) <= n`` when possible.
    ndim : int
        Number of dimensions. Supported values are 1, 2, and 3.

    Returns
    -------
    tuple[int, ...]
        A tuple of length ``ndim`` representing the chosen shape.

    Raises
    ------
    ValueError
        If ``ndim`` is not one of {1, 2, 3}.

    Examples
    --------
    >>> choose_shape(36, 3)
    (3, 3, 4)
    """
    if ndim == 1:
        return (max(1, int(n)),)

    if ndim == 2:
        root = int(math.isqrt(max(1, n)))
        best = None
        # search around sqrt(n)
        for a in range(max(1, root - 64), root + 65):
            b = n // a
            dims = tuple(sorted((a, b)))
            prod = dims[0] * dims[1]
            spread = dims[1] - dims[0]
            overshoot = prod - n
            score = spread * 1_000_000 + overshoot
            cand = (score, dims)
            if best is None or cand < best:
                best = cand
        return best[1]

    if ndim == 3:
        root = int(round(max(1, n) ** (1 / 3)))
        best = None
        # search a,b around cube-root; compute c as ceil(n/(a*b))
        for a in range(max(1, root - 64), root + 65):
            for b in range(max(1, root - 64), root + 65):
                ab = a * b
                if ab <= 0:
                    continue
                c = n // ab
                dims = tuple(sorted((a, b, max(1, c))))
                prod = dims[0] * dims[1] * dims[2]
                spread = dims[2] - dims[0]
                overshoot = prod - n
                score = spread * 1_000_000 + overshoot
                cand = (score, dims)
                if best is None or cand < best:
                    best = cand
        return best[1]

    raise ValueError(f"Unsupported ndim={ndim}")


def _make_uint64(shape: tuple[int, ...], seed: int):
    size = 1
    for d in shape:
        size *= d
    a = ak.randint(0, 2**64, size=size, dtype=ak.uint64, seed=seed)
    if len(shape) > 1:
        a = a.reshape(*shape)
    return a


def _make_bigint_2limb(shape: tuple[int, ...], seed: int):
    """Make a bigint array using exactly two uint64 limbs (hi, lo)."""
    size = 1
    for d in shape:
        size *= d

    hi = ak.randint(0, 2**64, size=size, dtype=ak.uint64, seed=seed)
    lo = ak.randint(0, 2**64, size=size, dtype=ak.uint64, seed=seed + 1)

    bi = ak.bigint_from_uint_arrays([hi, lo])
    if len(shape) > 1:
        bi = bi.reshape(*shape)
    return bi


def _make_arrays(shape: tuple[int, ...], dtype: str, seed: int):
    if dtype == "uint64":
        a = _make_uint64(shape, seed)
        b = _make_uint64(shape, seed + 10_000)
        return a, b
    elif dtype == "bigint":
        a = _make_bigint_2limb(shape, seed)
        b = _make_bigint_2limb(shape, seed + 10_000)
        return a, b
    else:
        raise ValueError(f"Unsupported dtype={dtype}")


def _get_binop(op: str):
    # Use Python operators so this works naturally on arkouda pdarrays.
    if op == "+":
        return operator.add
    if op == "-":
        return operator.sub
    if op == "*":
        return operator.mul
    if op == "/":
        return operator.truediv
    if op == "//":
        return operator.floordiv
    if op == "&":
        return operator.and_
    if op == "|":
        return operator.or_
    if op == "^":
        return operator.xor
    raise ValueError(f"Unknown op={op}")


@pytest.mark.skip_numpy(True)
@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.benchmark(group="AK_binop_ops")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ndim", NDIMS)
@pytest.mark.parametrize("op", OPS)
def bench_binop_ops(benchmark, dtype, ndim, op):
    """
    Benchmark binary operations on uint64 and bigint across 1D/2D/3D shapes.

    - Total element target is ~ pytest.prob_size * cfg["numLocales"]
    - Shapes are chosen to be as even as possible while keeping product close to N.
    - Bigint arrays are built from exactly two uint64 limbs via ak.bigint_from_uint_arrays.
    """
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]
    seed = pytest.seed or 0

    shape = choose_shape(N, ndim)
    a, b = _make_arrays(shape, dtype, seed)

    fn = _get_binop(op)

    bytes_a = calc_num_bytes(a)
    bytes_b = calc_num_bytes(b)
    num_bytes = bytes_a + bytes_b

    benchmark.pedantic(
        fn,
        args=[a, b],
        rounds=pytest.trials,
    )

    # metadata
    benchmark.extra_info["description"] = (
        f"Binary op '{op}' on dtype={dtype} with shape={shape} (target N={N}, "
        f"actual elements={math.prod(shape)})."
    )
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["shape"] = shape
    benchmark.extra_info["ndim"] = ndim
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["op"] = op
    benchmark.extra_info["num_bytes"] = num_bytes
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (num_bytes / benchmark.stats["mean"]) / 2**30
    )
