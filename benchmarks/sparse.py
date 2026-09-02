#!/usr/bin/env python3

import arkouda as ak
import argparse
import time
import numpy as np
from scipy.sparse import coo_array, csr_matrix, find
from arkouda.scipy.sparsematrix import create_sparse_matrix, sparse_matrix_matrix_mult

# TODO: Add support for 'float64'
TYPES = ("int64",)


def compare_scipy(left, right, rtol=1e-9, atol=0.0, equal_nan=False):
    if left.shape != right.shape:
        return False

    lr, lc, lv = find(left)
    rr, rc, rv = find(right)

    if len(lr) != len(rr) or not np.all(lr == rr):
        return False
    if len(lc) != len(rc) or not np.all(lc == rc):
        return False
    if not np.allclose(lv, rv, rtol=rtol, atol=atol, equal_nan=equal_nan):
        return False

    return True


def time_ak_sparse(N, trials, dtype, seed):
    print(">>> arkouda {} sparse".format(dtype))
    cfg = ak.get_config()
    print("numLocales = {}, numNodes {}, N = {:,}".format(cfg["numLocales"], cfg["numNodes"], N))

    nnz = N * 10

    rows = ak.randint(1, N - 1, nnz, seed=seed)
    cols = ak.randint(1, N - 1, nnz, seed=seed * 2)
    rows, cols = ak.unique([rows, cols])

    if dtype == "int64":
        vals = ak.randint(1, len(rows), len(rows), seed=seed * 3)
    elif dtype == "float64":
        vals = ak.uniform(len(rows), seed=seed * 3) + 0.5

    csr_times = []
    csc_times = []
    multiplication_times = []

    for i in range(trials):
        start = time.time()
        csr = create_sparse_matrix(N, rows, cols, vals, layout="CSR")
        csr_times.append(time.time() - start)

        start = time.time()
        csc = create_sparse_matrix(N, cols, rows, vals, layout="CSC")
        csc_times.append(time.time() - start)

        start = time.time()
        result = sparse_matrix_matrix_mult(csc, csr)
        multiplication_times.append(time.time() - start)

    print("Average CSR time = {:.4f} seconds".format(np.mean(csr_times)))
    print("Average CSC time = {:.4f} seconds".format(np.mean(csc_times)))
    print("Average CSCxCSR time = {:.4f} seconds".format(np.mean(multiplication_times)))


def time_np_sparse(N, trials, dtype, seed):
    print(">>> numpy {} sparse".format(dtype))
    print("N = {:,}".format(N))

    nnz = N * 10

    rows = ak.randint(1, N - 1, nnz, seed=seed)
    cols = ak.randint(1, N - 1, nnz, seed=seed * 2)
    rows, cols = ak.unique([rows, cols])

    if dtype == "int64":
        vals = ak.randint(1, len(rows), len(rows), seed=seed * 3)
    elif dtype == "float64":
        vals = ak.uniform(len(rows), seed=seed * 3) + 0.5

    rows = rows.to_ndarray()
    cols = cols.to_ndarray()
    vals = vals.to_ndarray()

    csr_times = []
    csc_times = []
    multiplication_times = []

    for i in range(trials):
        start = time.time()
        csr = coo_array((vals, (rows, cols)), shape=(N, N)).tocsr()
        csr_times.append(time.time() - start)

        start = time.time()
        csc = coo_array((vals, (cols, rows)), shape=(N, N)).tocsc()
        csc_times.append(time.time() - start)

        start = time.time()
        result = csc.dot(csr).tocsr()
        multiplication_times.append(time.time() - start)

    print("Average CSR time = {:.4f} seconds".format(np.mean(csr_times)))
    print("Average CSC time = {:.4f} seconds".format(np.mean(csc_times)))
    print("Average CSCxCSR time = {:.4f} seconds".format(np.mean(multiplication_times)))


def check_correctness(dtype):
    seed = 1234
    N = 10_000
    nnz = N * 10

    rows = ak.randint(1, N - 1, nnz, seed=seed)
    cols = ak.randint(1, N - 1, nnz, seed=seed * 2)
    rows, cols = ak.unique([rows, cols])

    if dtype == "int64":
        vals = ak.randint(1, len(rows), len(rows), seed=seed * 3)
    elif dtype == "float64":
        vals = ak.uniform(len(rows), seed=seed * 3) + 0.5

    csr = create_sparse_matrix(N, rows, cols, vals, layout="CSR")
    scipy_csr = coo_array(
        (vals.to_ndarray(), (rows.to_ndarray(), cols.to_ndarray())), shape=(N, N)
    ).tocsr()
    ak_csr = csr.to_scipy_sparse()
    assert compare_scipy(scipy_csr, ak_csr), "CSR matrices do not match"

    csc = create_sparse_matrix(N, cols, rows, vals, layout="CSC")
    scipy_csc = coo_array(
        (vals.to_ndarray(), (cols.to_ndarray(), rows.to_ndarray())), shape=(N, N)
    ).tocsc()
    ak_csc = csc.to_scipy_sparse()
    assert compare_scipy(scipy_csc, ak_csc), "CSC matrices do not match"

    result = sparse_matrix_matrix_mult(csc, csr)
    scipy_result = scipy_csc.dot(scipy_csr).tocsr()
    ak_result = result.to_scipy_sparse()
    assert compare_scipy(scipy_result, ak_result), "Multiplication results do not match"


def create_parser():
    parser = argparse.ArgumentParser(description="Benchmark sparse matrix creation and multiplication.")
    parser.add_argument("hostname", type=str, help="Name of the Arkouda server")
    parser.add_argument("port", type=int, help="Port of the Arkouda server")
    parser.add_argument("-n", "--size", type=int, default=(10**6), help="Size of the sparse matrices")
    parser.add_argument("-t", "--trials", type=int, default=3, help="Number of trials for benchmarking")
    parser.add_argument(
        "-d", "--dtype", default="int64", help="Dtype of array ({})".format(", ".join(TYPES))
    )
    parser.add_argument(
        "--numpy",
        default=False,
        action="store_true",
        help="Run the same operation in NumPy to compare performance.",
    )
    parser.add_argument(
        "--correctness-only",
        default=False,
        action="store_true",
        help="Only check correctness, not performance.",
    )
    parser.add_argument(
        "-s", "--seed", default=1234, type=int, help="Value to initialize random number generator"
    )
    return parser


if __name__ == "__main__":
    import sys

    args = create_parser().parse_args()

    if args.dtype not in TYPES:
        raise ValueError("Dtype must be {}, not {}".format("/".join(TYPES), args.dtype))
    ak.verbose = False
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        for dtype in TYPES:
            check_correctness(dtype)
        sys.exit(0)

    print("N = {:,}".format(args.size))
    print("number of trials = ", args.trials)

    time_ak_sparse(args.size, args.trials, args.dtype, args.seed)

    if args.numpy:
        time_np_sparse(args.size, args.trials, args.dtype, args.seed)

    sys.exit(0)
