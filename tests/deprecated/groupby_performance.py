from context import arkouda as ak
import numpy as np
from time import time
from base_test import ArkoudaTest

OPERATORS = ["sum", "min", "nunique"]


def generate_arrays(length, nkeys, nvals, dtype="int64"):
    keys = ak.randint(0, nkeys, length)
    if dtype == "int64":
        vals = ak.randint(0, nvals, length)
    elif dtype == "bool":
        vals = ak.zeros(length, dtype="bool")
        for i in np.random.randint(0, length, nkeys // 2):
            vals[i] = True
    else:
        vals = ak.linspace(-1, 1, length)
    return keys, vals


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 7:
        print(
            f"Usage: {sys.argv[0]} <server> <port> <strategy (0=global, 1=perLocale)> <length> <num_keys> <num_vals>"
        )
        sys.exit()
    per_locale = sys.argv[3] == "1"
    print("per_locale = ", per_locale)
    length = int(sys.argv[4])
    print("length     = ", length)
    nkeys = int(sys.argv[5])
    print("nkeys      = ", nkeys)
    nvals = int(sys.argv[6])
    print("nvals      = ", nvals)
    ak.connect(sys.argv[1], int(sys.argv[2]))
    print("Generating keys and vals...")
    start = time()
    keys, vals = generate_arrays(length, nkeys, nvals)
    print(f"{time() - start:.2f} seconds", end="\n\n")
    print("GroupBy...")
    start = time()
    g = ak.GroupBy(keys, per_locale)
    print(f"{time() - start:.2f} seconds", end="\n\n")
    for op in OPERATORS:
        print(f"Aggregate('{op}') ...")
        start = time()
        uk, rv = g.aggregate(vals, op)
        print(f"{time() - start:.2f} seconds", end="\n\n")
    sys.exit()
