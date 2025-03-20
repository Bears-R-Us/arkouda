#!/usr/bin/env python3

import argparse
import random
import string
import sys
import time

NINPUTS = 1_000_000
INPUTSIZE = 8


def test_hash(N, I):
    S = []
    for i in range(N):
        s = "".join(random.choices(string.ascii_letters, k=I))
        S.append(s)

    start = time.time()
    x = set(S)
    t = time.time() - start

    print(f"Hashed {NINPUTS} blocks ({NINPUTS * INPUTSIZE} bytes) in {t:.3f} seconds")
    print(f"Rate = {(NINPUTS * INPUTSIZE) / (1024 * 1024 * t):.2f} MB/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--NINPUTS", type=int, default=NINPUTS, help="Number of inputs to hash")
    parser.add_argument("--INPUTSIZE", type=int, default=INPUTSIZE, help="Size of each input.")
    args = parser.parse_args()
    test_hash(args.NINPUTS, args.INPUTSIZE)
