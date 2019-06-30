#!/usr/bin/env python3

import arkouda as ak
import numpy as np
import pandas as pd
SIZE = 10000
GROUPS = 64

def groupby_to_arrays(df, kname, vname, op):
    g = df.groupby(kname)[vname]
    agg = g.aggregate(op.replace('arg', 'idx'))
    return agg.index.values, agg.values

def make_arrays():
    keys = np.random.randint(0, GROUPS, SIZE)
    i = np.random.randint(0, SIZE//GROUPS, SIZE)
    f = np.random.randn(SIZE)
    b = (i % 2) == 0
    d = {'keys':keys, 'int64':i, 'float64':f, 'bool':b}
    return d

def run_test(num_locales):
    d = make_arrays()
    df = pd.DataFrame(d)
    akdf = {k:ak.array(v) for k, v in d.items()}
    akg = ak.GroupBy(akdf['keys'], num_locales > 0)
    tests = 0
    failures = 0
    not_impl = 0
    print(f"Doing .count()")
    tests += 1
    pdkeys, pdvals = groupby_to_arrays(df, 'keys', 'int64', 'count')
    print("Pandas:")
    print(pdkeys)
    print(pdvals)
    akkeys, akvals = akg.count()
    akkeys = akkeys.to_ndarray()
    akvals = akvals.to_ndarray()
    print("Arkouda:")
    print(akkeys)
    print(akvals)
    if not np.allclose(pdkeys, akkeys):
        print(f"Different keys")
        failures += 1
    elif not np.allclose(pdvals, akvals):
        print(f"Different values (abs diff = {np.abs(pdvals - akvals).sum()})")
        failures += 1
    for vname in ('int64', 'float64', 'bool'):
        for op in ak.GroupBy.Reductions:
            print(f"\nDoing aggregate({vname}, {op})")
            tests += 1
            do_check = True
            try:
                pdkeys, pdvals = groupby_to_arrays(df, 'keys', vname, op)
                print("Pandas:")
                print(pdkeys)
                print(pdvals)
            except Exception as E:
                print("Pandas does not implement")
                do_check = False
            try:
                akkeys, akvals = akg.aggregate(akdf[vname], op)
                akkeys = akkeys.to_ndarray()
                akvals = akvals.to_ndarray()
                print("Arkouda:")
                print(akkeys)
                print(akvals)
            except RuntimeError as E:
                print("Arkouda error: ", E)
                not_impl += 1
                do_check = False
                continue
            if not do_check:
                continue
            if op.startswith('arg'):
                pdextrema = df[vname][pdvals]
                akextrema = akdf[vname][ak.array(akvals)].to_ndarray()
                if not np.allclose(pdextrema, akextrema):
                    print(f"Different argmin/argmax: Arkouda failed to find an extremum")
                    print("pd: ", pdextrema)
                    print("ak: ", akextrema)
                    failures += 1
            else:
                if not np.allclose(pdkeys, akkeys):
                    print(f"Different keys")
                    failures += 1
                elif not np.allclose(pdvals, akvals):
                    print(f"Different values (abs diff = {np.where(np.isfinite(pdvals) & np.isfinite(akvals), np.abs(pdvals - akvals), 0).sum()})")
                    failures += 1
    print(f"\n{failures} failures in {tests} tests ({not_impl} not implemented)")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <server> <port> <per_locale_mode=0|1>")
    ak.connect(sys.argv[1], int(sys.argv[2]))
    run_test(int(sys.argv[3]))
    sys.exit()
