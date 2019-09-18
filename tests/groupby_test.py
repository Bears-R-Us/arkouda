#!/usr/bin/env python3

import arkouda as ak
import numpy as np
import pandas as pd
SIZE = 10000
GROUPS = 64

def groupby_to_arrays(df, kname, vname, op, levels):
    g = df.groupby(kname)[vname]
    agg = g.aggregate(op.replace('arg', 'idx'))
    if levels==1:
        keys = agg.index.values
    else:
        keys = tuple(zip(*(agg.index.values)))
    return keys, agg.values

def make_arrays():
    keys = np.random.randint(0, GROUPS, SIZE)
    keys2 = np.random.randint(0, GROUPS, SIZE)
    i = np.random.randint(0, SIZE//GROUPS, SIZE)
    f = np.random.randn(SIZE)
    b = (i % 2) == 0
    d = {'keys':keys, 'keys2':keys2, 'int64':i, 'float64':f, 'bool':b}
    return d

def compare_keys(pdkeys, akkeys, levels, pdvals, akvals):
    if levels == 1:
        akkeys = akkeys.to_ndarray()
        if not np.allclose(pdkeys, akkeys):
            print("Different keys")
            return 1
    else:
        for l in range(levels):
            if not np.allclose(pdkeys[l], akkeys[l].to_ndarray()):
                print("Different keys")
                return 1
    if not np.allclose(pdvals, akvals):
        print(f"Different values (abs diff = {np.abs(pdvals - akvals).sum()})")
        return 1
    return 0

def run_test(levels):
    d = make_arrays()
    df = pd.DataFrame(d)
    akdf = {k:ak.array(v) for k, v in d.items()}
    if levels == 1:
        akg = ak.GroupBy(akdf['keys'])
        keyname = 'keys'
    elif levels == 2:
        akg = ak.GroupBy([akdf['keys'], akdf['keys2']])
        keyname = ['keys', 'keys2']
    tests = 0
    failures = 0
    not_impl = 0
    print(f"Doing .count()")
    tests += 1
    pdkeys, pdvals = groupby_to_arrays(df, keyname, 'int64', 'count', levels)
    # print("Pandas:")
    # print(pdkeys)
    # print(pdvals)
    akkeys, akvals = akg.count()
    # akkeys = akkeys.to_ndarray()
    akvals = akvals.to_ndarray()
    # print("Arkouda:")
    # print(akkeys)
    # print(akvals)
    # if not np.allclose(pdkeys, akkeys):
    #     print(f"Different keys")
    #     failures += 1
    failures += compare_keys(pdkeys, akkeys, levels, pdvals, akvals)
    # elif not np.allclose(pdvals, akvals):
    #     print(f"Different values (abs diff = {np.abs(pdvals - akvals).sum()})")
    #     failures += 1
    for vname in ('int64', 'float64', 'bool'):
        for op in ak.GroupBy.Reductions:
            print(f"\nDoing aggregate({vname}, {op})")
            tests += 1
            do_check = True
            try:
                pdkeys, pdvals = groupby_to_arrays(df, keyname, vname, op, levels)
                # print("Pandas:")
                # print(pdkeys)
                # print(pdvals)
            except Exception as E:
                print("Pandas does not implement")
                do_check = False
            try:
                akkeys, akvals = akg.aggregate(akdf[vname], op)
                # akkeys = akkeys.to_ndarray()
                akvals = akvals.to_ndarray()
                # print("Arkouda:")
                # print(akkeys)
                # print(akvals)
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
                # if not np.allclose(pdkeys, akkeys):
                #     print(f"Different keys")
                #     failures += 1
                failures += compare_keys(pdkeys, akkeys, levels, pdvals, akvals)
                # elif not np.allclose(pdvals, akvals):
                #     print(f"Different values (abs diff = {np.where(np.isfinite(pdvals) & np.isfinite(akvals), np.abs(pdvals - akvals), 0).sum()})")
                #     failures += 1
    print(f"\n{failures} failures in {tests} tests ({not_impl} not implemented)")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <server> <port> <levels=1|2>")
        sys.exit()
    levels = int(sys.argv[3])
    if levels not in (1, 2):
        print(f"Levels must be 1 or 2")
        sys.exit()
    ak.connect(sys.argv[1], int(sys.argv[2]))
    run_test(levels)
    sys.exit()
