#!/usr/bin/env python3

import arkouda as ak
import numpy as np
import pandas as pd
SIZE = 1000
GROUPS = 11

## Need this monkey-patch until Mike implements find_segments in chapel
old_find_segments = ak.GroupBy.find_segments
def find_segments(self):
    if self.per_locale:
        global numLocales
        perm_keys = self.keys[self.permutation].to_ndarray()
        unique_keys = np.unique(perm_keys).sort()
        k2i = {k:i for i, k in enumerate(unique_keys)}      
        steps = np.hstack((np.array([True]), perm_keys[:-1]!=perm_keys[1:]))
        steps[0::len(perm_keys)//numLocales] = True
        offsets = np.arange(0, len(perm_keys))[steps]
        keyvals = perm_keys[steps]
        segments = -np.ones(len(unique_keys)*numLocales, dtype='int64')
        for o, k in zip(offsets, keyvals):
            segments[k2i[k]] = o
        return ak.array(segments), ak.array(unique_keys)
    else:
        return old_find_segments(self)

ak.GroupBy.find_segments = find_segments
## End monkey patch

def groupby_to_arrays(df, kname, vname, op):
    g = df.groupby(kname)[vname]
    agg = g.aggregate(op)
    return agg.index.values, agg.values


def run_test():
    keys = np.random.randint(0, GROUPS, SIZE)
    i = np.random.randint(0, SIZE//GROUPS, SIZE)
    f = np.random.randn(SIZE)
    b = (i % 2) == 0
    d = {'keys':keys, 'int64':i, 'float64':f, 'bool':b}
    df = pd.DataFrame(d)
    akdf = {k:ak.array(v) for k, v in d.items()}
    akg = ak.GroupBy(akdf['keys'])
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
    print("Arkouda:")
    print(akkeys)
    print(akvals)
    if not np.allclose(pdkeys, akkeys):
        print(f"Different keys")
        failures += 1
    elif not np.allclose(pdvals, akvals):
        print(f"Different values")
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
            if not np.allclose(pdkeys, akkeys):
                print(f"Different keys")
                failures += 1
                continue
            if not np.allclose(pdvals, akvals):
                print(f"Different values")
                failures += 1
    print(f"\n{failures} failures in {tests} tests ({not_impl} not implemented)")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <server> <port>")
    ak.connect(sys.argv[1], int(sys.argv[2]))
    run_test()
    sys.exit()
