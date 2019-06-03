#!/usr/bin/env python3

import arkouda as ak
import numpy as np
import pandas as pd
SIZE = 100
GROUPS = 65

# ## Need this monkey-patch until Mike implements find_segments in chapel
# old_find_segments = ak.GroupBy.find_segments
# def find_segments(self):
#     if self.per_locale:
#         global numLocales
#         perm_keys = self.keys[self.permutation].to_ndarray()
#         unique_keys = np.unique(perm_keys)
#         unique_keys.sort()
#         perm_keys = perm_keys.reshape((numLocales, -1))
#         k2i = {k:i for i, k in enumerate(unique_keys)}
#         segments = -np.ones(len(unique_keys)*numLocales, dtype='int64').reshape((numLocales, -1))
#         for loc in range(numLocales):
#             steps = np.hstack((np.array([True]), perm_keys[loc, :-1]!=perm_keys[loc, 1:]))
#             # steps[0::len(perm_keys)//numLocales] = True
#             start = loc * perm_keys.shape[1]
#             offsets = np.arange(start, start + perm_keys.shape[1])[steps]
#             keyvals = perm_keys[loc, :][steps]
#             for o, k in zip(offsets, keyvals):
#                 segments[loc, k2i[k]] = o
#         segments = segments.flatten()
#         last = self.keys.size
#         for i in range(len(segments)-1, -1, -1):
#             if segments[i] == -1:
#                 segments[i] = last
#             else:
#                 last = segments[i]
#         return ak.array(segments), ak.array(unique_keys)
#     else:
#         return old_find_segments(self)

# ak.GroupBy.find_segments = find_segments
# ## End monkey patch

def groupby_to_arrays(df, kname, vname, op):
    # if op == 'argmin':
    #     keys = np.unique(df[kname].values)
    #     argmins = []
    #     for k in keys:
    #         argmins.append(np.where(df[kname] == k, df[vname], np.inf).argmin())
    #     return keys, np.array(argmins)
    # elif op == 'argmax':
    #     keys = np.unique(df[kname].values)
    #     argmaxes = []
    #     for k in keys:
    #         argmaxes.append(np.where(df[kname] == k, df[vname], -np.inf).argmax())
    #     return keys, np.argmax(argmaxes)
    # else:
    g = df.groupby(kname)[vname]
    agg = g.aggregate(op.replace('arg', 'idx'))
    return agg.index.values, agg.values


def run_test(num_locales):
    global numLocales
    numLocales = num_locales
    keys = np.random.randint(0, GROUPS, SIZE)
    i = np.random.randint(0, SIZE//GROUPS, SIZE)
    f = np.random.randn(SIZE)
    b = (i % 2) == 0
    d = {'keys':keys, 'int64':i, 'float64':f, 'bool':b}
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
                    print(f"Different values")
                    failures += 1
    print(f"\n{failures} failures in {tests} tests ({not_impl} not implemented)")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <server> <port> <per_locale_mode=0|1>")
    ak.connect(sys.argv[1], int(sys.argv[2]))
    run_test(int(sys.argv[3]))
    sys.exit()
