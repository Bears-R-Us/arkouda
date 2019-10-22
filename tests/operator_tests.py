#!/usr/bin/env python3

import arkouda as ak
import numpy as np
import warnings
from itertools import product
SIZE = 10
warnings.simplefilter("always", UserWarning)

def run_tests():
    global pdarrays
    pdarrays = {'int64': ak.arange(1, SIZE+1, 1),
                'float64': ak.linspace(0, 1, SIZE),
                'bool': (ak.arange(0, SIZE, 1) % 2) == 0}
    global ndarrays
    ndarrays = {'int64': np.arange(1, SIZE+1, 1),
                'float64': np.linspace(0, 1, SIZE),
                'bool': (np.arange(0, SIZE, 1) % 2) == 0}
    global scalars
    #scalars = {k: v[SIZE//2] for k, v in ndarrays.items()}
    scalars = {'int64': 5,
               'float64': 3.14159,
               'bool': True}
    dtypes = pdarrays.keys()
    print("Dtypes: ", dtypes)
    print("pdarrays: ")
    for k, v in pdarrays.items():
        print(k, ": ", v)
    print("ndarrays: ")
    for k, v in ndarrays.items():
        print(k, ": ", v)
    print("scalars: ")
    for k, v in scalars.items():
        print(k, ": ", v)

    def do_op(lt, rt, ls, rs, isarkouda, oper):
        evalstr = ''
        if ls:
            evalstr += 'scalars["{}"]'.format(lt)
        else:
            evalstr += '{}["{}"]'.format(('ndarrays', 'pdarrays')[isarkouda], lt)
        evalstr += ' {} '.format(oper)
        if rs:
            evalstr += 'scalars["{}"]'.format(rt)
        else:
            evalstr += '{}["{}"]'.format(('ndarrays', 'pdarrays')[isarkouda], rt)
        #print(evalstr)
        res = eval(evalstr)
        return res

    results = {'not_implemented': [],
               'caught': [],
               'wrong_dtype': [],
               'wrong_value': [],
               'failure': []}
    tests = 0
    for ltype, rtype, op in product(dtypes, dtypes, ak.pdarray.BinOps):
        for lscalar, rscalar in ((False, False), (False, True), (True, False)):
            tests += 1
            expression = "{}({}) {} {}({})".format(ltype, ('array', 'scalar')[lscalar], op, rtype, ('array', 'scalar')[rscalar])
            try:
                npres = do_op(ltype, rtype, lscalar, rscalar, False, op)
            except TypeError: # numpy doesn't implement operation
                try:
                    akres = do_op(ltype, rtype, lscalar, rscalar, True, op)
                except RuntimeError as e:
                    results['not_implemented'].append((expression, e))
                continue
            try:
                akres = do_op(ltype, rtype, lscalar, rscalar, True, op)
            except RuntimeError as e:
                warnings.warn("Error computing {}\n{}".format(expression, str(e)))
                
                results['caught'].append((expression, e))
                continue
            try:
                akrestype = akres.dtype
            except Exception as e:
                warnings.warn("Cannot detect return dtype of ak result: {} (np result: {})".format(akres, npres))
                results['failure'].append((expression, e))
                continue
            
            if akrestype != npres.dtype:
                restypes = "{}(np) vs. {}(ak)".format(npres.dtype, akrestype)
                warnings.warn("dtype mismatch: {} = {}".format(expression, restypes))
                results['wrong_dtype'].append((expression, restypes))
                continue
            try:
                akasnp = akres.to_ndarray()
            except Exception as e:
                warnings.warn("Could not convert to ndarray: {}".format(akres))
                results['failure'].append((expression, e))
                continue
            if not np.allclose(akasnp, npres, equal_nan=True):
                res = "np: {}\nak: {}".format(npres, akasnp)
                warnings.warn("result mismatch: {} =\n{}".format(expression, res))
                results['wrong_value'].append((expression, res))
    for errtype, errs in results.items():
        print("{} {}".format(len(errs), errtype))
        for expr, msg in errs:
            print(expr)
            print(msg)
    print("{} differences from numpy in {} tests".format(sum(len(errs) for errs in results.values()) - len(results['not_implemented']), tests))


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <server_name> <port>")
    ak.connect(server=sys.argv[1], port=int(sys.argv[2]))
    run_tests()
    ak.shutdown()
    sys.exit()
