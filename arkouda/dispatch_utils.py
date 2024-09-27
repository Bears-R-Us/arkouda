import base64
import pickle
import numba #type: ignore

print("importing arkouda.dispatch_utils")


def compile(func, typ):
    f = pickle.loads(base64.b64decode(func))
    numf = numba.cfunc(f"{typ}({typ})")(f)
    return numf.address
