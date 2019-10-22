import numpy as np
import builtins

__all__ = ["DTypes", "dtype", "bool", "int64", "float64", "check_np_dtype", "translate_np_dtype", "resolve_scalar_dtype"]

# supported dtypes
structDtypeCodes = {'int64': 'q',
                    'float64': 'd',
                    'bool': '?'}
DTypes = frozenset(structDtypeCodes.keys())
NUMBER_FORMAT_STRINGS = {'bool': '{}',
                         'int64': '{:n}',
                         'float64': '{:.17f}'}

dtype = np.dtype
bool = np.bool
int64 = np.int64
float64 = np.float64

def check_np_dtype(dt):
    """
    Assert that numpy dtype dt is one of the dtypes supported by arkouda, 
    otherwise raise TypeError.
    """
    if dt.name not in DTypes:
        raise TypeError("Unsupported type: {}".format(dt))

def translate_np_dtype(dt):
    """
    Split numpy dtype dt into its kind and byte size, raising TypeError 
    for unsupported dtypes.
    """
    # Assert that dt is one of the arkouda supported dtypes
    check_np_dtype(dt)
    trans = {'i': 'int', 'f': 'float', 'b': 'bool'}
    kind = trans[dt.kind]
    return kind, dt.itemsize

def resolve_scalar_dtype(val):
    """
    Try to infer what dtype arkouda_server should treat val as.
    """
    # Python bool or np.bool
    if isinstance(val, builtins.bool) or (hasattr(val, 'dtype') and val.dtype.kind == 'b'):
        return 'bool'
    # Python int or np.int* or np.uint*
    elif isinstance(val, int) or (hasattr(val, 'dtype') and val.dtype.kind in 'ui'):
        return 'int64'
    # Python float or np.float*
    elif isinstance(val, float) or (hasattr(val, 'dtype') and val.dtype.kind == 'f'):
        return 'float64'
    # Other numpy dtype
    elif hasattr(val, 'dtype'):
        return val.dtype.name
    # Other python type
    else:
        return str(type(val))
