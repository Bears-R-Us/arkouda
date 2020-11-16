from typing import cast, Tuple, Union
import numpy as np # type: ignore
import builtins

__all__ = ["DTypes", "DTypeObjects", "dtype", "bool", "int64", "float64", 
           "uint8", "str_", "check_np_dtype", "translate_np_dtype", 
           "resolve_scalar_dtype"]

# supported dtypes
structDtypeCodes = {'int64': 'q',
                    'float64': 'd',
                    'bool': '?',
                    'uint8': 'B'}
NUMBER_FORMAT_STRINGS = {'bool': '{}',
                         'int64': '{:n}',
                         'float64': '{:.17f}',
                         'uint8': '{:n}'}

dtype = np.dtype
bool = np.dtype(np.bool)
int64 = np.dtype(np.int64)
float64 = np.dtype(np.float64)
uint8 = np.dtype(np.uint8)
str_ = np.dtype(np.str_)
str = np.dtype(np.str)
DTypes = frozenset(["bool", "int64", "float64", "uint8", "str"])
DTypeObjects = frozenset([bool, int64, float64, uint8, str])

def _as_dtype(dt) -> np.dtype:
    if not isinstance(dt, np.dtype):
        return np.dtype(dt)
    return dt

def check_np_dtype(dt : np.dtype) -> None:
    """
    Assert that numpy dtype dt is one of the dtypes supported
    by arkouda, otherwise raise TypeError.
    
    Raises
    ------
    TypeError
        Raised if the dtype is not in supported dtypes
    """
    
    if _as_dtype(dt).name not in DTypes:
        raise TypeError("Unsupported type: {}".format(dt))

def translate_np_dtype(dt: np.dtype) -> Tuple[builtins.str, int]:
    """
    Split numpy dtype dt into its kind and byte size, raising
    TypeError for unsupported dtypes.
    
    Raises
    ------
    TypeError
        Raised if the dtype is not in supported dtypes
    """
    # Assert that dt is one of the arkouda supported dtypes
    dt = _as_dtype(dt)
    check_np_dtype(dt)
    trans = {'i': 'int', 'f': 'float', 'b': 'bool', 
             'u': 'uint', 'U' : 'str'}
    kind = trans[dt.kind]
    return kind, dt.itemsize

def resolve_scalar_dtype(val : Union[np.bool,np.float64,int,np.int,np.uint8]) -> str: #type: ignore
    """
    Try to infer what dtype arkouda_server should treat val as.
    """
    # Python bool or np.bool
    if isinstance(val, builtins.bool) or (hasattr(val, 'dtype') \
                                and cast(np,val).dtype.kind == 'b'):
        return 'bool'
    # Python int or np.int* or np.uint*
    elif isinstance(val, int) or (hasattr(val, 'dtype') and \
                                  val.dtype.kind in 'ui'):
        return 'int64'
    # Python float or np.float*
    elif isinstance(val, float) or (hasattr(val, 'dtype') and \
                                    val.dtype.kind == 'f'):
        return 'float64'
    elif isinstance(val, builtins.str) or isinstance(val, np.str):
        return 'str'
    # Other numpy dtype
    elif hasattr(val, 'dtype'):
        return val.dtype.name
    # Other python type
    else:
        return builtins.str(type(val))
