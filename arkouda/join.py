import numpy as np
import struct

from arkouda.client import generic_msg, verbose
from arkouda.dtypes import *
from arkouda.dtypes import structDtypeCodes, NUMBER_FORMAT_STRINGS
from arkouda.dtypes import dtype as akdtype
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.pdarraycreation import zeros, array
from arkouda.groupbyclass import GroupBy

global verbose

__all__ = ["join_on_eq_with_dt"]
global verbose

def join_on_eq_with_dt(a1, a2, t1, t2, dt, pred, result_limit=1000):
    if not isinstance(a1, pdarray):
        raise ValueError("a1 must be pdarray")
    else:
        if not (a1.dtype == int64):
            raise ValueError("a1 must be int64 dtype")
        
    if not isinstance(a2, pdarray):
        raise ValueError("a2 must be pdarray")
    else:
        if not (a2.dtype == int64):
            raise ValueError("a2 must be int64 dtype")
        
    if not isinstance(t1, pdarray):
        raise ValueError("t1 must be pdarray")
    else:
        if not (t1.dtype == int64):
            raise ValueError("t1 must be int64 dtype")
        
    if not isinstance(t2, pdarray):
        raise ValueError("t2 must be pdarray")
    else:
        if not (t2.dtype == int64):
            raise ValueError("t2 must be int64 dtype")
        
    if not isinstance(dt, int):
        raise ValueError("dt must be an int")
    
    if not isinstance(pred, int):
        raise ValueError("pred must be an int")
    else:
        if (pred < 0) or (pred > 2):
            raise ValueError("pred must be 0,1,2")
    
    if not isinstance(result_limit, int):
        raise ValueError("result_limit must be an int")

    # groupby on a2
    g2 = GroupBy(a2)
    # pass result into server joinEqWithDT operation
    repMsg = generic_msg("joinEqWithDT {} {} {} {} {} {} {} {} {}".format(a1.name,
                                                                          g2.segments.name,
                                                                          g2.unique_keys.name,
                                                                          g2.permutation.name,
                                                                          t1.name,
                                                                          t2.name,
                                                                          dt, pred, result_limit))
    # create pdarrays for results
    resIAttr, resJAttr = repMsg.split("+")
    resI = create_pdarray(resIAttr)
    resJ = create_pdarray(resJAttr)
    return (resI, resJ)



    

    
