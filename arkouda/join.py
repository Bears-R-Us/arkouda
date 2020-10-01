from typing import Tuple
from arkouda.client import generic_msg, verbose
from arkouda.dtypes import *
from arkouda.dtypes import structDtypeCodes, NUMBER_FORMAT_STRINGS
from arkouda.dtypes import dtype as akdtype
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.groupbyclass import GroupBy

global verbose

__all__ = ["join_on_eq_with_dt"]

predicates = {"true_dt":0, "abs_dt":1, "pos_dt":2}

def join_on_eq_with_dt(a1 : pdarray, a2 : pdarray, t1 : pdarray, 
                       t2 : pdarray, dt : int, pred : str, 
                       result_limit : int=1000) -> Tuple[pdarray,pdarray]:
    """
    Performs an inner-join on equality between two integer arrays where 
    the time-window predicate is also true

    Parameters
    ----------
    a1 : pdarray, int64
        pdarray to be joined
    a2 : pdarray, int64
        pdarray to be joined
    t1 : pdarray
        timestamps in millis corresponding to the a1 pdarray
    t2 : pdarray, 
        timestamps in millis corresponding to the a2 pdarray
    dt : int
        time delta
    pred : str
        time window predicate
    result_limit : int
        size limit for returned result    

    Returns
    -------
    result_array_one : pdarray, int64
        a1 indices where a1 == a2
    result_array_one : pdarray, int64
        a2 indices where a2 == a1
        
    Raises
    ------
    TypeError
        Raised if a1, a2, t1, or t2 is not a pdarray, or if dt or 
        result_limit is not an int
    ValueError
        if a1, a2, t1, or t2 dtype is not int64, pred is not 
        'true_dt', 'abs_dt', or 'pos_dt', or result_limit is < 0    
    """
    if not isinstance(a1, pdarray):
        raise TypeError("a1 must be pdarray")
    else:
        if not (a1.dtype == int64):
            raise ValueError("a1 must be int64 dtype")
        
    if not isinstance(a2, pdarray):
        raise TypeError("a2 must be pdarray")
    else:
        if not (a2.dtype == int64):
            raise ValueError("a2 must be int64 dtype")
        
    if not isinstance(t1, pdarray):
        raise TypeError("t1 must be pdarray")
    else:
        if not (t1.dtype == int64):
            raise ValueError("t1 must be int64 dtype")
        
    if not isinstance(t2, pdarray):
        raise TypeError("t2 must be pdarray")
    else:
        if not (t2.dtype == int64):
            raise ValueError("t2 must be int64 dtype")
        
    if not isinstance(dt, int):
        raise TypeError("dt must be an an int")
    
    if not (pred in predicates.keys()):
        raise ValueError("pred must be one of ", predicates.keys())
    
    if not isinstance(result_limit, int):
        raise TypeError("result_limit must be a scalar")
    else:
        if result_limit < 0:
            raise ValueError('the result_limit must 0 or greater')

    # format numbers for request message
    dttype = resolve_scalar_dtype(dt)
    dtstr = NUMBER_FORMAT_STRINGS[dttype].format(dt)
    predtype = resolve_scalar_dtype(predicates[pred])
    predstr = NUMBER_FORMAT_STRINGS[predtype].format(predicates[pred])
    result_limittype = resolve_scalar_dtype(result_limit)
    result_limitstr = NUMBER_FORMAT_STRINGS[result_limittype].\
                                 format(result_limit)
    # groupby on a2
    g2 = GroupBy(a2)
    # pass result into server joinEqWithDT operation
    repMsg = generic_msg("joinEqWithDT {} {} {} {} {} {} {} {} {}".\
                         format(a1.name,
                                g2.segments.name,
                                g2.unique_keys.name,
                                g2.permutation.name,
                                t1.name,
                                t2.name,
                                dtstr, predstr, result_limitstr))
    # create pdarrays for results
    resIAttr, resJAttr = repMsg.split("+")
    resI = create_pdarray(resIAttr)
    resJ = create_pdarray(resJAttr)
    return (resI, resJ)
