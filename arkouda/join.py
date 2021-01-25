from typing import cast, Tuple, Union
from typeguard import typechecked
import numpy as np # type: ignore
from arkouda.client import generic_msg
from arkouda.dtypes import int64 as akint64
from arkouda.dtypes import resolve_scalar_dtype, NUMBER_FORMAT_STRINGS
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.groupbyclass import GroupBy

__all__ = ["join_on_eq_with_dt"]

predicates = {"true_dt":0, "abs_dt":1, "pos_dt":2}

@typechecked
def join_on_eq_with_dt(a1 : pdarray, a2 : pdarray, t1 : pdarray, 
                       t2 : pdarray, dt : Union[int,np.int64], pred : str, 
                       result_limit : Union[int,np.int64]=1000) -> Tuple[pdarray,pdarray]:
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
    dt : Union[int,np.int64]
        time delta
    pred : str
        time window predicate
    result_limit : Union[int,np.int64]
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
    if not (a1.dtype == akint64):
        raise ValueError("a1 must be int64 dtype")

    if not (a2.dtype == akint64):
        raise ValueError("a2 must be int64 dtype")

    if not (t1.dtype == akint64):
        raise ValueError("t1 must be int64 dtype")
        
    if not (t2.dtype == akint64):
        raise ValueError("t2 must be int64 dtype")
    
    if not (pred in predicates.keys()):
        raise ValueError("pred must be one of ", predicates.keys())
    
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
    repMsg = generic_msg(cmd="joinEqWithDT", args="{} {} {} {} {} {} {} {} {}".\
                         format(a1.name,
                                cast(pdarray, g2.segments).name,  # type: ignore
                                cast(pdarray, g2.unique_keys).name,  # type: ignore
                                g2.permutation.name,
                                t1.name,
                                t2.name,
                                dtstr, predstr, result_limitstr))
    # create pdarrays for results
    resIAttr, resJAttr = cast(str,repMsg).split("+")
    resI = create_pdarray(resIAttr)
    resJ = create_pdarray(resJAttr)
    return (resI, resJ)
