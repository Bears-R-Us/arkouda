from typing import Callable, Optional, Sequence, Tuple, Union, cast

import numpy as np
from typeguard import typechecked

from arkouda.alignment import right_align
from arkouda.categorical import Categorical
from arkouda.client import generic_msg
from arkouda.dtypes import NUMBER_FORMAT_STRINGS
from arkouda.dtypes import int64 as akint64
from arkouda.dtypes import resolve_scalar_dtype
from arkouda.groupbyclass import GroupBy, broadcast
from arkouda.numeric import cumsum
from arkouda.pdarrayclass import create_pdarray, pdarray
from arkouda.pdarraycreation import arange, array, ones, zeros
from arkouda.pdarraysetops import concatenate, in1d
from arkouda.strings import Strings

__all__ = ["join_on_eq_with_dt", "gen_ranges", "compute_join_size"]

predicates = {"true_dt": 0, "abs_dt": 1, "pos_dt": 2}


@typechecked
def join_on_eq_with_dt(
    a1: pdarray,
    a2: pdarray,
    t1: pdarray,
    t2: pdarray,
    dt: Union[int, np.int64],
    pred: str,
    result_limit: Union[int, np.int64] = 1000,
) -> Tuple[pdarray, pdarray]:
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
    t2 : pdarray
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
        raise ValueError(f"pred must be one of {predicates.keys()}")

    if result_limit < 0:
        raise ValueError("the result_limit must 0 or greater")

    # format numbers for request message
    dttype = resolve_scalar_dtype(dt)
    dtstr = NUMBER_FORMAT_STRINGS[dttype].format(dt)
    predtype = resolve_scalar_dtype(predicates[pred])
    predstr = NUMBER_FORMAT_STRINGS[predtype].format(predicates[pred])
    result_limittype = resolve_scalar_dtype(result_limit)
    result_limitstr = NUMBER_FORMAT_STRINGS[result_limittype].format(result_limit)
    # groupby on a2
    g2 = GroupBy(a2)
    # pass result into server joinEqWithDT operation
    repMsg = generic_msg(
        cmd="joinEqWithDT",
        args={
            "a1": a1,
            "g2seg": cast(pdarray, g2.segments),
            "g2keys": cast(pdarray, g2.unique_keys),
            "g2perm": g2.permutation,
            "t1": t1,
            "t2": t2,
            "dt": dtstr,
            "pred": predstr,
            "resLimit": result_limitstr,
        },
    )
    # create pdarrays for results
    resIAttr, resJAttr = cast(str, repMsg).split("+")
    resI = create_pdarray(resIAttr)
    resJ = create_pdarray(resJAttr)
    return resI, resJ


def gen_ranges(starts, ends, stride=1, return_lengths=False):
    """
    Generate a segmented array of variable-length, contiguous ranges between pairs of
    start- and end-points.

    Parameters
    ----------
    starts : pdarray, int64
        The start value of each range
    ends : pdarray, int64
        The end value (exclusive) of each range
    stride: int
        Difference between successive elements of each range
    return_lengths: bool, optional
        Whether or not to return the lengths of each segment. Default False.

    Returns
    -------
    segments : pdarray, int64
        The starting index of each range in the resulting array
    ranges : pdarray, int64
        The actual ranges, flattened into a single array
    lengths : pdarray, int64
        The lengths of each segment. Only returned if return_lengths=True.
    """
    if starts.size != ends.size:
        raise ValueError("starts and ends must be same length")
    if starts.size == 0:
        return zeros(0, dtype=akint64), zeros(0, dtype=akint64)
    lengths = (ends - starts) // stride
    if not (lengths >= 0).all():
        raise ValueError("all ends must be greater than or equal to starts")
    non_empty = lengths != 0
    segs = cumsum(lengths) - lengths
    totlen = lengths.sum()
    slices = ones(totlen, dtype=akint64)
    non_empty_starts = starts[non_empty]
    non_empty_lengths = lengths[non_empty]
    diffs = concatenate(
        (
            array([non_empty_starts[0]]),
            non_empty_starts[1:] - non_empty_starts[:-1] - (non_empty_lengths[:-1] - 1) * stride,
        )
    )
    slices[segs[non_empty]] = diffs

    sums = cumsum(slices)
    if return_lengths:
        return segs, sums, lengths
    else:
        return segs, sums


@typechecked
def compute_join_size(a: pdarray, b: pdarray) -> Tuple[int, int]:
    """
    Compute the internal size of a hypothetical join between a and b. Returns
    both the number of elements and number of bytes required for the join.
    """
    bya = GroupBy(a)
    ua, asize = bya.size()
    byb = GroupBy(b)
    ub, bsize = byb.size()
    afact = asize[in1d(ua, ub)]
    bfact = bsize[in1d(ub, ua)]
    nelem = (afact * bfact).sum()
    nbytes = 3 * 8 * nelem
    return nelem, nbytes


@typechecked
def inner_join(
    left: Union[pdarray, Strings, Categorical, Sequence[Union[pdarray, Strings]]],
    right: Union[pdarray, Strings, Categorical, Sequence[Union[pdarray, Strings]]],
    wherefunc: Optional[Callable] = None,
    whereargs: Optional[
        Tuple[
            Union[pdarray, Strings, Categorical, Sequence[Union[pdarray, Strings]]],
            Union[pdarray, Strings, Categorical, Sequence[Union[pdarray, Strings]]],
        ]
    ] = None,
) -> Tuple[pdarray, pdarray]:
    """Perform inner join on values in <left> and <right>,
    using conditions defined by <wherefunc> evaluated on
    <whereargs>, returning indices of left-right pairs.

    Parameters
    ----------
    left : pdarray(int64), Strings, Categorical, or Sequence of pdarray
        The left values to join
    right : pdarray(int64), Strings, Categorical, or Sequence of pdarray
        The right values to join
    wherefunc : function, optional
        Function that takes two pdarray arguments and returns
        a pdarray(bool) used to filter the join. Results for
        which wherefunc is False will be dropped.
    whereargs : 2-tuple of pdarray, Strings, Categorical, or Sequence of pdarray, optional
        The two arguments for wherefunc

    Returns
    -------
    leftInds : pdarray(int64)
        The left indices of pairs that meet the join condition
    rightInds : pdarray(int64)
        The right indices of pairs that meet the join condition

    Notes
    -----
    The return values satisfy the following assertions

    `assert (left[leftInds] == right[rightInds]).all()`
    `assert wherefunc(whereargs[0][leftInds], whereargs[1][rightInds]).all()`

    """
    from inspect import signature

    is_sequence = isinstance(left, Sequence) and isinstance(right, Sequence)

    # Reduce processing to codes to prevent groupby on entire Categorical
    if isinstance(left, Categorical) and isinstance(right, Categorical):
        l, r = Categorical.standardize_categories([left, right])
        left, right = l.codes, r.codes

    if is_sequence:
        if len(left) != len(right):
            raise ValueError("Left must have same num arrays as right")
        left_size, right_size = left[0].size, right[0].size
        if not all(lf.size == left_size for lf in left) or not all(
            rt.size == right_size for rt in right
        ):
            raise ValueError("Multi-array arguments must have equal-length arrays")
    else:
        left_size, right_size = left.size, right.size  # type: ignore

    sample = np.min((left_size, right_size, 5))  # type: ignore
    if wherefunc is not None:
        if len(signature(wherefunc).parameters) != 2:
            raise ValueError("wherefunc must be a function that accepts exactly two arguments")
        if whereargs is None or len(whereargs) != 2:
            raise ValueError("whereargs must be a 2-tuple with left and right arg arrays")
        if is_sequence:
            if len(whereargs[0]) != len(whereargs[1]):
                raise ValueError("Left must have same num arrays as right")
            first_wa_size, second_wa_size = whereargs[0][0].size, whereargs[1][0].size
            if not all(wa.size == first_wa_size for wa in whereargs[0]) or not all(
                wa.size == second_wa_size for wa in whereargs[1]
            ):
                raise ValueError("Multi-array arguments must have equal-length arrays")
        else:
            first_wa_size, second_wa_size = whereargs[0].size, whereargs[1].size  # type: ignore
        if first_wa_size != left_size:
            raise ValueError("Left whereargs must be same size as left join values")
        if second_wa_size != right_size:
            raise ValueError("Right whereargs must be same size as right join values")
        try:
            _ = wherefunc(whereargs[0][:sample], whereargs[1][:sample])
        except Exception as e:
            raise ValueError("Error evaluating wherefunc") from e

    # Need dense 0-up right index, to filter out left not in right
    keep, (denseLeft, denseRight) = right_align(left, right)
    if keep.sum() == 0:
        # Intersection is empty
        return zeros(0, dtype=akint64), zeros(0, dtype=akint64)
    keep = arange(keep.size)[keep]
    # GroupBy right
    byRight = GroupBy(denseRight)
    # Get segment boundaries (starts, ends) of right for each left item
    rightSegs = concatenate((byRight.segments, array([denseRight.size])))
    starts = rightSegs[denseLeft]
    ends = rightSegs[denseLeft + 1]
    # gen_ranges for gather of right items
    fullSegs, ranges = gen_ranges(starts, ends)
    # Evaluate where clause
    if wherefunc is None:
        filtRanges = ranges
        filtSegs = fullSegs
        keep12 = keep
    else:
        if whereargs is not None:
            if not is_sequence:
                # Gather right whereargs
                rightWhere = whereargs[1][byRight.permutation][ranges]
                # Expand left whereargs
                keep_where = whereargs[0][keep]
                keep_where = keep_where.codes if isinstance(keep_where, Categorical) else keep_where
                leftWhere = broadcast(fullSegs, keep_where, ranges.size)
            else:
                # Gather right whereargs
                rightWhere = [wa[byRight.permutation][ranges] for wa in whereargs[1]]
                # Expand left whereargs
                keep_where = [wa[keep] for wa in whereargs[0]]
                leftWhere = [broadcast(fullSegs, kw, ranges.size) for kw in keep_where]
            # Evaluate wherefunc and filter ranges, recompute segments
            whereSatisfied = wherefunc(leftWhere, rightWhere)
            filtRanges = ranges[whereSatisfied]
            scan = cumsum(whereSatisfied) - whereSatisfied
            filtSegsWithZeros = scan[fullSegs]
            filtSegSizes = concatenate(
                (
                    filtSegsWithZeros[1:] - filtSegsWithZeros[:-1],
                    array([whereSatisfied.sum() - filtSegsWithZeros[-1]]),
                )
            )
            keep2 = filtSegSizes > 0
            filtSegs = filtSegsWithZeros[keep2]
            keep12 = keep[keep2]
    # Gather right inds and expand left inds
    rightInds = byRight.permutation[filtRanges]
    leftInds = broadcast(filtSegs, arange(left_size)[keep12], filtRanges.size)
    return leftInds, rightInds
