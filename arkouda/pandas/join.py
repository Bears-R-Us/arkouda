from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np

from typeguard import typechecked

from arkouda.numpy.dtypes import NUMBER_FORMAT_STRINGS, resolve_scalar_dtype
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.pdarraysetops import concatenate, in1d
from arkouda.pandas.categorical import Categorical
from arkouda.pandas.groupbyclass import GroupBy, broadcast


if TYPE_CHECKING:
    from arkouda.numpy.strings import Strings
else:
    Strings = TypeVar("Strings")

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
    Inner-join on equality between two integer arrays where the time-window predicate is also true.

    Parameters
    ----------
    a1 : pdarray
        Values to join (must be int64 dtype).
    a2 : pdarray
        Values to join (must be int64 dtype).
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
    Tuple[pdarray, pdarray]
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
    from arkouda.client import generic_msg

    if not (a1.dtype == akint64):
        raise ValueError("a1 must be int64 dtype")

    if not (a2.dtype == akint64):
        raise ValueError("a2 must be int64 dtype")

    if not (t1.dtype == akint64):
        raise ValueError("t1 must be int64 dtype")

    if not (t2.dtype == akint64):
        raise ValueError("t2 must be int64 dtype")

    if pred not in predicates.keys():
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
    rep_msg = generic_msg(
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
    res_i_attr, res_j_attr = cast(str, rep_msg).split("+")
    res_i = create_pdarray(res_i_attr)
    res_j = create_pdarray(res_j_attr)
    return res_i, res_j


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
    pdarray|int64, pdarray|int64, pdarray|int64
        segments : pdarray, int64
            The starting index of each range in the resulting array
        ranges : pdarray, int64
            The actual ranges, flattened into a single array
        lengths : pdarray, int64
            The lengths of each segment. Only returned if return_lengths=True.

    """
    from arkouda.numpy import cumsum
    from arkouda.numpy.pdarraycreation import array, ones, zeros

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
        Tuple[Union[pdarray, Strings, Categorical, Sequence[Union[pdarray, Strings]]], ...]
    ] = None,
) -> Tuple[pdarray, pdarray]:
    """
    Perform inner join on values in <left> and <right>,
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
    Tuple[pdarray, pdarray]
        left_inds : pdarray(int64)
            The left indices of pairs that meet the join condition
        right_inds : pdarray(int64)
            The right indices of pairs that meet the join condition

    Notes
    -----
    The return values satisfy the following assertions

    `assert (left[left_inds] == right[right_inds]).all()`
    `assert wherefunc(whereargs[0][left_inds], whereargs[1][right_inds]).all()`

    """
    from inspect import signature

    from arkouda.numpy import cumsum
    from arkouda.numpy.alignment import right_align
    from arkouda.numpy.pdarraycreation import arange, array, zeros

    is_sequence = isinstance(left, Sequence) and isinstance(right, Sequence)

    # Reduce processing to codes to prevent groupby on entire Categorical
    if isinstance(left, Categorical) and isinstance(right, Categorical):
        lft, r = Categorical.standardize_categories([left, right])
        left, right = lft.codes, r.codes

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
    keep, (dense_left, dense_right) = right_align(left, right)
    if keep.sum() == 0:
        # Intersection is empty
        return zeros(0, dtype=akint64), zeros(0, dtype=akint64)
    keep = arange(keep.size)[keep]
    # GroupBy right
    by_right = GroupBy(dense_right)
    # Get segment boundaries (starts, ends) of right for each left item
    right_segs = concatenate((by_right.segments, array([dense_right.size])))
    starts = right_segs[dense_left]
    ends = right_segs[dense_left + 1]
    # gen_ranges for gather of right items
    full_segs, ranges = gen_ranges(starts, ends)
    # Evaluate where clause
    if wherefunc is None:
        filt_ranges = ranges
        filt_segs = full_segs
        keep12 = keep
    else:
        if whereargs is not None:
            right = whereargs[1]

            left = whereargs[0]
            if not isinstance(right, Sequence) and not isinstance(left, Sequence):
                # Gather right whereargs
                right_where = right[by_right.permutation][ranges]
                # Expand left whereargs
                keep_where = left[keep]
                keep_where = keep_where.codes if isinstance(keep_where, Categorical) else keep_where
                left_where = broadcast(full_segs, keep_where, ranges.size)
            else:
                # Gather right whereargs
                right_where = [wa[by_right.permutation][ranges] for wa in whereargs[1]]
                # Expand left whereargs
                keep_where = [wa[keep] for wa in whereargs[0]]
                left_where = [broadcast(full_segs, kw, ranges.size) for kw in keep_where]
            # Evaluate wherefunc and filter ranges, recompute segments
            where_satisfied = wherefunc(left_where, right_where)
            filt_ranges = ranges[where_satisfied]
            scan = cumsum(where_satisfied) - where_satisfied
            filt_segs_with_zeros = scan[full_segs]
            filt_seg_sizes = concatenate(
                (
                    filt_segs_with_zeros[1:] - filt_segs_with_zeros[:-1],
                    array([where_satisfied.sum() - filt_segs_with_zeros[-1]]),
                )
            )
            keep2 = filt_seg_sizes > 0
            filt_segs = filt_segs_with_zeros[keep2]
            keep12 = keep[keep2]
    # Gather right inds and expand left inds
    right_inds = by_right.permutation[filt_ranges]
    left_inds = broadcast(filt_segs, arange(left_size)[keep12], filt_ranges.size)
    return left_inds, right_inds
