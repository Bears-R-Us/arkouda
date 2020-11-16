import arkouda as ak
from akutil.alignment import right_align

def gen_ranges(starts, ends):
    """ Generate a segmented array of variable-length, contiguous 
    ranges between pairs of start- and end-points.

    Parameters
    ----------
    starts : pdarray, int64
        The start value of each range
    ends : pdarray, int64
        The end value (exclusive) of each range

    Returns
    -------
    segments : pdarray, int64
        The starting index of each range in the resulting array
    ranges : pdarray, int64
        The actual ranges, flattened into a single array
    """
    if starts.size != ends.size:
        raise ValueError("starts and ends must be same size")
    if not ((ends - starts) > 0).all():
        raise ValueError("all ends must be greater than starts")
    lengths = ends - starts
    segs = ak.cumsum(lengths) - lengths
    totlen = lengths.sum()
    slices = ak.ones(totlen, dtype=ak.int64)
    diffs = ak.concatenate((ak.array([starts[0]]), 
                            starts[1:] - starts[:-1] - lengths[:-1] + 1))
    slices[segs] = diffs
    return segs, ak.cumsum(slices)

def expand(vals, segs, size):
    """ Broadcast per-segment values to a segmented array. Equivalent 
    to ak.GroupBy.broadcast(vals) but accepts explicit segments and 
    size arguments.

    Parameters
    ----------
    vals : pdarray
        Values (one per segment) to broadcast over segments
    segs : pdarray
        Start indices of segments
    size : int
        Total size of result array

    Returns
    -------
    pdarray
        Values broadcasted out to segments
    """
    if vals.size != segs.size:
        raise ValueError("vals and segs must have same size")
    if vals.size == 0:
        return ak.array([])
    if size < segs.size or size <= segs.max():
        raise ValueError("Total size cannot be less than max segment")
    if segs[0] != 0 or not (segs[:-1] < segs[1:]).all():
        raise ValueError("segs must start at zero and be monotonically increasing")
    temp = ak.zeros(size, dtype=vals.dtype)
    diffs = ak.concatenate((ak.array([vals[0]]), vals[1:] - vals[:-1]))
    temp[segs] = diffs
    return ak.cumsum(temp)

def inner_join(left, right, wherefunc=None, whereargs=None):
    '''Perform inner join on values in <left> and <right>, 
    using conditions defined by <wherefunc> evaluated on 
    <whereargs>, returning indices of left-right pairs. 

    Parameters
    ----------
    left : pdarray(int64)
        The left values to join
    right : pdarray(int64)
        The right values to join
    wherefunc : function, optional
        Function that takes two pdarray arguments and returns 
        a pdarray(bool) used to filter the join. Results for
        which wherefunc is False will be dropped.
    whereargs : 2-tuple of pdarray
        The two pdarray arguments to wherefunc
        
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
        
    '''
    from inspect import signature
    sample = min((left.size, right.size, 5))
    if wherefunc is not None:
        if len(signature(wherefunc).parameters) != 2:
            raise ValueError("wherefunc must be a function that accepts exactly two arguments")
        if whereargs is None or len(whereargs) != 2:
            raise ValueError("whereargs must be a 2-tuple with left and right arg arrays")
        if whereargs[0].size != left.size:
            raise ValueError("Left whereargs must be same size as left join values")
        if whereargs[1].size != right.size:
            raise ValueError("Right whereargs must be same size as right join values")
        try:
            _ = wherefunc(whereargs[0][:sample], whereargs[1][:sample])
        except Exception as e:
            raise ValueError("Error evaluating wherefunc") from e

    # Need dense 0-up right index, to filter out left not in right
    keep, (denseLeft, denseRight) = right_align(left, right)
    keep = ak.arange(keep.size)[keep]
    # GroupBy right
    byRight = ak.GroupBy(denseRight)
    # Get segment boundaries (starts, ends) of right for each left item
    rightSegs = ak.concatenate((byRight.segments, ak.array([denseRight.size])))
    starts = rightSegs[denseLeft]
    ends = rightSegs[denseLeft+1]
    fullSize = (ends - starts).sum()
    # print(f"{left.size+right.size:,} input rows --> {fullSize:,} joins ({fullSize/(left.size+right.size):.1f} x) ")
    # gen_ranges for gather of right items
    fullSegs, ranges = gen_ranges(starts, ends)
    # Evaluate where clause
    if wherefunc is None:
        filtRanges = ranges
        filtSegs = fullSegs
        keep12 = keep
    else:
        # Gather right whereargs
        rightWhere = whereargs[1][byRight.permutation][ranges]
        # Expand left whereargs
        leftWhere = expand(whereargs[0][keep], fullSegs, ranges.size)
        # Evaluate wherefunc and filter ranges, recompute segments
        whereSatisfied = wherefunc(leftWhere, rightWhere)
        filtRanges = ranges[whereSatisfied]
        scan = ak.cumsum(whereSatisfied) - whereSatisfied
        filtSegsWithZeros = scan[fullSegs]    
        filtSegSizes = ak.concatenate((filtSegsWithZeros[1:] - filtSegsWithZeros[:-1], 
                                       ak.array([whereSatisfied.sum() - filtSegsWithZeros[-1]])))
        keep2 = (filtSegSizes > 0)
        filtSegs = filtSegsWithZeros[keep2]
        keep12 = keep[keep2]
    # Gather right inds and expand left inds
    rightInds = byRight.permutation[filtRanges]
    leftInds = expand(ak.arange(left.size)[keep12], filtSegs, filtRanges.size)
    return leftInds, rightInds
