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
    if starts.size == 0:
        return ak.zeros(0, dtype=ak.int64), ak.zeros(0, dtype=ak.int64)
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

def compute_join_size(a, b):
    '''Compute the internal size of a hypothetical join between a and b. Returns
    both the number of elements and number of bytes required for the join.
    '''
    bya = ak.GroupBy(a)
    ua, asize = bya.count()
    byb = ak.GroupBy(b)
    ub, bsize = byb.count()
    afact = asize[ak.in1d(ua, ub)]
    bfact = bsize[ak.in1d(ub, ua)]
    nelem = (afact*bfact).sum()
    nbytes = 3*8*nelem
    return nelem, nbytes

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
    if keep.sum() == 0:
        # Intersection is empty
        return ak.zeros(0, dtype=ak.int64), ak.zeros(0, dtype=ak.int64)
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
        leftWhere = ak.broadcast(fullSegs, whereargs[0][keep], ranges.size)
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
    leftInds = ak.broadcast(filtSegs, ak.arange(left.size)[keep12], filtRanges.size)
    return leftInds, rightInds

def inner_join2(left, right, wherefunc=None, whereargs=None, forceDense=False):
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
    if not isinstance(left, ak.pdarray) or left.dtype != ak.int64 or not isinstance(right, ak.pdarray) or right.dtype != ak.int64:
        raise ValueError("left and right must be pdarray(int64)")
    if wherefunc is not None:
        from inspect import signature
        sample = min((left.size, right.size, 5))
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
    # Only join on intersection
    inter = ak.intersect1d(left, right)
    # Indices of left values present in intersection
    leftInds = ak.arange(left.size)[ak.in1d(left, inter)]
    # Left vals in intersection
    leftFilt = left[leftInds]
    # Indices of right vals present in inter
    rightInds = ak.arange(right.size)[ak.in1d(right, inter)]
    # Right vals in inter
    rightFilt = right[rightInds]
    byLeft = ak.GroupBy(leftFilt)
    byRight = ak.GroupBy(rightFilt)
    maxVal = inter.max()
    if forceDense or maxVal > 3*(left.size + right.size):
        # Remap intersection to dense, 0-up codes
        # Replace left values with dense codes
        uniqLeftVals = byLeft.unique_keys
        uniqLeftCodes = ak.arange(inter.size)[ak.in1d(inter, uniqLeftVals)]
        leftCodes = ak.zeros_like(leftFilt) - 1
        leftCodes[byLeft.permutation] = byLeft.broadcast(uniqLeftCodes, permute=False)
        # Replace right values with dense codes
        uniqRightVals = byRight.unique_keys
        uniqRightCodes = ak.arange(inter.size)[ak.in1d(inter, uniqRightVals)]
        rightCodes = ak.zeros_like(rightFilt) - 1
        rightCodes[byRight.permutation] = byRight.broadcast(uniqRightCodes, permute=False)
        countSize = inter.size
    else:
        uniqLeftCodes = byLeft.unique_keys
        uniqRightCodes = byRight.unique_keys
        leftCodes = leftFilt
        rightCodes = rightFilt
        countSize = maxVal + 1
    # Expand indices to product domain
    # First count occurrences of each code in left and right
    leftCounts = ak.zeros(countSize, dtype=ak.int64)
    leftCounts[uniqLeftCodes] = byLeft.count()[1]
    rightCounts = ak.zeros(countSize, dtype=ak.int64)
    rightCounts[uniqRightCodes] = byRight.count()[1]
    # Repeat each left index as many times as that code occurs in right
    prodLeft = rightCounts[leftCodes]
    leftFullInds = ak.broadcast(ak.cumsum(prodLeft)-prodLeft, leftInds, prodLeft.sum())
    prodRight = leftCounts[rightCodes]
    rightFullInds = ak.broadcast(ak.cumsum(prodRight)-prodRight, rightInds, prodRight.sum())
    # Evaluate where clause
    if wherefunc is None:
        return leftFullInds, rightFullInds
    else:
        # Gather whereargs
        leftWhere = whereargs[0][leftFullInds]
        rightWhere = whereargs[1][rightFullInds]
        # Evaluate wherefunc and filter ranges, recompute segments
        whereSatisfied = wherefunc(leftWhere, rightWhere)
        return leftFullInds[whereSatisfied], rightFullInds[whereSatisfied]
