module ReductionMsgFunctions
{
    use BigInteger;
    use List;
    use AryUtil;
    use ReductionMsg;
    use SliceReductionOps;
    
    @arkouda.registerCommand
    proc anyAll(const ref x:[?d] ?t): bool throws
    {
      use SliceReductionOps;
      return anySlice(x, x.domain);
    }

    @arkouda.registerCommand
    proc any(const ref x:[?d] ?t, axis: list(int)): [] bool throws
      {
      use SliceReductionOps;
      type opType = bool;
      const (valid, axes) = validateNegativeAxes(axis, x.rank);
      if !valid {
        throw new Error("Invalid axis value(s) '%?' in slicing reduction".format(axis));
      } else {
        const outShape = reducedShape(x.shape, axes);
        var ret = makeDistArray((...outShape), opType);
        forall (sliceDom, sliceIdx) in axisSlices(x.domain, axes)
          do ret[sliceIdx] = anySlice(x, sliceDom);
        return ret;
      }
    }

    @arkouda.registerCommand
    proc allAll(const ref x:[?d] ?t): bool throws
    {
      use SliceReductionOps;
      return allSlice(x, x.domain);
    }

    @arkouda.registerCommand
    proc all(const ref x:[?d] ?t, axis: list(int)): [] bool throws
      {
      use SliceReductionOps;
      type opType = bool;
      const (valid, axes) = validateNegativeAxes(axis, x.rank);
      if !valid {
        throw new Error("Invalid axis value(s) '%?' in slicing reduction".format(axis));
      } else {
        const outShape = reducedShape(x.shape, axes);
        var ret = makeDistArray((...outShape), opType);
        forall (sliceDom, sliceIdx) in axisSlices(x.domain, axes)
          do ret[sliceIdx] = allSlice(x, sliceDom);
        return ret;
      }
    }

    @arkouda.registerCommand
    proc isSortedAll(const ref x:[?d] ?t): bool throws
    {
      use SliceReductionOps;
      return isSortedSlice(x, x.domain);
    }

    @arkouda.registerCommand
    proc isSorted(const ref x:[?d] ?t, axis: list(int)): [] bool throws
      {
      use SliceReductionOps;
      type opType = bool;
      const (valid, axes) = validateNegativeAxes(axis, x.rank);
      if !valid {
        throw new Error("Invalid axis value(s) '%?' in slicing reduction".format(axis));
      } else {
        const outShape = reducedShape(x.shape, axes);
        var ret = makeDistArray((...outShape), opType);
        forall (sliceDom, sliceIdx) in axisSlices(x.domain, axes)
          do ret[sliceIdx] = isSortedSlice(x, sliceDom);
        return ret;
      }
    }

    @arkouda.registerCommand
    proc isSortedLocallyAll(const ref x:[?d] ?t): bool throws
    {
      use SliceReductionOps;
      return isSortedLocallySlice(x, x.domain);
    }

    @arkouda.registerCommand
    proc isSortedLocally(const ref x:[?d] ?t, axis: list(int)): [] bool throws
      {
      use SliceReductionOps;
      type opType = bool;
      const (valid, axes) = validateNegativeAxes(axis, x.rank);
      if !valid {
        throw new Error("Invalid axis value(s) '%?' in slicing reduction".format(axis));
      } else {
        const outShape = reducedShape(x.shape, axes);
        var ret = makeDistArray((...outShape), opType);
        forall (sliceDom, sliceIdx) in axisSlices(x.domain, axes)
          do ret[sliceIdx] = isSortedLocallySlice(x, sliceDom);
        return ret;
      }
    }

    @arkouda.registerCommand
    proc argmaxAll(const ref x:[?d] ?t): d.idxType throws
    where (t != bigint) {
      use SliceReductionOps;
      if d.rank == 1 {
        return argmaxSlice(x, d):d.idxType;
      } else {
        const ord = new orderer(x.shape);
        const ret = ord.indexToOrder(argmaxSlice(x, d)):d.idxType;
        return ret;
      }
    }

    @arkouda.registerCommand
    proc argmax(const ref x:[?d] ?t, axis: int): [] d.idxType throws
      where (t != bigint) && (d.rank > 1) {
      use SliceReductionOps;
      const axisArry = [axis];
      const outShape = reducedShape(x.shape, axisArry);
      var ret = makeDistArray((...outShape), d.idxType);
      forall sliceIdx in domOffAxis(d, axisArry) {
        const sliceDom = domOnAxis(d, sliceIdx, axis);
        ret[sliceIdx] = argmaxSlice(x, sliceDom)[axis]:d.idxType;
      }
      return ret;
    }


    @arkouda.registerCommand
    proc argminAll(const ref x:[?d] ?t): d.idxType throws
    where (t != bigint) {
      use SliceReductionOps;
      if d.rank == 1 {
        return argminSlice(x, d):d.idxType;
      } else {
        const ord = new orderer(x.shape);
        const ret = ord.indexToOrder(argminSlice(x, d)):d.idxType;
        return ret;
      }
    }

    @arkouda.registerCommand
    proc argmin(const ref x:[?d] ?t, axis: int): [] d.idxType throws
      where (t != bigint) && (d.rank > 1) {
      use SliceReductionOps;
      const axisArry = [axis];
      const outShape = reducedShape(x.shape, axisArry);
      var ret = makeDistArray((...outShape), d.idxType);
      forall sliceIdx in domOffAxis(d, axisArry) {
        const sliceDom = domOnAxis(d, sliceIdx, axis);
        ret[sliceIdx] = argminSlice(x, sliceDom)[axis]:d.idxType;
      }
      return ret;
    }


}
