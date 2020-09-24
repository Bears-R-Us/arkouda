module KReduce {
  use Extremas;

 /*
    :class:`KReduce` is a user defined reduction.
    Returns array of the k extreme values of an array of type eltType.
    Whether it will be the max or min values is specified through the "isMin" field (true gives min values, false gives max values).
   
   This uses the current user-defined reduction framwork,
   creating a unique :class: `KReduce` instance on each
   task, which then calls the ``accumulate()`` procedure
   on the input array to build up the data structure in
   a heap like manner. Once that is complete, each task
   will migrate to locale 0 and call the ``combine()``
   procedure on each heap, which will reduce all of the
   task :class: `KReduce` instances down into the final
   result of the reduction.

   This reduction performs well with a small `k` 
   value, but sees a significant drop off in 
   performance as `k` grows larger, although the 
   exact value for this threshold is system dependent. 
   This is because this is a per-task reduciton, so 
   as `k` increases, the amount of random access that 
   occurs will increase, slowing down access. This 
   operation is equivalent to performing an `argsort()` 
   on an array, and then using a slice of these values 
   to retrieve the first or last `k` indices from that 
   sorted indice array. 
  */
  class kreduce : ReduceScanOp {
    type eltType;
    const k: int;
    const isMin=true;

    // create a new heap per task
    var v = new KExtreme(eltType=eltType, size=k, isMinReduction=isMin);

    proc identity {
      var v = new KExtreme(eltType=eltType, size=k, isMinReduction=isMin); return v;
    }

    proc accumulateOntoState(ref v, value: (eltType, int)) {
      v.push(value);
    }

    proc accumulate(value: (eltType, int)) {
      accumulateOntoState(v, value);
    }

    proc accumulate(accumState: KExtreme(eltType)) {
      for stateValue in accumState {
        accumulate(stateValue);
      }
    }

    proc accumulate(accumState: []) {
      for stateValue in accumState {
        accumulate(stateValue);
      }
    }

    // when combining, merge instead of
    // accumulating each individual value
    proc combine(state: borrowed kreduce(eltType)) {
      v._data = merge(v, state.v);
    }

    proc generate() {
      if !v.isSorted then v.doSort();
      return v;
    }

    proc clone() {
      return new unmanaged kreduce(eltType=eltType, k=k, isMin=isMin);
    }
  }

  /*
   * Instinatiate the kreduce reduction class
   * so that a custom `k` value can be
   * passed into the class, returning a tuple
   * array that contains both the extreme values
   * and the indices associated with those values.
   */
  proc computeExtremaValuesAndInds(arr: [?D] ?t, kval: int, isMin=true) {
    var kred = new unmanaged kreduce(eltType=t, k=kval, isMin=isMin);
    var result = kred.identity;
    forall idx in zip(arr, arr.domain) with (kred reduce result) {
      result reduce= idx;
    }
    delete kred;
    return result;
  }

  /*
   * Return the `kval` largest elements of an array `arr`
   */
  proc computeExtremaValues(arr, kval:int, isMin=true) {
    const extrema = computeExtremaValuesAndInds(arr, kval, isMin);
    var res = [elem in extrema] elem(0);
    
    return res;
  }

  /*
   * Return the indices of the `kval` largest 
   * elements of an array `arr`
   */
  proc computeExtremaIndices(arr, kval:int, isMin=true) {
    const extrema = computeExtremaValuesAndInds(arr, kval, isMin);
    var res = [elem in extrema] elem(1);
    
    return res;
  }
}
