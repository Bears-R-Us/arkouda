/*
 * 'kreduce' reduction implementation. Returns 
 * array of the k extreme values of an array 
 * of type eltType. Whether it will be the max
 * or min values is specified through the "isMin"
 * field (true gives min values, false gives max
 * values).
 */

module KReduce {
  use KExtreme;

  class kreduce : ReduceScanOp {
    type eltType;
    const k: int;
    const isMin=true;

    // create a new heap per task
    var v = new kextreme(eltType=eltType, size=k, isMinReduction=isMin);

    proc identity {
      var v = new kextreme(eltType=eltType, size=k, isMinReduction=isMin); return v;
    }

    proc accumulateOntoState(ref v, value: (eltType, int)) {
      v.push(value);
    }

    proc accumulate(value: (eltType, int)) {
      accumulateOntoState(v, value);
    }

    proc accumulate(accumState: kextreme(eltType)) {
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
  proc computeExtremaInds(arr: [?D] ?t, kval: int, isMin=true) {
    var kred = new unmanaged kreduce(eltType=t, k=kval, isMin=isMin);
    var result = kred.identity;

    var tmpArr: [D] (t, D.idxType);
    forall (elem, val, i) in zip(tmpArr, arr, arr.domain) {
      elem = (val, i);
    }
    [ elm in tmpArr with (kred reduce result) ]
      result reduce= elm;
    delete kred;

    return result;
  }

  /*
   * Return the `kval` largest elements of an array `arr`
   */
  proc computeExtrema(arr, kval:int, isMin=true) {
    const extrema = computeExtremaInds(arr, kval, isMin);
    var res = [elem in extrema] elem(0);
    
    return res;
  }

  /*
   * Return the indices of the `kval` largest 
   * elements of an array `arr`
   */
  proc computeInds(arr, kval:int, isMin=true) {
    const extrema = computeExtremaInds(arr, kval, isMin);
    var res = [elem in extrema] elem(1);
    
    return res;
  }
}
