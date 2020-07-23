/*
 * 'mink' reduction implementation. Returns 
 * vector of k elements of type eltType.
 */

module MinK {
  use KExtreme;

  class mink : ReduceScanOp {
    type eltType;
    const k: int;
    const isMin=true;

    // create a new heap per task
    var v = new kextreme(eltType=eltType, size=k, isMinReduction=isMin);

    proc identity {
      var v = new kextreme(eltType=eltType, size=k, isMinReduction=isMin); return v;
    }

    proc accumulateOntoState(ref v, value: eltType) {
      v.push(value);
    }

    proc accumulate(value: eltType) {
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
    proc combine(state: borrowed mink(eltType)) {
      v._data = merge(v, state.v);
    }

    proc generate() {
      return v;
    }

    proc clone() {
      return new unmanaged mink(eltType=eltType, k=k, isMin=isMin);
    }
  }

  /*
   * Instinatiate the mink reduction class
   * so that a custom `k` value can be
   * passed into the class
   */
  proc computeMyMink(arr, kval:int, isMin=true) {
    var minkInstance = new unmanaged mink(eltType=int, k=kval, isMin=isMin);
    var result = minkInstance.identity;
    [ elm in arr with (minkInstance reduce result) ]
      result reduce= elm;
    delete minkInstance;
    return result;
  }
}
