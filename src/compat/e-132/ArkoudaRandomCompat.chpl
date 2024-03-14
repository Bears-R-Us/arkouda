module ArkoudaRandomCompat {
  use Random.PCGRandom only PCGRandomStream;
  record randomStream {
    type eltType = int;
    forwarding var r: shared PCGRandomStream(eltType);
    proc init(type t) {
      eltType = t;
      r = new shared PCGRandomStream(eltType);
    }
    proc init(type t, seed) {
      eltType = t;
      r = new shared PCGRandomStream(eltType, seed);
    }
    proc ref fill(ref arr: []) where arr.isRectangular() {
      r.fillRandom(arr);
    }
    proc ref fill(ref arr: [], min: arr.eltType, max: arr.eltType) where arr.isRectangular() {
      r.fillRandom(arr, min, max);
    }
    proc skipTo(n: int) do r.skipToNth(n);
  }

  proc sample(arr: [?d] ?t, n: int, withReplacement: bool): [] t throws {
    var r = new randomStream(int);
    return r.r.choice(arr, size=n, replace=withReplacement);
  }
}
