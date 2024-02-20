module ArkoudaRandomCompat {
  use Random.PCGRandom only PCGRandomStream;
  record randomStream {
    type eltType = int;
    forwarding var r: owned PCGRandomStream(eltType);
    proc init(type t) {
      eltType = t;
      r = new owned PCGRandomStream(eltType);
    }
    proc init(type t, seed) {
      eltType = t;
      r = new owned PCGRandomStream(eltType, seed);
    }
  }

  proc sample(arr: [?d] ?t, n: int, withReplacement: bool): [] t throws {
    var r = new randomStream(int);
    return r.r.choice(arr, size=n, replace=withReplacement);
  }
}
