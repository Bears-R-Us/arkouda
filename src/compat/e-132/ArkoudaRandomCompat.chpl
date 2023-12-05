module ArkoudaRandomCompat {
  use Random.PCGRandom only PCGRandomStream;
  record randomStream {
    type eltType;
    forwarding var r: owned PCGRandomStream(eltType);
    proc init(eltType) {
      r = new owned PCGRandomStream(eltType);
    }
    proc init(eltType, seed) {
      r = new owned PCGRandomStream(eltType, seed);
    }
    
  }
}
