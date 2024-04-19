module ArkoudaRandomCompat {
  use Random.PCGRandom only PCGRandomStream;

  private proc is1DRectangularDomain(d) param do
    return d.isRectangular() && d.rank == 1;

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
    proc ref permute(const ref arr: [?d] ?t): [] t  where is1DRectangularDomain(d) && isCoercible(this.eltType, d.idxType) {
      return r.permutation(arr);
    }
    proc ref permute(d: domain): [] d.idxType  where is1DRectangularDomain(d) && isCoercible(this.eltType, d.idxType) {
      // unfortunately there isn't a domain permutation function so we will create an array to permute
      var domArr: [d] d.idxType = d;
      r.permutation(domArr);
      return domArr;
    }
    proc ref sample(d: domain, n: int, withReplacement = false): [] d.idxType throws  where is1DRectangularDomain(d) {
      return r.choice(d, n, withReplacement);
    }
    proc ref next(): eltType do return r.getNext();
    proc skipTo(n: int) do try! r.skipToNth(n);
  }

  proc sample(arr: [?d] ?t, n: int, withReplacement: bool): [] t throws {
    var r = new randomStream(int);
    return r.r.choice(arr, size=n, replace=withReplacement);
  }
}
