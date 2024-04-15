module ArkoudaRandomCompat {
  public use Random;

  private proc is1DRectangularDomain(d) param do
    return d.isRectangular() && d.rank == 1;

  proc sample(arr: [?d] ?t, n: int, withReplacement: bool): [] t throws {
    var r = new randomStream(int);
    return r.choice(arr, size=n, replace=withReplacement);
  }
  proc ref randomStream.skipTo(n: int) do try! this.skipToNth(n);
  proc ref randomStream.permute(const ref arr: [?d] ?t): [] t  where is1DRectangularDomain(d) && isCoercible(this.eltType, d.idxType) {
    return this.permutation(arr);
  }
  proc ref randomStream.permute(d: domain(?)): [] d.idxType  where is1DRectangularDomain(d) && isCoercible(this.eltType, d.idxType) {
    // unfortunately there isn't a domain permutation function so we will create an array to permute
    var domArr: [d] d.idxType = d;
    this.permutation(domArr);
    return domArr;
  }
}
