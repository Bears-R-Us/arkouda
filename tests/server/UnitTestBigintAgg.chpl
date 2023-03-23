use TestBase;
use BigInteger;
use CommAggregation;

config const n = 10**4;

proc main() {
  var remArr = makeDistArray(n, bigint);
  var locArr:[0..<n] bigint;
  forall r in remArr do r = 1;

  forall (l, r) in zip(locArr, remArr) with (var agg = newSrcAggregator(bigint)) {
    agg.copy(l, r);
  }
  assert((+ reduce locArr) == n);

  locArr += 1;

  forall (l, r) in zip(locArr, remArr) with (var agg = newDstAggregator(bigint)) {
    agg.copy(r, l);
  }
  assert((+ reduce remArr) == n*2);

  // TODO negative values
  // TODO massive values
  // TODO variable size limbs
  // TODO mixed bigint-record/mpz-class locale
  // TODO limb size that will grow target
  // TODO limb size that will shrink target
  // TODO limb size that will zero target
}
