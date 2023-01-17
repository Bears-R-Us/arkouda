use TestBase;
use BigInteger;
use CommAggregation;

proc main() {
  var bi = 1:bigint;
  bi <<= 20;

  var size = 5;
  var D = makeDistDom(size);
  var distArr: [D] bigint;
  var locArr = [bi - 1, bi, bi+1, bi+2, bi+3];
  var iv = [4, 3, 2, 1, 0];

  var d: Diags;
  d.start();
  forall (i, la) in zip(iv, locArr) with (var agg = newDstAggregator(bigint)) {
    agg.copy(distArr[i], la);
  }
  d.stop("Scatter");
  writeln(distArr);
  writeln(locArr);
  writeln();

  d.start();
  on Locales[numLocales-1] {
    for d in distArr.localSlice(distArr.localSubdomain()) {
      d += 1;
    }
  }
  d.stop("Add");


  writeln(locArr);
  d.start();
  forall (la, da) in zip(locArr, distArr) with (var agg = newSrcAggregator(bigint)) {
    agg.copy(la, da);
  }
  d.stop("Gather");
  writeln(locArr);
}
