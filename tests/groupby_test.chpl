use ReductionMsg;
use MultiTypeSymEntry;

proc main() {
  var unique_keys = [ 0,  1,  2,  4,  5,  7,  9, 10, 11];
  var perm_keys = [ 0,  9, 10,
		    0,  4,  5,
		    1,  2,  7,
		    2,  5, 11];
  var segments = [ 0,  1,  1,  1,  1,  1,  1,  2,  3,
		   3,  4,  4,  4,  5,  6,  6,  6,  6,
		   6,  6,  7,  8,  8,  8,  9,  9,  9,
		   9,  9,  9, 10, 10, 11, 11, 11, 11];
  var segD = makeDistArray(segments.size, int);
  segD = segments;
  var values = [ 2,  1,  0,
		 4,  3,  5,
		 6,  7,  8,
		 9, 11, 10];
  var valD = makeDistArray(values.size, int);
  valD = values;
  writeln("segD = ", segD);
  writeln("valD = ", valD);

  write_result("count", unique_keys, perLocCount(segD, valD.size));
  write_result("sum", unique_keys, perLocSum(valD, segD));
  write_result("prod", unique_keys, perLocProduct(valD, segD));
  write_result("mean", unique_keys, perLocMean(valD, segD));
  write_result("min", unique_keys, perLocMin(valD, segD));
  write_result("max", unique_keys, perLocMax(valD, segD));
  write_result("argmin", unique_keys, perLocArgmin(valD, segD));
  write_result("argmax", unique_keys, perLocArgmax(valD, segD));
  write_result("nunique", unique_keys, perLocNumUnique(valD, segD));
}

proc write_result(op, keys, vals) {
  writeln(op);
  for (k, v) in zip(keys, vals) {
    writeln(k, ": ", v);
  }
}