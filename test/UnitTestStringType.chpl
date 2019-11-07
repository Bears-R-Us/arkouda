use GenSymIO;
use MultiTypeSymbolTable;
use SegmentedArray;

config const filename = "test-data/strings.hdf";
config const dsetName = "segstring";
config const testIndex = 0;
config const testStart = 0;
config const testStop = 5;
config const testString = "Comp141988";

proc parseNames(msg) {
  var halves = msg.split('+', 1);
  var fieldsA = halves[1].split();
  var nameA = fieldsA[2];
  var fieldsB = halves[2].split();
  var nameB = fieldsB[2];
  return (nameA, nameB);
}

proc main() {
  var st = new owned SymTab();
  var cmd = "readhdf";
  var reqMsg = "%s %s %i %jt".format(cmd, dsetName, 1, [filename]);
  writeln(">>> ", reqMsg);
  var repMsg = readhdfMsg(reqMsg, st);
  writeln("<<< ", repMsg);
  if repMsg.startsWith("Error") {
    halt();
  }
  var (segName, valName) = parseNames(repMsg);
  /* var gs = st.lookup(segName); */
  /* var segs = toSymEntry(gs, int); */
  /* var vs = st.lookup(valName); */
  /* var vals = toSymEntry(vs, uint(8)); */
  var strings = new owned SegString(segName, valName, st);
  printAry("Segment offsets: ", strings.offsets.a);
  printAry("Raw bytes: ", strings.values.a);
  writeln("Strings:");
  /* for (i, start, end) in zip(0..#5, segs.a[0..#5], segs.a[1..#5]) { */
  /*   writeln("%i: %s".format(i, interpretAsString(vals.a[start..end-1], copy=false))); */
  /* } */
  for i in 0..#5 {
    writeln("%i: %s".format(i, strings[i]));
  }

  // strings[int]
  reqMsg = "%s %s %s %s %s %i".format("segmentedIndexMsg", "intIndex", "string", segName, valName, testIndex);
  writeln(">>> ", reqMsg);
  repMsg = segmentedIndexMsg(reqMsg, st);
  writeln("<<< ", repMsg);

  // strings[slice]
  reqMsg = "%s %s %s %s %s %i %i %i".format("segmentedIndexMsg", "sliceIndex", "string", segName, valName, testStart, testStop, 1);
  writeln(">>> ", reqMsg);
  repMsg = segmentedIndexMsg(reqMsg, st);
  writeln("<<< ", repMsg);
  var (a, b) = parseNames(repMsg);
  var strSlice = new owned SegString(a, b, st);
  printAry("strSlice offsets: ", strSlice.offsets.a);
  printAry("strSlice raw bytes: ", strSlice.values.a);
  writeln("strSlice:");
  for i in 0..#strSlice.size {
    writeln("%i: %s".format(i, strSlice[i]));
  }

  // strings == val
  reqMsg = "%s %s %s %s %s %s %s".format("segBinopvs", "==", "string", segName, valName, "string", testString);
  writeln(">>> ", reqMsg);
  repMsg = segBinopvsMsg(reqMsg, st);
  writeln("<<< ", repMsg);
  var fields = repMsg.split();
  var aname = fields[2];
  var giv = st.lookup(aname);
  var iv = toSymEntry(giv, bool);
  printAry("strings == %s: ".format(testString), iv.a);
  writeln("pop = ", + reduce iv.a);

  // group strings
  reqMsg = "%s %s %s %s".format("segGroup", "string", segName, valName);
  writeln(">>> ", reqMsg);
  repMsg = segGroupMsg(reqMsg, st);
  writeln("<<< ", repMsg);
  fields = repMsg.split();
  var permname = fields[2];
  var gperm = st.lookup(permname);
  var perm = toSymEntry(gperm, int);

  // permute strings
  reqMsg = "%s %s %s %s %s %s".format("segmentedIndex", "pdarrayIndex", "string", segName, valName, permname);
  writeln(">>> ", reqMsg);
  repMsg = segmentedIndexMsg(reqMsg, st);
  writeln("<<< ", repMsg);
  var (permSegName, permValName) = parseNames(repMsg);
  var permStrings = new owned SegString(permSegName, permValName, st);
  writeln("grouped strings:");
  for i in 0..#5 {
    writeln("%i: %s".format(i, permStrings[i]));
  }
  writeln("...");
  for i in strings.size-6..#5 {
    writeln("%i: %s".format(i, permStrings[i]));
  }
}