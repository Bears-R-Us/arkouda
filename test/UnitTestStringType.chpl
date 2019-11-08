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

  // strings[pdarray]
  var cname = st.nextName();
  var gcountdown = st.addEntry(cname, new shared SymEntry(5, int));
  var countdown = toSymEntry(gcountdown, int);
  countdown.a = [4, 3, 2, 1, 0];
  reqMsg = "%s %s %s %s %s %s".format("segmentedIndexMsg", "pdarrayIndex", "string", segName, valName, cname);
  writeln(">>> ", reqMsg);
  repMsg = segmentedIndexMsg(reqMsg, st);
  writeln("<<< ", repMsg);
  (a, b) = parseNames(repMsg);
  var strCountdown = new owned SegString(a, b, st);
  printAry("strCountdown offsets: ", strCountdown.offsets.a);
  printAry("strCountdown raw bytes: ", strCountdown.values.a);
  for i in 0..#strCountdown.size {
    writeln("%i: %s".format(i, strCountdown[i]));
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
  var steps = + scan iv.a;
  var pop = steps[iv.aD.high];
  printAry("strings == %s: ".format(testString), iv.a);
  writeln("pop = ", pop);
  var inds: [0..#pop] int;
  [(idx, present, i) in zip(iv.aD, iv.a, steps)] if present {inds[i-1] = idx;}
  printAry("inds: ", inds);
  var diff = inds[1..#(pop-1)] - inds[0..#(pop-1)];
  var consecutive = && reduce (diff == 1);
  writeln("consecutive? ", consecutive);

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

  // check that permuted strings grouped 
  // strings == val
  reqMsg = "%s %s %s %s %s %s %s".format("segBinopvs", "==", "string", permSegName, permValName, "string", testString);
  writeln(">>> ", reqMsg);
  repMsg = segBinopvsMsg(reqMsg, st);
  writeln("<<< ", repMsg);
  fields = repMsg.split();
  aname = fields[2];
  giv = st.lookup(aname);
  iv = toSymEntry(giv, bool);
  steps = + scan iv.a;
  pop = steps[iv.aD.high];
  printAry("strings == %s: ".format(testString), iv.a);
  writeln("pop = ", pop);
  var permInds: [0..#pop] int;
  [(idx, present, i) in zip(iv.aD, iv.a, steps)] if present {permInds[i-1] = idx;}
  //printAry("permInds: ", permInds);
  writeln("permInds: ", permInds);
  var permDiff = permInds[1..#(pop-1)] - permInds[0..#(pop-1)];
  consecutive = && reduce (permDiff == 1);
  writeln("consecutive? ", consecutive);

  // compress out the matches
  // strings[pdarray(bool)]
  reqMsg = "%s %s %s %s %s %s".format("segmentedIndexMsg", "pdarrayIndex", "string", permSegName, permValName, aname);
  writeln(">>> ", reqMsg);
  repMsg = segmentedIndexMsg(reqMsg, st);
  writeln("<<< ", repMsg);
  (a, b) = parseNames(repMsg);
  var strMatches = new owned SegString(a, b, st);
  printAry("strMatches offsets: ", strMatches.offsets.a);
  printAry("strMatches raw bytes: ", strMatches.values.a);
  for i in 0..#strMatches.size {
    writeln("%i: %s".format(i, strMatches[i]));
  }

  for i in 5177..5183 {
    var hashval = permStrings.internalHash(permStrings.values.a[permStrings.offsets.a[i]..(permStrings.offsets.a[i+1]-1)]);
    writeln("%i: %s, (%i, %i)".format(i, permStrings[i], hashval[1], hashval[2]));
  }
}