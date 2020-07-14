use TestBase;

use GenSymIO;
use SegmentedMsg;
use UniqueMsg;

config const filename = "test-data/strings.hdf";
config const dsetName = "segstring";
config const testIndex = 0;
config const testStart = 0;
config const testStop = 5;
config const testString = "Comp141988";
config const testSubstr = "Comp";

proc parseNames(msg) {
  return parseTwoNames(msg);
}

proc parseNames(msg, param k) {
  return msg.splitMsgToTuple('+', k);
}

proc main() {
  var st = new owned SymTab();
  var cmd = "readhdf";
  var reqMsg = "%s %i %jt".format(dsetName, 1, [filename]);
  writeln(">>> ", reqMsg);
  var repMsg = readhdfMsg(cmd=cmd, payload=reqMsg.encode(), st);
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
  for i in 0..#5 {
    writeln("%i: %s".format(i, strings[i]));
  }

  // strings[int]
  writeln();
  reqMsg = "%s %s %s %s %i".format("intIndex", "str", segName, valName, testIndex);
  writeln(">>> ", reqMsg);
  repMsg = segmentedIndexMsg(cmd="segmentedIndex", payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);

  // strings[slice]
  writeln();
  reqMsg = "%s %s %s %s %i %i %i".format("sliceIndex", "str", segName, valName, testStart, testStop, 1);
  writeln(">>> ", reqMsg);
  repMsg = segmentedIndexMsg(cmd="segmentedIndex", payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);
  var (sliceSegName, sliceValName) = parseNames(repMsg);
  var strSlice = new owned SegString(sliceSegName, sliceValName, st);
  printAry("strSlice offsets: ", strSlice.offsets.a);
  printAry("strSlice raw bytes: ", strSlice.values.a);
  writeln("strSlice:");
  for i in 0..#strSlice.size {
    writeln("%i: %s".format(i, strSlice[i]));
  }

  // strings[pdarray]
  writeln();
  var cname = st.nextName();
  var gcountdown = st.addEntry(cname, new shared SymEntry(5, int));
  var countdown = toSymEntry(gcountdown, int);
  countdown.a = [4, 3, 2, 1, 0];
  reqMsg = "%s %s %s %s %s".format("pdarrayIndex", "str", segName, valName, cname);
  writeln(">>> ", reqMsg);
  repMsg = segmentedIndexMsg(cmd="segmentedIndex", payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);
  var (a, b) = parseNames(repMsg);
  var strCountdown = new owned SegString(a, b, st);
  printAry("strCountdown offsets: ", strCountdown.offsets.a);
  printAry("strCountdown raw bytes: ", strCountdown.values.a);
  for i in 0..#strCountdown.size {
    writeln("%i: %s".format(i, strCountdown[i]));
  }

  // strings == val
  writeln();
  reqMsg = "%s %s %s %s %s %s".format("==", "str", segName, valName, "str", testString);
  writeln(">>> ", reqMsg);
  repMsg = segBinopvsMsg(cmd="segBinopvs", payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);
  var aname = parseName(repMsg);
  var giv = st.lookup(aname);
  var iv = toSymEntry(giv, bool);
  var steps = + scan iv.a;
  var pop = steps[iv.aD.high];
  printAry("strings == %s: ".format(testString), iv.a);
  writeln("pop = ", pop);
  if (pop > 0) {
    var inds: [0..#pop] int;
    [(idx, present, i) in zip(iv.aD, iv.a, steps)] if present {inds[i-1] = idx;}
    printAry("inds: ", inds);
    var diff = inds[1..#(pop-1)] - inds[0..#(pop-1)];
    var consecutive = && reduce (diff == 1);
    writeln("consecutive? ", consecutive);
  }

  // strings != val
  writeln();
  reqMsg = "%s %s %s %s %s %s".format("!=", "str", segName, valName, "str", testString);
  writeln(">>> ", reqMsg);
  repMsg = segBinopvsMsg(cmd="segBinopvs", payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);
  aname = parseName(repMsg);
  var giv2 = st.lookup(aname);
  var iv2 = toSymEntry(giv2, bool);
  printAry("strings != %s: ".format(testString), iv2.a);
  var check = (&& reduce (iv.a != iv2.a));
  writeln("All strings should either be equal or not equal to testStr: >>> ", check, " <<<");
    
  // group strings
  writeln();
  reqMsg = "%s %s %s".format("str", segName, valName);
  writeln(">>> ", reqMsg);
  repMsg = segGroupMsg(cmd="segGroup", payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);
  var permname = parseName(repMsg);
  var gperm = st.lookup(permname);
  var perm = toSymEntry(gperm, int);

  // permute strings
  writeln();
  reqMsg = "%s %s %s %s %s".format("pdarrayIndex", "str", segName, valName, permname);
  writeln(">>> ", reqMsg);
  repMsg = segmentedIndexMsg(cmd="segmentedIndex", payload=reqMsg.encode(), st);
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

  if !SegmentedArrayUseHash {
    writeln("Checking if strings are sorted..."); stdout.flush();
    writeln("Strings sorted? >>> ", permStrings.isSorted(), " <<<"); stdout.flush();
  }
  
  // check that permuted strings grouped 
  // strings == val
  writeln();
  reqMsg = "%s %s %s %s %s %s".format("==", "str", permSegName, permValName, "str", testString);
  writeln(">>> ", reqMsg);
  repMsg = segBinopvsMsg(cmd="segBinopvs", payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);
  aname = parseName(repMsg);
  giv = st.lookup(aname);
  iv = toSymEntry(giv, bool);
  steps = + scan iv.a;
  pop = steps[iv.aD.high];
  printAry("strings == %s: ".format(testString), iv.a);
  writeln("pop = ", pop);
  if pop > 0 {
    var permInds: [0..#pop] int;
    [(idx, present, i) in zip(iv.aD, iv.a, steps)] if present {permInds[i-1] = idx;}
    //printAry("permInds: ", permInds);
    writeln("permInds: ", permInds);
    var permDiff = permInds[1..#(pop-1)] - permInds[0..#(pop-1)];
    var consecutive = && reduce (permDiff == 1);
    writeln("consecutive? ", consecutive);
  }
    
  // compress out the matches
  // strings[pdarray(bool)]
  writeln();
  reqMsg = "%s %s %s %s %s".format("pdarrayIndex", "str", permSegName, permValName, aname);
  writeln(">>> ", reqMsg);
  repMsg = segmentedIndexMsg(cmd="segmentedIndexMsg", payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);
  (a, b) = parseNames(repMsg);
  var strMatches = new owned SegString(a, b, st);
  printAry("strMatches offsets: ", strMatches.offsets.a);
  printAry("strMatches raw bytes: ", strMatches.values.a);
  for i in 0..#min(strMatches.size, 5) {
    writeln("%i: %s".format(i, strMatches[i]));
  }

  // Unique
  writeln();
  reqMsg = "%s %s %s".format("str", "+".join([permSegName, permValName]), "True");
  writeln(">>> ", reqMsg);
  repMsg = uniqueMsg(cmd="unique", payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);
  var (myA, myB, myC) = parseNames(repMsg, 3);
  var uniqueStr = new owned SegString(myA, myB, st);
  var gcounts = st.lookup(myC);
  var counts = toSymEntry(gcounts, int);
  writeln("Found %t unique strings".format(uniqueStr.size));
  for i in 0..#min(uniqueStr.size, 5) {
    writeln("%i: %i, %s".format(i, counts.a[i], uniqueStr[i]));
  }
  writeln("Sum of counts equals number of strings? >>> ", (+ reduce counts.a) == strings.size, " <<<");

  // In1d(strings, strSlice)
  writeln();
  reqMsg = "%s %s %s %s %s %s %s".format("str", segName, valName, "str", sliceSegName, sliceValName, "False");
  writeln(">>> ", reqMsg);
  repMsg = segIn1dMsg(cmd="segIn1d", payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);
  aname = parseName(repMsg);
  giv = st.lookup(aname);
  iv = toSymEntry(giv, bool);
  pop = + reduce iv.a;
  writeln("Found %t matches".format(pop));

  // Contains
  writeln();
  reqMsg = "%s %s %s %s %s %s".format("contains", "str", segName, valName, "str", testSubstr);
  writeln(">>> ", reqMsg);
  repMsg = segmentedEfuncMsg(cmd="segEfunc", payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);
  aname = parseName(repMsg);
  giv = st.lookup(aname);
  iv = toSymEntry(giv, bool);
  pop = + reduce iv.a;
  writeln("Found %t strings containing %s".format(pop, testSubstr));

  // Starts with
  writeln();
  reqMsg = "%s %s %s %s %s %s".format("startswith", "str", segName, valName, "str", testSubstr);
  writeln(">>> ", reqMsg);
  repMsg = segmentedEfuncMsg(cmd="segEfunc", payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);
  aname = parseName(repMsg);
  giv = st.lookup(aname);
  iv = toSymEntry(giv, bool);
  pop = + reduce iv.a;
  writeln("Found %t strings starting with %s".format(pop, testSubstr));

  // Ends with
  writeln();
  reqMsg = "%s %s %s %s %s %s".format("endswith", "str", segName, valName, "str", testSubstr);
  writeln(">>> ", reqMsg);
  repMsg = segmentedEfuncMsg(cmd="segEfunc", payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);
  aname = parseName(repMsg);
  giv = st.lookup(aname);
  iv = toSymEntry(giv, bool);
  pop = + reduce iv.a;
  writeln("Found %t strings ending with %s".format(pop, testSubstr));
  
  /* for i in testStart..testStop { */
  /*   var hashval = permStrings.murmurHash(permStrings.values.a[permStrings.offsets.a[i]..(permStrings.offsets.a[i+1]-1)]); */
  /*   writeln("%i: %s, (%016xu, %016xu)".format(i, permStrings[i], hashval[1], hashval[2])); */
  /* } */

  /* // Manually hash strings and sort hashes */
  /* writeln(); */
  /* var hashes = strings.hash(); */
  /* var manPerm = radixSortLSD_ranks(hashes); */
  /* writeln("Sorts equal? ", && reduce (manPerm == perm.a)); */
  /* writeln("Manually hashed values:"); */
  /* for i in testStart..testStop { */
  /*   var myStr = strings[manPerm[i]]; */
  /*   var myHash = hashes[manPerm[i]]; */
  /*   writeln("%i: %s, (%016xu, %016xu)".format(i, myStr, myHash[1], myHash[2])); */
  /* } */
  
}
