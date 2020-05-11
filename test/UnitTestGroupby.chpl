prototype module UnitTestGroupby
{
  use TestBase;

  use ArgSortMsg;
  use FindSegmentsMsg;
  use ReductionMsg;
  use RandMsg;
  use IndexingMsg;
    
  config const LEN:int;
  config const NKEYS:int;
  config const NVALS:int;
  config const OPERATOR:string = "sum";
  config const STRATEGY:string = "default";
  config const nShow:int = 5;

  proc show(k:[?D] int, v:[D] int, n=5) {
    if (D.size <= 2*n) {
      for (ki, vi) in zip(k, v) {
        try! writeln("%5i: %5i".format(ki, vi));
      }
    } else {
      for (ki, vi) in zip(k[D.low..#n], v[D.low..#n]) {
        try! writeln("%5i: %5i".format(ki, vi));
      }
      writeln("...");
      for (ki, vi) in zip(k[D.high-n+1..#n], v[D.high-n+1..#n]) {
        try! writeln("%5i: %5i".format(ki, vi));
      }
    }
  }

  proc show(k:[?D] int, n=5) {
    if (D.size <= 2*n) {
      for ki in k {
        try! writeln("%5i".format(ki));
      }
    } else {
      for ki in k[D.low..#n] {
        try! writeln("%5i".format(ki));
      }
      writeln("...");
      for ki in k[D.high-n+1..#n] {
        try! writeln("%5i".format(ki));
      }
    }
  }
  
  proc main() {
    writeln("Unit Test for localArgSortMsg");
    var st = new owned SymTab();
    
    var reqMsg: string;
    var repMsg: string;
    
    // create random keys array
    var kname = nameForRandintMsg(LEN, DType.Int64, 0, NKEYS, st);
    var kg = st.lookup(kname);
    var keys = toSymEntry(kg, int);

    // create random vals array
    var vname = nameForRandintMsg(LEN, DType.Int64, 0, NVALS, st);
    var vg = st.lookup(vname);
    var vals = toSymEntry(vg, int);

    writeln("Key: value");
    show(keys.a, vals.a, nShow);
    writeln();
    
    // sort keys and return iv
    var ivname = st.nextName();
    var iv: [keys.aD] int;
    var eMin = min reduce keys.a;
    var eMax = max reduce keys.a;
    var t1 = Time.getCurrentTime();
    if (STRATEGY == "default") {
      writeln("argsortDefault");
      iv = argsortDefault(keys.a);
    } else {
      halt("Unrecognized STRATEGY: ", STRATEGY);
    }
    st.addEntry(ivname, new shared SymEntry(iv));
    writeln("argsort time = ",Time.getCurrentTime() - t1,"sec\n"); try! stdout.flush();

    // find segment boundaries and unique keys
    t1 = Time.getCurrentTime();
    var cmd: string;
    if (STRATEGY == "default") {
      cmd = "findSegments";
      reqMsg = try! "%s %s %i %s %s".format(cmd, ivname, 1, kname, "pdarray");
      repMsg = findSegmentsMsg(reqMsg, st);
    }
    writeln(cmd, " time = ",Time.getCurrentTime() - t1,"sec\n"); try! stdout.flush();
    var (segname, ukiname) = parseTwoNames(repMsg);
    var segg = st.lookup(segname);
    var segs = toSymEntry(segg, int);
    var ukig = st.lookup(ukiname);
    var ukeyinds = toSymEntry(ukig, int);

    // get unique keys
    cmd = "[pdarray]";
    reqMsg = try! "%s %s %s".format(cmd, kname, ukiname);
    t1 = Time.getCurrentTime();
    repMsg = pdarrayIndexMsg(reqMsg, st);
    writeln(cmd, " time = ",Time.getCurrentTime() - t1,"sec\n"); try! stdout.flush();
    writeln(repMsg);
    var ukname = parseName(repMsg);
    var ukg = st.lookup(ukname);
    var ukeys = toSymEntry(ukg, int);

    writeln("Unique key: segment start");
    show(ukeys.a, segs.a, nShow);
    writeln();
    
    // permute the values array
    cmd = "[pdarray]";
    reqMsg = try! "%s %s %s".format(cmd, vname, ivname);
    t1 = Time.getCurrentTime();
    repMsg = pdarrayIndexMsg(reqMsg, st);
    writeln(cmd, " time = ",Time.getCurrentTime() - t1,"sec\n"); try! stdout.flush();
    writeln(repMsg);
    var svname = parseName(repMsg);
    var svg = st.lookup(svname);
    var svals = toSymEntry(svg, int);

    writeln("Sorted vals");
    show(svals.a, nShow);
    writeln();
    
    // do segmented reduction
    t1 = Time.getCurrentTime();
    if (STRATEGY == "default") {
      cmd = "segmentedReduction";
      reqMsg = try! "%s %s %s %s".format(cmd, svname, segname, OPERATOR);
      //writeln(reqMsg);
      repMsg = segmentedReductionMsg(reqMsg, st);
    } 
    writeln(cmd, " time = ",Time.getCurrentTime() - t1,"sec\n"); try! stdout.flush();
    writeln(repMsg);
    var redname = parseName(repMsg);
    var redg = st.lookup(redname);
    var red = toSymEntry(redg, int);

    writeln("Key: reduced value");
    show(ukeys.a, red.a, nShow);
  }
}
