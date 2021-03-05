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
    writeln("Unit Test for groupBy");
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
    var d: Diags;
    if (STRATEGY == "default") {
      writeln("argsortDefault");
      d.start();
      iv = argsortDefault(keys.a);
      d.stop("argsortDefault");
    } else {
      halt("Unrecognized STRATEGY: ", STRATEGY);
    }
    st.addEntry(ivname, new shared SymEntry(iv));

    // find segment boundaries and unique keys
    var cmd: string;
    if (STRATEGY == "default") {
      cmd = "findSegments";
      reqMsg = try! "%s %i %s %s".format(ivname, 1, kname, "pdarray");
      d.start();
      repMsg = findSegmentsMsg(cmd=cmd, payload=reqMsg, st).msg;
      d.stop("findSegmentsMsg");
    }
    var (segname, ukiname) = parseTwoNames(repMsg);
    var segg = st.lookup(segname);
    var segs = toSymEntry(segg, int);
    var ukig = st.lookup(ukiname);
    var ukeyinds = toSymEntry(ukig, int);

    // get unique keys
    cmd = "[pdarray]";
    reqMsg = try! "%s %s".format(kname, ukiname);
    d.start();
    repMsg = pdarrayIndexMsg(cmd=cmd, payload=reqMsg, st).msg;
    d.stop("pdarrayIndexMsg");
    writeRep(repMsg);
    var ukname = parseName(repMsg);
    var ukg = st.lookup(ukname);
    var ukeys = toSymEntry(ukg, int);

    writeln("Unique key: segment start");
    show(ukeys.a, segs.a, nShow);
    writeln();
    
    // permute the values array
    cmd = "[pdarray]";
    reqMsg = try! "%s %s".format(vname, ivname);
    d.start();
    repMsg = pdarrayIndexMsg(cmd=cmd, payload=reqMsg, st).msg;
    d.stop("pdarrayIndexMsg");
    writeRep(repMsg);
    var svname = parseName(repMsg);
    var svg = st.lookup(svname);
    var svals = toSymEntry(svg, int);

    writeln("Sorted vals");
    show(svals.a, nShow);
    writeln();
    
    // do segmented reduction
    if (STRATEGY == "default") {
      cmd = "segmentedReduction";
      var skip_nan="False";
      reqMsg = try! "%s %s %s %s".format(svname, segname, OPERATOR, skip_nan);
      //writeReq(reqMsg);
      d.start();
      repMsg = segmentedReductionMsg(cmd=cmd, payload=reqMsg, st).msg;
      d.stop("segmentedReductionMsg");
    } 
    writeRep(repMsg);
    var redname = parseName(repMsg);
    var redg = st.lookup(redname);
    var red = toSymEntry(redg, int);

    writeln("Key: reduced value");
    show(ukeys.a, red.a, nShow);
  }
}
