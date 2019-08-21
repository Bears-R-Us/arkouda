module UnitTestGroupby
{
  use ArgSortMsg;
  use FindSegmentsMsg;
  use ReductionMsg;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use RandMsg;
  use IndexingMsg;

  use Time only;
    
  config const LEN:int;
  config const NKEYS:int;
  config const NVALS:int;
  config const OPERATOR:string = "sum";
  config const STRATEGY:string;
  config const nShow:int = 5;

  proc parseName(s: string, st: borrowed SymTab): string {
    var fields = s.split();
    var name = fields[2];
    return name;
  }

  proc parseTwoNames(s: string, st: borrowed SymTab) {
    var entries = s.split("+");
    var n1 = parseName(entries[1], st);
    var n2 = parseName(entries[2], st);
    return (n1, n2);
  }

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
      for (ki, vi) in zip(k[D.high-n..#n], v[D.high-n..#n]) {
	try! writeln("%5i: %5i".format(ki, vi));
      }
    }
  }

  proc main() {
    writeln("Unit Test for localArgSortMsg");
    var st = new owned SymTab();
    
    var reqMsg: string;
    var repMsg: string;
    
    // create random keys array
    var cmd = "randint";
    var aMin = 0;
    var aMax = NKEYS;
    var len = LEN;
    var dtype = DType.Int64;
    reqMsg = try! "%s %i %i %i %s".format(cmd, aMin, aMax, len, dtype2str(dtype));
    var t1 = Time.getCurrentTime();
    repMsg = randintMsg(reqMsg, st);
    writeln(cmd, " time = ",Time.getCurrentTime() - t1,"sec\n"); try! stdout.flush();
    writeln(repMsg);
    var kname = parseName(repMsg, st);
    var kg = st.lookup(kname);
    var keys = toSymEntry(kg, int);

    // create random vals array
    aMax = NVALS;
    reqMsg = try! "%s %i %i %i %s".format(cmd, aMin, aMax, len, dtype2str(dtype));
    t1 = Time.getCurrentTime();
    repMsg = randintMsg(reqMsg, st);
    writeln(cmd, " time = ",Time.getCurrentTime() - t1,"sec\n"); try! stdout.flush();
    writeln(repMsg);
    var vname = parseName(repMsg, st);
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
    t1 = Time.getCurrentTime();
    if (STRATEGY == "global-count") {
      writeln("argCountSortLocHistGlobHistPDDW");
      iv = argCountSortLocHistGlobHistPDDW(keys.a, eMin, eMax);
    } else if (STRATEGY == "global-DRS") {
      writeln("argsortDRS");
      iv = argsortDRS(keys.a, eMin, eMax);
    } else if (STRATEGY == "per-locale") {
      writeln("perLocaleArgsort");
      iv = perLocaleArgSort(keys.a);
    } else {
      halt("Unrecognized STRATEGY: ", STRATEGY);
    }
    st.addEntry(ivname, new shared SymEntry(iv));
    writeln("argsort time = ",Time.getCurrentTime() - t1,"sec\n"); try! stdout.flush();

    // permute keys array
    cmd = "[pdarray]";
    reqMsg = try! "%s %s %s".format(cmd, kname, ivname);
    t1 = Time.getCurrentTime();
    repMsg = pdarrayIndexMsg(reqMsg, st);
    writeln(cmd, " time = ",Time.getCurrentTime() - t1,"sec\n"); try! stdout.flush();
    writeln(repMsg);
    var skname = parseName(repMsg, st);
    var skg = st.lookup(skname);
    var skeys = toSymEntry(skg, int);

    // find segment boundaries and unique keys
    t1 = Time.getCurrentTime();
    if (STRATEGY == "global-count") || (STRATEGY == "global-DRS") {
      cmd = "findSegments";
      reqMsg = try! "%s %s".format(cmd, skname);
      repMsg = findSegmentsMsg(reqMsg, st);
    } else {
      cmd = "findLocalSegments";
      reqMsg = try! "%s %s".format(cmd, skname);
      repMsg = findLocalSegmentsMsg(reqMsg, st);
    } 
    writeln(cmd, " time = ",Time.getCurrentTime() - t1,"sec\n"); try! stdout.flush();
    var (segname, ukname) = parseTwoNames(repMsg, st);
    var segg = st.lookup(segname);
    var segs = toSymEntry(segg, int);
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
    var svname = parseName(repMsg, st);
    var svg = st.lookup(svname);
    var svals = toSymEntry(svg, int);

    writeln("Sorted keys, vals");
    show(skeys.a, svals.a, nShow);
    writeln();
    
    // do segmented reduction
    t1 = Time.getCurrentTime();
    if (STRATEGY == "global-count") || (STRATEGY == "global-DRS") {
      cmd = "segmentedReduction";
      reqMsg = try! "%s %s %s %s %s".format(cmd, skname, svname, segname, OPERATOR);
      //writeln(reqMsg);
      repMsg = segmentedReductionMsg(reqMsg, st);
    } else {
      cmd = "segmentedLocalRdx";
      reqMsg = try! "%s %s %s %s %s".format(cmd, skname, svname, segname, OPERATOR);
      //writeln(reqMsg);
      repMsg = segmentedLocalRdxMsg(reqMsg, st);
    }
    writeln(cmd, " time = ",Time.getCurrentTime() - t1,"sec\n"); try! stdout.flush();
    writeln(repMsg);
    var redname = parseName(repMsg, st);
    var redg = st.lookup(redname);
    var red = toSymEntry(redg, int);

    writeln("Key: reduced value");
    show(ukeys.a, red.a, nShow);
  }
}
