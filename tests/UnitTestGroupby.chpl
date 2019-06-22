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
  config const OPERATION:string;
  config const STRATEGY:string;

  proc lookupInt(s: string, st: borrowed SymTab) {
    var fields = s.split();
    var name = fields[2];
    var gse = st.lookup(fields[2]);
    return (name, toSymEntry(gse, int));
  }

  proc lookupTwoInts(s: string, st: borrowedSymTab) {

  }

  proc main() {
    writeln("Unit Test for localArgSortMsg");
    var st = new owned SymTab();
    
    var reqMsg: string;
    var repMsg: string;
    
    // create an array filled with random int64 returned in symbol table
    var cmd = "randint";
    var aMin = 0;
    var aMax = NKEYS;
    var len = LEN;
    var dtype = DType.Int64;
    reqMsg = try! "%s %i %i %i %s".format(cmd, aMin, aMax, len, dtype2str(dtype));
    var t1 = Time.getCurrentTime();
    repMsg = randintMsg(reqMsg, st);
    writeln(cmd, " time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
    writeln(repMsg);
    var (kname, keys) = lookupInt(repMsg, st);

    var ivname = st.nextName();
    var iv: SymEntry(int);
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
    writeln("argsort time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

    cmd = "[pdarray]";
    reqMsg = try! "%s %s %s".format(cmd, kname, ivname);
    t1 = Time.getCurrentTime();
    repMsg = pdarrayIndexMsg(reqMsg, st);
    writeln(cmd, " time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
    writeln(repMsg);
    var (skname, skeys) = lookupInt(repMsg, st);

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
    writeln(cmd, " time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
    var (segname, segs, ukname, ukeys) = parseTwoInts(repMsg);

    // permute the values

    // do segmented reduction
}
