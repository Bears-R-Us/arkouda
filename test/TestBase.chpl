public use Time;
public use CommDiagnostics;
public use IO;

public use ServerConfig;
public use ServerErrorStrings;

public use MultiTypeSymEntry;
public use MultiTypeSymbolTable;

public use SymArrayDmap;
public use SegmentedArray;
public use RandArray;
public use AryUtil;


// Diag helpers (timers, comm diags, etc.)

config var printTimes = true;
config var printDiags = false;
config var printDiagsSum = false;
// Don't gather comm diags by default, they have a non-trivial perf overhead
const dfltGatherDiags = printDiags || printDiagsSum;

proc byteToMB(b) {
  return b / 1024 / 1024;
}

record CommDiagSummary {
  const GETS, PUTS, ONS: uint;
}

record Diags {
  var T: Timer;
  var elapsedTime: real;
  var gatherDiags = dfltGatherDiags;
  var D: [LocaleSpace] commDiagnostics;

  proc init() {
    this.complete();
    resetCommDiagnostics();
    D = getCommDiagnostics();
  }

  proc start() {
    if gatherDiags then startCommDiagnostics();
    T.start();
  }

  proc stop(param name="", printTime=printTimes, printDiag=printDiags, printDiagSum=printDiagsSum) {
    T.stop();
    if gatherDiags then stopCommDiagnostics();

    elapsedTime = T.elapsed();
    if gatherDiags then D = getCommDiagnostics();

    T.clear();
    if gatherDiags then resetCommDiagnostics();

    if !gatherDiags && (printDiag || printDiagSum) then warning("gatherDiags was not enabled");
    param s = if name != "" then name + ": " else "";
    if printTime    then writef("%s%.2drs\n", s, this.elapsed());
    if printDiag    then writef('%s%s\n',     s, this.comm():string);
    if printDiagSum then writef("%s%t\n",     s, this.commSum());
  }

  proc elapsed() {
    return elapsedTime;
  }

  proc comm() {
    if !gatherDiags then warning("gatherDiags was not enabled");
    return D;
  }

  proc commSum() {
    if !gatherDiags then warning("gatherDiags was not enabled");
    const GETS = + reduce (D.get + D.get_nb);
    const PUTS = + reduce (D.put + D.put_nb);
    const ONS  = + reduce (D.execute_on + D.execute_on_fast + D.execute_on_nb);
    return new CommDiagSummary(GETS, PUTS, ONS);
  }
}


// Message helpers

proc parseName(s: string): string {
  const low = [1..1].domain.low;
  var fields = s.split();
  return fields[low+1];
}

proc parseTwoNames(s: string): (string, string) {
  var (nameOne, nameTwo) = s.splitMsgToTuple('+', 2);
  return (parseName(nameOne), parseName(nameTwo));
}


config const writeReqRep = false;
proc writeReq(req: string) {
  if writeReqRep then writeln(req);
}
proc writeRep(rep: string) {
  if writeReqRep then writeln(rep);
}

config const writeSegStr = false;
config const showSegStrLen = 5;
proc writeSegString(msg: string, ss: SegString) {
  if writeSegStr {
    writeln(msg);
    ss.show(showSegStrLen);
  }
}


proc nameForRandintMsg(len: int, dtype:DType, aMin: int, aMax: int, st: borrowed SymTab) {
  use RandMsg;
  const payload = try! "%i %s %i %i None".format(len, dtype2str(dtype), aMin, aMax);
  writeReq(payload);
  const repMsg = randintMsg(cmd='randint', payload=payload, st).msg;
  writeRep(repMsg);
  return parseName(repMsg);
}
