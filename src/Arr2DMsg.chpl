module Arr2DMsg {
  use GenSymIO;
  use SymEntry2D;

  use ServerConfig;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Message;
  use ServerErrors;
  use Reflection;
  use RandArray;
  use Logging;
  use ServerErrorStrings;

  use OperatorMsg;

  private config const logLevel = ServerConfig.logLevel;
  const randLogger = new Logger(logLevel);

  proc array2DMsg(cmd: string, args: string, st: borrowed SymTab): MsgTuple throws {
    var (dtypeBytes, val, mStr, nStr) = args.splitMsgToTuple(" ", 4);
    var dtype = DType.UNDEF;
    var m: int;
    var n: int;
    var rname:string = "";

    try {
      dtype = str2dtype(dtypeBytes);
      m = mStr: int;
      n = nStr: int;
    } catch {
      var errorMsg = "Error parsing/decoding either dtypeBytes, m, or n";
      gsLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    overMemLimit(2*m*n);

    if dtype == DType.Int64 {
      var entry = new shared SymEntry2D(m, n, int);
      var localA: [{0..#m, 0..#n}] int = val:int;
      entry.a = localA;
      rname = st.nextName();
      st.addEntry(rname, entry);
    } else if dtype == DType.Float64 {
      var entry = new shared SymEntry2D(m, n, real);
      var localA: [{0..#m, 0..#n}] real = val:real;
      entry.a = localA;
      rname = st.nextName();
      st.addEntry(rname, entry);
    } else if dtype == DType.Bool {
      var entry = new shared SymEntry2D(m, n, bool);
      var localA: [{0..#m, 0..#n}] bool = if val == "True" then true else false;
      entry.a = localA;
      rname = st.nextName();
      st.addEntry(rname, entry);
    }

    var msgType = MsgType.NORMAL;
    var msg:string = "";

    if (MsgType.ERROR != msgType) {
      if (msg.isEmpty()) {
        msg = "created " + st.attrib(rname);
      }
      gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),msg);
    }
    return new MsgTuple(msg, msgType);
  }

  proc randint2DMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    var repMsg: string; // response message
    // split request into fields
    var (dtypeStr,aMinStr,aMaxStr,mStr,nStr,seed) = payload.splitMsgToTuple(6);
    var dtype = str2dtype(dtypeStr);
    var m = mStr:int;
    var n = nStr:int;
    var rname = st.nextName();
    
    select (dtype) {
      when (DType.Int64) {
        overMemLimit(8*m*n);
        var aMin = aMinStr:int;
        var aMax = aMaxStr:int;

        var entry = new shared SymEntry2D(m, n, int);
        var localA: [{0..#m, 0..#n}] int;
        entry.a = localA;
        st.addEntry(rname, entry);
        fillInt(entry.a, aMin, aMax, seed);
      }
      when (DType.Float64) {
        overMemLimit(8*m*n);
        var aMin = aMinStr:real;
        var aMax = aMaxStr:real;

        var entry = new shared SymEntry2D(m, n, real);
        var localA: [{0..#m, 0..#n}] real;
        entry.a = localA;
        st.addEntry(rname, entry);
        fillReal(entry.a, aMin, aMax, seed);
      }
      when (DType.Bool) {
        overMemLimit(8*m*n);
        
        var entry = new shared SymEntry2D(m, n, bool);
        var localA: [{0..#m, 0..#n}] bool;
        entry.a = localA;
        st.addEntry(rname, entry);
        fillBool(entry.a, seed);
      }
      otherwise {
        var errorMsg = notImplementedError(pn,dtype);
        randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }

    repMsg = "created " + st.attrib(rname);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc binopvv2DMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    var repMsg: string; // response message

    // split request into fields
    var (op, aname, bname) = payload.splitMsgToTuple(3);

    var rname = st.nextName();
    var left: borrowed GenSymEntry = getGenericTypedArrayEntry(aname, st);
    var right: borrowed GenSymEntry = getGenericTypedArrayEntry(bname, st);

    use Set;
    var boolOps: set(string);
    boolOps.add("<");
    boolOps.add("<=");
    boolOps.add(">");
    boolOps.add(">=");
    boolOps.add("==");
    boolOps.add("!=");

    select (left.dtype, right.dtype) {
    when (DType.Int64, DType.Int64) {
      var l = left: SymEntry2D(int);
      var r = right: SymEntry2D(int);
      if boolOps.contains(op) {
        var e = st.addEntry2D(rname, l.m, l.n, bool);
        return doBinOp(l, r, e, op, rname, pn, st);
      } else if op == "/" {
        var e = st.addEntry2D(rname, l.m, l.n, real);
        return doBinOp(l, r, e, op, rname, pn, st);
      }
      var e = st.addEntry2D(rname, l.m, l.n, int);
      return doBinOp(l, r, e, op, rname, pn, st);
    }
    when (DType.Int64, DType.Float64) {
      var l = left: SymEntry2D(int);
      var r = right: SymEntry2D(real);
      if boolOps.contains(op) {
        var e = st.addEntry2D(rname, l.m, l.n, bool);
        return doBinOp(l, r, e, op, rname, pn, st);
      }
      var e = st.addEntry2D(rname, l.m, l.n, real);
      return doBinOp(l, r, e, op, rname, pn, st);
    }
    when (DType.Float64, DType.Int64) {
      var l = left: SymEntry2D(real);
      var r = right: SymEntry2D(int);
      if boolOps.contains(op) {
        var e = st.addEntry2D(rname, l.m, l.n, bool);
        return doBinOp(l, r, e, op, rname, pn, st);
      }
      var e = st.addEntry2D(rname, l.m, l.n, real);
      return doBinOp(l, r, e, op, rname, pn, st);
    }
    when (DType.Float64, DType.Float64) {
      var l = left: SymEntry2D(real);
      var r = right: SymEntry2D(real);
      if boolOps.contains(op) {
        var e = st.addEntry2D(rname, l.m, l.n, bool);
        return doBinOp(l, r, e, op, rname, pn, st);
      }
      var e = st.addEntry2D(rname, l.m, l.n, real);
      return doBinOp(l, r, e, op, rname, pn, st);
    }
    when (DType.Bool, DType.Bool) {
      var l = left: SymEntry2D(bool);
      var r = right: SymEntry2D(bool);
      var e = st.addEntry2D(rname, l.m, l.n, bool);
      return doBinOp(l, r, e, op, rname, pn, st);
    }
    when (DType.Bool, DType.Int64) {
      var l = left: SymEntry2D(bool);
      var r = right: SymEntry2D(int);
      var e = st.addEntry2D(rname, l.m, l.n, int);
      return doBinOp(l, r, e, op, rname, pn, st);
    }
    when (DType.Int64, DType.Bool) {
      var l = left: SymEntry2D(int);
      var r = right: SymEntry2D(bool);
      var e = st.addEntry2D(rname, l.m, l.n, int);
      return doBinOp(l, r, e, op, rname, pn, st);
    }
    when (DType.Bool, DType.Float64) {
      var l = left: SymEntry2D(bool);
      var r = right: SymEntry2D(real);
      var e = st.addEntry2D(rname, l.m, l.n, real);
      return doBinOp(l, r, e, op, rname, pn, st);
    }
    when (DType.Float64, DType.Bool) {
      var l = left: SymEntry2D(real);
      var r = right: SymEntry2D(bool);
      var e = st.addEntry2D(rname, l.m, l.n, real);
      return doBinOp(l, r, e, op, rname, pn, st);
    }
    }
    return new MsgTuple("Bin op not supported", MsgType.NORMAL);
  }

  proc SymTab.addEntry2D(name: string, m, n, type t): borrowed SymEntry2D(t) throws {
    if t == bool {overMemLimit(m*n);} else {overMemLimit(m*n*numBytes(t));}

    var entry = new shared SymEntry2D(m, n, t);
    if (tab.contains(name)) {
      mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                     "redefined symbol: %s ".format(name));
    } else {
      mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                     "adding symbol: %s ".format(name));
    }

    tab.addOrSet(name, entry);
    return (tab.getBorrowed(name):borrowed GenSymEntry): SymEntry2D(t);
  }

  proc registerMe() {
    use CommandMap;
    registerFunction("array2d", array2DMsg);
    registerFunction("randint2d", randint2DMsg);
    registerFunction("binopvv2d", binopvv2DMsg);
  }
}
