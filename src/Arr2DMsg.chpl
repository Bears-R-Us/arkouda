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

  use OperatorMsg;

  proc array2DMsg(cmd: string, args: string, st: borrowed SymTab): MsgTuple throws {
    var (val, mStr, nStr) = args.splitMsgToTuple(" ", 3);
    var m = mStr: int;
    var n = nStr: int;
    var dtype = DType.Int64;
    var entry = new shared SymEntry2D(m, n, int);
    var localA: [{0..#m, 0..#n}] int = val:int;
    entry.a = localA;
    var rname = st.nextName();
    st.addEntry(rname, entry);

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
    var (aMinStr,aMaxStr,mStr,nStr,seed) = payload.splitMsgToTuple(5);
    var m = mStr:int;
    var n = nStr:int;

    overMemLimit(8*m*n);
    var aMin = aMinStr:int;
    var aMax = aMaxStr:int;
    var entry = new shared SymEntry2D(m, n, int);

    var localA: [{0..#m, 0..#n}] int;
    entry.a = localA;
    var rname = st.nextName();
    st.addEntry(rname, entry);
    fillInt(entry.a, aMin, aMax, seed);

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
    // TODO: get actual m and n dimensions from left/right instead
    // of hardcoding a 2x2 array
    var e = st.addEntry2D(rname, 2, 2, int);

    var l = left: SymEntry2D(int);
    var r = right: SymEntry2D(int);

    return doBinOp(l, r, e, op, rname, pn, st);
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
    return (tab.getBorrowed(name):borrowed GenSymEntry): SymEntry2D(int);
  }

  proc registerMe() {
    use CommandMap;
    registerFunction("array2d", array2DMsg);
    registerFunction("randint2d", randint2DMsg);
    registerFunction("binopvv2d", binopvv2DMsg);
  }
}
