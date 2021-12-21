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

  proc registerMe() {
    use CommandMap;
    registerFunction("array2d", array2DMsg);
    registerFunction("randint2d", randint2DMsg);
  }
}
