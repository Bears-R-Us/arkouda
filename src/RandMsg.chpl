module RandMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use RandArray;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    private config const logLevel = ServerConfig.logLevel;
    const randLogger = new Logger(logLevel);

    /*
    parse, execute, and respond to randint message
    uniform int in half-open interval [min,max)

    :arg reqMsg: message to process (contains cmd,aMin,aMax,len,dtype)
    */
    proc randintMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        
        const len = msgArgs.get("size").getIntValue();
        const dtype = str2dtype(msgArgs.getValueOf("dtype"));
        const seed = msgArgs.getValueOf("seed");
        const low = msgArgs.get("low");
        const high = msgArgs.get("high");

        // get next symbol name
        var rname = st.nextName();

        // if verbose print action
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
               "cmd: %s len: %i dtype: %s rname: %s aMin: %s: aMax: %s".format(
                                           cmd,len,dtype2str(dtype),rname,low.getValue(),high.getValue()));
        select (dtype) {
            when (DType.Int64) {
                overMemLimit(8*len);
                var aMin = low.getIntValue();
                var aMax = high.getIntValue();
                var t1 = Time.getCurrentTime();
                var e = st.addEntry(rname, len, int);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                   "alloc time = %i sec".format(Time.getCurrentTime() - t1));
                
                t1 = Time.getCurrentTime();
                fillInt(e.a, aMin, aMax, seed);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                  "compute time = %i sec".format(Time.getCurrentTime() - t1));
            }
            when (DType.UInt8) {
                overMemLimit(len);
                var aMin = low.getUInt8Value();
                var aMax = high.getUInt8Value();
                var t1 = Time.getCurrentTime();
                var e = st.addEntry(rname, len, uint(8));
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                     "alloc time = %i sec".format(Time.getCurrentTime() - t1));
                
                t1 = Time.getCurrentTime();
                fillUInt(e.a, aMin, aMax, seed);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "compute time = %i".format(Time.getCurrentTime() - t1));
            }
            when (DType.UInt64) {
                overMemLimit(len);
                var aMin = low.getUIntValue();
                var aMax = high.getUIntValue();
                var t1 = Time.getCurrentTime();
                var e = st.addEntry(rname, len, uint);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                     "alloc time = %i sec".format(Time.getCurrentTime() - t1));
                
                t1 = Time.getCurrentTime();
                fillUInt(e.a, aMin, aMax, seed);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "compute time = %i".format(Time.getCurrentTime() - t1));
            }
            when (DType.Float64) {
                overMemLimit(8*len);
                var aMin = low.getRealValue();
                var aMax = high.getRealValue();
                var t1 = Time.getCurrentTime();
                var e = st.addEntry(rname, len, real);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "alloc time = %i sec".format(Time.getCurrentTime() - t1));
                
                t1 = Time.getCurrentTime();
                fillReal(e.a, aMin, aMax, seed);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                          "compute time = %i sec".format(Time.getCurrentTime() - t1));
            }
            when (DType.Bool) {
                overMemLimit(len);
                var t1 = Time.getCurrentTime();
                var e = st.addEntry(rname, len, bool);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                  "alloc time = %i sec".format(Time.getCurrentTime() - t1));
                
                t1 = Time.getCurrentTime();
                fillBool(e.a, seed);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "compute time = %i sec".format(Time.getCurrentTime() - t1));
            }            
            otherwise {
                var errorMsg = notImplementedError(pn,dtype);
                randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }

        repMsg = "created " + st.attrib(rname);
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc randomNormalMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var pn = Reflection.getRoutineName();
        const len = msgArgs.get("size").getIntValue();
        // Result + 2 scratch arrays
        overMemLimit(3*8*len);
        var rname = st.nextName();
        var entry = new shared SymEntry(len, real);
        fillNormal(entry.a, msgArgs.getValueOf("seed"));
        st.addEntry(rname, entry);

        var repMsg = "created " + st.attrib(rname);
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    
    use CommandMap;
    registerFunction("randint", randintMsg, getModuleName());
    registerFunction("randomNormal", randomNormalMsg, getModuleName());
}
