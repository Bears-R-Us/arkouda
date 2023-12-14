module RandMsg
{
    use ServerConfig;
    
    use ArkoudaTimeCompat as Time;
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
    private config const logChannel = ServerConfig.logChannel;
    const randLogger = new Logger(logLevel, logChannel);

    /*
    parse, execute, and respond to randint message
    uniform int in half-open interval [min,max)

    :arg reqMsg: message to process (contains cmd,aMin,aMax,len,dtype)
    */
    @arkouda.registerND
    proc randintMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        
        const shape = msgArgs.get("shape").getTuple(nd);
        const dtype = str2dtype(msgArgs.getValueOf("dtype"));
        const seed = msgArgs.getValueOf("seed");
        const low = msgArgs.get("low");
        const high = msgArgs.get("high");

        var len = 1;
        for s in shape do len *= s;

        // get next symbol name
        var rname = st.nextName();

        // if verbose print action
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
               "cmd: %s shape: %? dtype: %s rname: %s aMin: %s: aMax: %s".doFormat(
                                           cmd,shape,dtype2str(dtype),rname,low.getValue(),high.getValue()));
        select (dtype) {
            when (DType.Int64) {
                var aMin = low.getIntValue();
                var aMax = high.getIntValue();
                var t1 = Time.timeSinceEpoch().totalSeconds();
                var e = st.addEntry(rname, (...shape), int);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                   "alloc time = %i sec".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
                
                t1 = Time.timeSinceEpoch().totalSeconds();
                fillInt(e.a, aMin, aMax, seed);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                  "compute time = %i sec".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
            }
            when (DType.UInt8) {
                overMemLimit(len);
                var aMin = low.getUInt8Value();
                var aMax = high.getUInt8Value();
                var t1 = Time.timeSinceEpoch().totalSeconds();
                var e = st.addEntry(rname, (...shape), uint(8));
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                     "alloc time = %i sec".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
                
                t1 = Time.timeSinceEpoch().totalSeconds();
                fillUInt(e.a, aMin, aMax, seed);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "compute time = %i".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
            }
            when (DType.UInt64) {
                overMemLimit(len);
                var aMin = low.getUIntValue();
                var aMax = high.getUIntValue();
                var t1 = Time.timeSinceEpoch().totalSeconds();
                var e = st.addEntry(rname, (...shape), uint);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                     "alloc time = %i sec".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
                
                t1 = Time.timeSinceEpoch().totalSeconds();
                fillUInt(e.a, aMin, aMax, seed);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "compute time = %i".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
            }
            when (DType.Float64) {
                overMemLimit(8*len);
                var aMin = low.getRealValue();
                var aMax = high.getRealValue();
                var t1 = Time.timeSinceEpoch().totalSeconds();
                var e = st.addEntry(rname, (...shape), real);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "alloc time = %i sec".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
                
                t1 = Time.timeSinceEpoch().totalSeconds();
                fillReal(e.a, aMin, aMax, seed);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                          "compute time = %i sec".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
            }
            when (DType.Bool) {
                overMemLimit(len);
                var t1 = Time.timeSinceEpoch().totalSeconds();
                var e = st.addEntry(rname, (...shape), bool);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                  "alloc time = %i sec".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
                
                t1 = Time.timeSinceEpoch().totalSeconds();
                fillBool(e.a, seed);
                randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "compute time = %i sec".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
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
        var entry = createSymEntry(len, real);
        fillNormal(entry.a, msgArgs.getValueOf("seed"));
        st.addEntry(rname, entry);

        var repMsg = "created " + st.attrib(rname);
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    
    use CommandMap;
    registerFunction("randomNormal", randomNormalMsg, getModuleName());
}
