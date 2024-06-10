/* min-k reduction 
 * Stores the sorted minimum k values onto the server
 */

module KExtremeMsg
{
    use ServerConfig;

    use Time;
    use Math only;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use SegmentedString;
    use ServerErrorStrings;

    use KReduce;
    use Indexing;
    use RadixSortLSD;
    use ArraySetopsMsg;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const keLogger = new Logger(logLevel, logChannel);

    /*
    Parse, execute, and respond to a mink message
    :arg reqMsg: request containing (name,k,returnIndices)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (MsgTuple) response message
    */
    proc minkMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("array"), st);
        var k = msgArgs.get("k").getIntValue();
        var returnIndices = msgArgs.get("rtnInd").getBoolValue();

        select(gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var aV = if !returnIndices then computeExtremaValues(e.a, k) else computeExtremaIndices(e.a, k);
                st.addEntry(vname, createSymEntry(aV));

                repMsg = "created " + st.attrib(vname);
                keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            when (DType.UInt64) {
                var e = toSymEntry(gEnt,uint);
                if !returnIndices {
                    var aV = computeExtremaValues(e.a, k);
                    st.addEntry(vname, createSymEntry(aV));

                    repMsg = "created " + st.attrib(vname);
                    keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                    return new MsgTuple(repMsg, MsgType.NORMAL);
                } else {
                    var aV = computeExtremaIndices(e.a, k);
                    st.addEntry(vname, createSymEntry(aV));

                    repMsg = "created " + st.attrib(vname);
                    keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                    return new MsgTuple(repMsg, MsgType.NORMAL);
                }
            }
            when (DType.Float64) {
                var e = toSymEntry(gEnt,real);
                if !returnIndices {
                    var aV = computeExtremaValues(e.a, k);
                    st.addEntry(vname, createSymEntry(aV));

                    repMsg = "created " + st.attrib(vname);
                    keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                    return new MsgTuple(repMsg, MsgType.NORMAL);
                } else {
                    var aV = computeExtremaIndices(e.a, k);
                    st.addEntry(vname, createSymEntry(aV));

                    repMsg = "created " + st.attrib(vname);
                    keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                    return new MsgTuple(repMsg, MsgType.NORMAL);
                }
            }
            otherwise {
               var errorMsg = notImplementedError("mink",gEnt.dtype);
               keLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
               return new MsgTuple(errorMsg, MsgType.ERROR);              
            }
        }
    }
    
   /*
    Parse, execute, and respond to a maxk message
    :arg reqMsg: request containing (name,k,returnIndices)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (MsgTuple) response message
    */
    proc maxkMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message

        var vname = st.nextName();
        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("array"), st);
        var k = msgArgs.get("k").getIntValue();
        var returnIndices = msgArgs.get("rtnInd").getBoolValue();

        select(gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var aV = if !returnIndices then computeExtremaValues(e.a, k, false) else computeExtremaIndices(e.a, k, false);
                st.addEntry(vname, createSymEntry(aV));

                repMsg = "created " + st.attrib(vname);
                keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
           }
            when (DType.UInt64) {
                var e = toSymEntry(gEnt,uint);
               if !returnIndices {
                   var aV = computeExtremaValues(e.a, k, false);
                   st.addEntry(vname, createSymEntry(aV));

                   repMsg = "created " + st.attrib(vname);
                   keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                   return new MsgTuple(repMsg, MsgType.NORMAL);
               } else {
                   var aV = computeExtremaIndices(e.a, k, false);
                   st.addEntry(vname, createSymEntry(aV));

                   repMsg = "created " + st.attrib(vname);
                   keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                   return new MsgTuple(repMsg, MsgType.NORMAL);
               }
           }
           when (DType.Float64) {
                var e = toSymEntry(gEnt,real);
               if !returnIndices {
                   var aV = computeExtremaValues(e.a, k, false);
                   st.addEntry(vname, createSymEntry(aV));

                   repMsg = "created " + st.attrib(vname);
                   keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                   return new MsgTuple(repMsg, MsgType.NORMAL);
               } else {
                   var aV = computeExtremaIndices(e.a, k, false);
                   st.addEntry(vname, createSymEntry(aV));

                   repMsg = "created " + st.attrib(vname);
                   keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                   return new MsgTuple(repMsg, MsgType.NORMAL);
               
               }
           }
           otherwise {
               var errorMsg = notImplementedError("maxk",gEnt.dtype);
               keLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
               return new MsgTuple(errorMsg, MsgType.ERROR);
           }
        }
    }

    use CommandMap;
    registerFunction("mink", minkMsg, getModuleName());
    registerFunction("maxk", maxkMsg, getModuleName());
}
