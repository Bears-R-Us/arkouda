/* min-k reduction 
 * Stores the sorted minimum k values onto the server
 */

module KExtremeMsg
{
    use ServerConfig;

    use Time only;
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
    const keLogger = new Logger(logLevel);

    /*
    Parse, execute, and respond to a mink message
    :arg reqMsg: request containing (name,k,returnIndices)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (MsgTuple) response message
    */
    proc minkMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, k, returnIndices) = payload.splitMsgToTuple(3);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        select(gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var aV;

                if !stringtobool(returnIndices) {
                    aV = computeExtremaValues(e.a, k:int);
                } else {
                    aV = computeExtremaIndices(e.a, k:int);
                }

                st.addEntry(vname, new shared SymEntry(aV));

                repMsg = "created " + st.attrib(vname);
                keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            when (DType.Float64) {
                if !stringtobool(returnIndices) {
                    var e = toSymEntry(gEnt,real);
                    var aV = computeExtremaValues(e.a, k:int);
                    st.addEntry(vname, new shared SymEntry(aV));

                    repMsg = "created " + st.attrib(vname);
                    keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                    return new MsgTuple(repMsg, MsgType.NORMAL);
                } else {
                    var e = toSymEntry(gEnt,real);
                    var aV = computeExtremaIndices(e.a, k:int);
                    st.addEntry(vname, new shared SymEntry(aV));

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
    proc maxkMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, k, returnIndices) = payload.splitMsgToTuple(3);

        var vname = st.nextName();
        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        select(gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var aV;
                if !stringtobool(returnIndices) {
                    aV = computeExtremaValues(e.a, k:int, false);
                } else {
                    aV = computeExtremaIndices(e.a, k:int, false);
                }

                st.addEntry(vname, new shared SymEntry(aV));

                repMsg = "created " + st.attrib(vname);
                keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
           }
           when (DType.Float64) {
               if !stringtobool(returnIndices) {
                   var e = toSymEntry(gEnt,real);
                   var aV = computeExtremaValues(e.a, k:int, false);

                   st.addEntry(vname, new shared SymEntry(aV));

                   repMsg = "created " + st.attrib(vname);
                   keLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                   return new MsgTuple(repMsg, MsgType.NORMAL);
               } else {
                   var e = toSymEntry(gEnt,real);
                   var aV = computeExtremaIndices(e.a, k:int, false);

                   st.addEntry(vname, new shared SymEntry(aV));

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

    proc registerMe() {
      use CommandMap;
      registerFunction("mink", minkMsg, getModuleName());
      registerFunction("maxk", maxkMsg, getModuleName());
    }
}
