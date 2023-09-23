module HistogramMsg
{
    use ServerConfig;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use Histogram;
    use Message;
 
    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const hgmLogger = new Logger(logLevel, logChannel);
    
    private config const sBound = 2**12;
    private config const mBound = 2**25;

    /* histogram takes a pdarray and returns a pdarray with the histogram in it */
    proc histogramMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const bins = msgArgs.get("bins").getIntValue();
        const name = msgArgs.getValueOf("array");
        
        // get next symbol name
        var rname = st.nextName();
        hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "cmd: %s name: %s bins: %i rname: %s".doFormat(cmd, name, bins, rname));

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        // helper nested procedure
        proc histogramHelper(type t) throws {
          var e = toSymEntry(gEnt,t);
          var aMin = min reduce e.a;
          var aMax = max reduce e.a;
          var binWidth:real = (aMax - aMin):real / bins:real;
          hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                           "binWidth %r".doFormat(binWidth));

          if (bins <= sBound) {
              hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                           "%? <= %?".doFormat(bins,sBound));
              var hist = histogramReduceIntent(e.a, aMin, aMax, bins, binWidth);
              st.addEntry(rname, createSymEntry(hist));
          }
          else if (bins <= mBound) {
              hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                           "%? <= %?".doFormat(bins,mBound));
              var hist = histogramLocalAtomic(e.a, aMin, aMax, bins, binWidth);
              st.addEntry(rname, createSymEntry(hist));
          }
          else {
              hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "%? > %?".doFormat(bins,mBound));
              var hist = histogramGlobalAtomic(e.a, aMin, aMax, bins, binWidth);
              st.addEntry(rname, createSymEntry(hist));
          }
        }

        select (gEnt.dtype) {
            when (DType.Int64)   {histogramHelper(int);}
            when (DType.UInt64)   {histogramHelper(uint);}
            when (DType.Float64) {histogramHelper(real);}
            otherwise {
                var errorMsg = notImplementedError(pn,gEnt.dtype);
                hgmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);  
                return new MsgTuple(errorMsg, MsgType.ERROR);             
            }
        }
        
        repMsg = "created " + st.attrib(rname);
        hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    use CommandMap;
    registerFunction("histogram", histogramMsg, getModuleName());
}
