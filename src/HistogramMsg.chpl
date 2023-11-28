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


    /* histogram takes a pdarray and returns a pdarray with the histogram in it */
    proc histogram2DMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const xBins = msgArgs.get("xBins").getIntValue();
        const yBins = msgArgs.get("yBins").getIntValue();
        const xName = msgArgs.getValueOf("x");
        const yName = msgArgs.getValueOf("y");

        // get next symbol name
        var rname = st.nextName();
        hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "cmd: %s xName: %s yName: %s xBins: %i yBins: %i rname: %s".doFormat(cmd, xName, yName, xBins, yBins, rname));

        var xGenEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(xName, st);
        var yGenEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(yName, st);

        // helper nested procedure
        proc histogramHelper(type t) throws {
            var x = toSymEntry(xGenEnt,t);
            var y = toSymEntry(yGenEnt,t);
            var xMin = min reduce x.a;
            var xMax = max reduce x.a;
            var yMin = min reduce y.a;
            var yMax = max reduce y.a;
            var xBinWidth:real = (xMax - xMin):real / xBins:real;
            var yBinWidth:real = (yMax - yMin):real / yBins:real;
            hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "xBinWidth %r yBinWidth %r".doFormat(xBinWidth, yBinWidth));

            var totBins = xBins * yBins;
            if (totBins <= sBound) {
                hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "%? <= %?".doFormat(totBins,sBound));
                var hist = histogramReduceIntent(x.a, y.a, xMin, xMax, yMin, yMax, xBins, yBins, xBinWidth, yBinWidth);
                st.addEntry(rname, createSymEntry(hist));
            }
            else if (totBins <= mBound) {
                hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "%? <= %?".doFormat(totBins,mBound));
                var hist = histogramLocalAtomic(x.a, y.a, xMin, xMax, yMin, yMax, xBins, yBins, xBinWidth, yBinWidth);
                st.addEntry(rname, createSymEntry(hist));
            }
            else {
                hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                                "%? > %?".doFormat(totBins,mBound));
                var hist = histogramGlobalAtomic(x.a, y.a, xMin, xMax, yMin, yMax, xBins, yBins, xBinWidth, yBinWidth);
                st.addEntry(rname, createSymEntry(hist));
            }
        }
        select (xGenEnt.dtype, yGenEnt.dtype) {
            when (DType.Int64, DType.Int64)   {histogramHelper(int);}
            when (DType.UInt64, DType.UInt64)  {histogramHelper(uint);}
            when (DType.Float64, DType.Float64) {histogramHelper(real);}
            otherwise {
                var errorMsg = notImplementedError(pn,"("+dtype2str(xGenEnt.dtype)+","+dtype2str(yGenEnt.dtype)+")");
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
    registerFunction("histogram2D", histogram2DMsg, getModuleName());
}
