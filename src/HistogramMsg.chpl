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
                      "cmd: %s name: %s bins: %i rname: %s".format(cmd, name, bins, rname));

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        // helper nested procedure
        proc histogramHelper(type t) throws {
          var e = toSymEntry(gEnt,t);
          var aMin = msgArgs.get("minVal").toScalar(real);
          var aMax = msgArgs.get("maxVal").toScalar(real);
          var binWidth:real = (aMax - aMin):real / bins:real;
          hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                           "binWidth %r".format(binWidth));

          if (bins <= sBound) {
              hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                           "%? <= %?".format(bins,sBound));
              var hist = histogramReduceIntent(e.a, aMin, aMax, bins, binWidth);
              st.addEntry(rname, createSymEntry(hist));
          }
          else if (bins <= mBound) {
              hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                           "%? <= %?".format(bins,mBound));
              var hist = histogramLocalAtomic(e.a, aMin, aMax, bins, binWidth);
              st.addEntry(rname, createSymEntry(hist));
          }
          else {
              hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "%? > %?".format(bins,mBound));
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
                      "cmd: %s xName: %s yName: %s xBins: %i yBins: %i rname: %s".format(cmd, xName, yName, xBins, yBins, rname));

        var xGenEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(xName, st);
        var yGenEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(yName, st);

        // helper nested procedure
        proc histogramHelper(type t1, type t2) throws {
            var x = toSymEntry(xGenEnt,t1);
            var y = toSymEntry(yGenEnt,t2);
            var xMin = msgArgs.get("xMin").toScalar(real);
            var xMax = msgArgs.get("xMax").toScalar(real);
            var yMin = msgArgs.get("yMin").toScalar(real);
            var yMax = msgArgs.get("yMax").toScalar(real);
            var xBinWidth:real = (xMax - xMin):real / xBins:real;
            var yBinWidth:real = (yMax - yMin):real / yBins:real;
            hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "xBinWidth %r yBinWidth %r".format(xBinWidth, yBinWidth));

            var totBins = xBins * yBins;
            if (totBins <= sBound) {
                hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "%? <= %?".format(totBins,sBound));
                var hist = histogramReduceIntent(x.a, y.a, xMin, xMax, yMin, yMax, xBins, yBins, xBinWidth, yBinWidth);
                st.addEntry(rname, createSymEntry(hist));
            }
            else if (totBins <= mBound) {
                hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "%? <= %?".format(totBins,mBound));
                var hist = histogramLocalAtomic(x.a, y.a, xMin, xMax, yMin, yMax, xBins, yBins, xBinWidth, yBinWidth);
                st.addEntry(rname, createSymEntry(hist));
            }
            else {
                hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                                "%? > %?".format(totBins,mBound));
                var hist = histogramGlobalAtomic(x.a, y.a, xMin, xMax, yMin, yMax, xBins, yBins, xBinWidth, yBinWidth);
                st.addEntry(rname, createSymEntry(hist));
            }
        }
        select (xGenEnt.dtype, yGenEnt.dtype) {
            when (DType.Int64, DType.Int64) {histogramHelper(int, int);}
            when (DType.Int64, DType.UInt64) {histogramHelper(int, uint);}
            when (DType.Int64, DType.Float64) {histogramHelper(int, real);}
            when (DType.UInt64, DType.Int64) {histogramHelper(uint, int);}
            when (DType.UInt64, DType.UInt64) {histogramHelper(uint, uint);}
            when (DType.UInt64, DType.Float64) {histogramHelper(uint, real);}
            when (DType.Float64, DType.Int64) {histogramHelper(real, int);}
            when (DType.Float64, DType.UInt64) {histogramHelper(real, uint);}
            when (DType.Float64, DType.Float64) {histogramHelper(real, real);}
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


    proc histogramdDMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const numDims = msgArgs.get("num_dims").getIntValue();
        const numSamples = msgArgs.get("num_samples").getIntValue();
        const binsStrs = msgArgs.get("bins").getList(numDims);
        const bins = try! [b in binsStrs] b:int;
        const names = msgArgs.get("sample").getList(numDims);
        const dimProd = msgArgs.get("dim_prod").toScalarArray(int, numDims);
        const totNumBins = * reduce bins;

        // get next symbol name
        var rname = st.nextName();
        hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "cmd: %s name: %? bins: %? rname: %s".format(cmd, names, bins, rname));

        var gEnts = try! [name in names] getGenericTypedArrayEntry(name, st);

        // helper nested procedure
        proc histogramHelper(type t) throws {
            var rangeMin = msgArgs.get("rangeMin").toScalarArray(real, numDims);
            var rangeMax = msgArgs.get("rangeMax").toScalarArray(real, numDims);
            var indices = makeDistArray(numSamples, int);

            // compute index into flattened array:
            // for each dimension calculate which bin that value falls into in parallel
            // we can then scale by the product of the previous dimensions to get that dimensions impact on the flattened index
            // summing all these together gives the index into the flattened array
            hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "compute indices");
            for (gEnt, stride, bin, aMin, aMax) in zip(gEnts, dimProd, bins, rangeMin, rangeMax) {
                var e = toSymEntry(gEnt,t);
                var binWidth:real = (aMax - aMin):real / bin:real;
                forall (v, idx) in zip(e.a, indices) {
                  if idx < 0 {
                    // value for this index was out of bounds in a previous sample, skip
                  } else if ! histValWithinRange(v, aMin, aMax) {
                    idx = -1;
                  } else {
                    const vBin = histValToBin(v, aMin, aMax, bin, binWidth);
                    idx += (vBin * stride):int;
                  }
                }
            }

            // The result will be here.
            const histSE = createSymEntry(totNumBins, real);
            st.addEntry(rname, histSE);
            ref hist = histSE.a;

            // 3 different implementations depending on size of histogram
            // this is due to the time memory tradeoff between creating one/few atomic arrays
            // or many non-atomic arrays and reducing them
            if (totNumBins <= sBound) {
                // "histogramReduceIntent"
                // small number of buckets (so histogram is relatively small):
                // each task gets it's own copy of the histogram and they're reduced
                hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "%? <= %?".format(bins,sBound));
                var gHist: [0..#totNumBins] int;

                // count into per-task/per-locale histogram and then reduce as tasks complete
                forall idx in indices with (+ reduce gHist) {
                    if idx<0 then continue;
                    gHist[idx] += 1;
                }

                hist = gHist;
            }
            else if (totNumBins <= mBound) {
                // "histogramLocalAtomic"
                // medium number of buckets:
                // each locale gets it's own atomic copy of the histogram and these are reduced together
                hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "%? <= %?".format(bins,mBound));
                use PrivateDist;

                // allocate per-locale atomic histogram
                var atomicHist: [PrivateSpace] [0..#totNumBins] atomic int;

                // count into per-locale private atomic histogram
                forall idx in indices {
                    if idx<0 then continue;
                    atomicHist[here.id][idx].add(1);
                }

                // +reduce across per-locale histograms to get counts
                var lHist: [0..#totNumBins] int;
                forall i in PrivateSpace with (+ reduce lHist) {
                    lHist reduce= atomicHist[i].read();
                }

                hist = lHist;
            }
            else {

                // "histogramGlobalAtomic"
                // large number of buckets:
                // one global atomic histogram
                hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                                "%? > %?".format(bins,mBound));
                // allocate single global atomic histogram
                var atomicHist: [makeDistDom(totNumBins)] atomic int;

                // count into atomic histogram
                forall idx in indices {
                    if idx<0 then continue;
                    atomicHist[idx].add(1);
                }

                // copy from atomic histogram to normal histogram
                [(e,ae) in zip(hist, atomicHist)] e = ae.read();
            }
        }

        select (gEnts[0].dtype) {
            when (DType.Int64)   {histogramHelper(int);}
            when (DType.UInt64)  {histogramHelper(uint);}
            when (DType.Float64) {histogramHelper(real);}
            otherwise {
                var errorMsg = notImplementedError(pn,gEnts[0].dtype);
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
    registerFunction("histogramdD", histogramdDMsg, getModuleName());
}
