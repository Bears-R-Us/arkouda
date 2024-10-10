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
          var aMin = min reduce e.a;
          var aMax = max reduce e.a;
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
            var xMin = min reduce x.a;
            var xMax = max reduce x.a;
            var yMin = min reduce y.a;
            var yMax = max reduce y.a;
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
        const dimProdName = msgArgs.getValueOf("dim_prod");
        const totNumBins = * reduce bins;
        
        // get next symbol name
        var rname = st.nextName();
        hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "cmd: %s name: %? bins: %? rname: %s".format(cmd, names, bins, rname));

        var gEnts = try! [name in names] getGenericTypedArrayEntry(name, st);
        var dimProdGenEnt = getGenericTypedArrayEntry(dimProdName, st);
        var dimProd = toSymEntry(dimProdGenEnt,int);

        // helper nested procedure
        proc histogramHelper(type t) throws {
            var indices = makeDistArray(numSamples, int);
            // 3 different implementations depending on size of histogram
            // this is due to the time memory tradeoff between creating one/few atomic arrays
            // or many non-atomic arrays and reducing them

            // compute index into flattened array:
            // for each dimension calculate which bin that value falls into in parallel
            // we can then scale by the product of the previous dimensions to get that dimensions impact on the flattened index
            // summing all these together gives the index into the flattened array
            if (totNumBins <= sBound) {
                // small number of buckets (so histogram is relatively small):
                // each task gets it's own copy of the histogram and they're reduced
                hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "%? <= %?".format(bins,sBound));
                forall (gEnt, stride, bin) in zip(gEnts, dimProd.a, bins) with (+ reduce indices) {
                    var e = toSymEntry(gEnt,t);
                    var aMin = min reduce e.a;
                    var aMax = max reduce e.a;
                    var binWidth:real = (aMax - aMin):real / bin:real;
                    forall (v, idx) in zip(e.a, indices) {
                        var vBin = ((v - aMin) / binWidth):int;
                        if v == aMax {vBin = bin-1;}
                        idx += (vBin * stride):int;
                    }
                }
                var gHist: [0..#totNumBins] int;
                
                // count into per-task/per-locale histogram and then reduce as tasks complete
                forall idx in indices with (+ reduce gHist) {
                    gHist[idx] += 1;
                }

                var hist = makeDistArray(totNumBins,int);        
                hist = gHist;
                st.addEntry(rname, createSymEntry(hist));
            }
            else if (totNumBins <= mBound) {
                // medium number of buckets:
                // each locale gets it's own atomic copy of the histogram and these are reduced together
                hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "%? <= %?".format(bins,mBound));
                use PrivateDist;
                var atomicIdx: [PrivateSpace] [0..#numSamples] atomic int;

                forall (gEnt, stride, bin) in zip(gEnts, dimProd.a, bins) {
                    var e = toSymEntry(gEnt,t);
                    var aMin = min reduce e.a;
                    var aMax = max reduce e.a;
                    var binWidth:real = (aMax - aMin):real / bin:real;
                    forall (v, i) in zip(e.a, 0..) {
                        var vBin = ((v - aMin) / binWidth):int;
                        if v == aMax {vBin = bin-1;}
                        atomicIdx[here.id][i].add((vBin * stride):int);
                    }
                }

                var lIdx: [0..#numSamples] int;
                forall i in PrivateSpace with (+ reduce lIdx) {
                    lIdx reduce= atomicIdx[i].read();
                }

                indices = lIdx;
                // allocate per-locale atomic histogram
                var atomicHist: [PrivateSpace] [0..#totNumBins] atomic int;

                // count into per-locale private atomic histogram
                forall idx in indices {
                    atomicHist[here.id][idx].add(1);
                }

                // +reduce across per-locale histograms to get counts
                var lHist: [0..#totNumBins] int;
                forall i in PrivateSpace with (+ reduce lHist) {
                    lHist reduce= atomicHist[i].read();
                }

                var hist = makeDistArray(totNumBins,int);        
                hist = lHist;
                st.addEntry(rname, createSymEntry(hist));
            }
            else {
                // large number of buckets:
                // one global atomic histogram
                hgmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                                "%? > %?".format(bins,mBound));
                use PrivateDist;
                var atomicIdx: [makeDistDom(numSamples)] atomic int;

                forall (gEnt, stride, bin) in zip(gEnts, dimProd.a, bins) {
                    var e = toSymEntry(gEnt,t);
                    var aMin = min reduce e.a;
                    var aMax = max reduce e.a;
                    var binWidth:real = (aMax - aMin):real / bin:real;
                    forall (v, i) in zip(e.a, 0..) {
                        var vBin = ((v - aMin) / binWidth):int;
                        if v == aMax {vBin = bin-1;}
                        atomicIdx[i].add((vBin * stride):int);
                    }
                }

                [(i,ai) in zip(indices, atomicIdx)] i = ai.read();
                // allocate single global atomic histogram
                var atomicHist: [makeDistDom(totNumBins)] atomic int;

                // count into atomic histogram
                forall idx in indices {
                    atomicHist[idx].add(1);
                }

                var hist = makeDistArray(totNumBins,int);
                // copy from atomic histogram to normal histogram
                [(e,ae) in zip(hist, atomicHist)] e = ae.read();
                st.addEntry(rname, createSymEntry(hist));
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
