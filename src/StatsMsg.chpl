module StatsMsg {
    use ServerConfig;

    use AryUtil;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use Stats;

    use Map;
    use ArkoudaIOCompat;
    use ArkoudaAryUtilCompat;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const sLogger = new Logger(logLevel, logChannel);

    @chpldoc.nodoc
    const computations = {"mean", "var", "std"};

    /*
        Compute the mean, variance, or standard deviation of an array along one or more axes.
    */
    @arkouda.registerND
    proc statsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        const x = msgArgs.getValueOf("x"),
              comp = msgArgs.getValueOf("comp"),
              nAxes = msgArgs.get("nAxes").getIntValue(),
              axesRaw = msgArgs.get("axis").getListAs(int, nAxes),
              ddof = msgArgs.get("ddof").getRealValue(), // "correction" for std and variance
              skipNan = msgArgs.get("skipNan").getBoolValue(),
              rname = st.nextName();

        if !computations.contains(comp) {
            var errorMsg = "Unrecognized stats computation: %s".doFormat(comp);
            sLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg,MsgType.ERROR);
        }

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(x, st);

        proc computeStat(type tIn, type tOut): MsgTuple throws {
            const eIn = toSymEntry(gEnt, tIn, nd);

            if nd == 1 || nAxes == 0 {
                var s: tOut;
                select comp {
                    when "mean" do s = if skipNan then meanSkipNan(eIn.a) else mean(eIn.a);
                    when "var" do s = if skipNan then varianceSkipNan(eIn.a, ddof) else variance(eIn.a, ddof);
                    when "std" do s = if skipNan then stdSkipNan(eIn.a, ddof) else std(eIn.a, ddof);
                    otherwise halt("unreachable");
                }

                const scalarValue = "float64 %.17r".doFormat(s);
                sLogger.debug(getModuleName(),pn,getLineNumber(),scalarValue);
                return new MsgTuple(scalarValue, MsgType.NORMAL);
            } else {
                const (valid, axes) = validateNegativeAxes(axesRaw, nd);
                if !valid {
                    const errMsg = "Unable to compute 'std' on array with shape %? using axes %?".doFormat(eIn.tupShape, axesRaw);
                    sLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
                    return new MsgTuple(errMsg,MsgType.ERROR);
                } else {
                    const outShape = reducedShape(eIn.tupShape, axes);
                    var eOut = st.addEntry(rname, outShape, tOut);

                    forall sliceIdx in domOffAxis(eIn.a.domain, axes) {
                        const sliceDom = domOnAxis(eIn.a.domain, sliceIdx, axes);
                        var s: tOut;
                        select comp {
                            when "mean" do s = if skipNan
                                then meanSkipNan(eIn.a, sliceDom)
                                else meanOver(eIn.a, sliceDom);
                            when "var" do s = if skipNan
                                then varianceSkipNan(eIn.a, sliceDom, ddof)
                                else varianceOver(eIn.a, sliceDom, ddof);
                            when "std" do s = if skipNan
                                then stdSkipNan(eIn.a, sliceDom, ddof)
                                else stdOver(eIn.a, sliceDom, ddof);
                            otherwise halt("unreachable");
                        }
                        eOut.a[sliceIdx] = s;
                    }

                    const repMsg = "created " + st.attrib(rname);
                    sLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
                    return new MsgTuple(repMsg, MsgType.NORMAL);
                }
            }
        }

        select gEnt.dtype {
            when DType.Int64 do return computeStat(int, real);
            when DType.UInt64 do return computeStat(uint, real);
            when DType.Float64 do return computeStat(real, real);
            when DType.Bool do return computeStat(bool, real);
            otherwise {
                var errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
                sLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
        }

    }

    proc covMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;

        var x: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("x"), st);
        var y: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("y"), st);

        select (x.dtype, y.dtype) {
            when (DType.Int64, DType.Int64) {
                var eX = toSymEntry(x,int);
                var eY = toSymEntry(y,int);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.Int64, DType.Float64) {
                var eX = toSymEntry(x,int);
                var eY = toSymEntry(y,real);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.Int64, DType.Bool) {
                var eX = toSymEntry(x,int);
                var eY = toSymEntry(y,bool);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.Int64, DType.UInt64) {
                var eX = toSymEntry(x,int);
                var eY = toSymEntry(y,uint);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.Float64, DType.Float64) {
                var eX = toSymEntry(x,real);
                var eY = toSymEntry(y,real);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.Float64, DType.Int64) {
                var eX = toSymEntry(x,real);
                var eY = toSymEntry(y,int);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.Float64, DType.Bool) {
                var eX = toSymEntry(x,real);
                var eY = toSymEntry(y,bool);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.Float64, DType.UInt64) {
                var eX = toSymEntry(x,real);
                var eY = toSymEntry(y,uint);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.Bool, DType.Bool) {
                var eX = toSymEntry(x,bool);
                var eY = toSymEntry(y,bool);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.Bool, DType.Float64) {
                var eX = toSymEntry(x,bool);
                var eY = toSymEntry(y,real);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.Bool, DType.Int64) {
                var eX = toSymEntry(x,bool);
                var eY = toSymEntry(y,int);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.Bool, DType.UInt64) {
                var eX = toSymEntry(x,bool);
                var eY = toSymEntry(y,uint);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.UInt64, DType.UInt64) {
                var eX = toSymEntry(x,uint);
                var eY = toSymEntry(y,uint);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.UInt64, DType.Int64) {
                var eX = toSymEntry(x,uint);
                var eY = toSymEntry(y,int);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.UInt64, DType.Float64) {
                var eX = toSymEntry(x,uint);
                var eY = toSymEntry(y,real);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            when (DType.UInt64, DType.Bool) {
                var eX = toSymEntry(x,uint);
                var eY = toSymEntry(y,bool);
                repMsg = "float64 %.17r".doFormat(cov(eX.a, eY.a));
            }
            otherwise {
                var errorMsg = unrecognizedTypeError(pn, "(%s,%s)".doFormat(dtype2str(x.dtype),dtype2str(y.dtype)));
                sLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
        sLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc corrMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        var x: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("x"), st);
        var y: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("y"), st);

        var repMsg: string = "float64 %.17r".doFormat(corrHelper(x, y));
        sLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc corrHelper(x: borrowed GenSymEntry, y: borrowed GenSymEntry): real throws {
        param pn = Reflection.getRoutineName();
        select (x.dtype, y.dtype) {
            when (DType.Int64, DType.Int64) {
                return corr(toSymEntry(x,int).a, toSymEntry(y,int).a);
            }
            when (DType.Int64, DType.Float64) {
                return corr(toSymEntry(x,int).a, toSymEntry(y,real).a);
            }
            when (DType.Int64, DType.Bool) {
                return corr(toSymEntry(x,int).a, toSymEntry(y,bool).a);
            }
            when (DType.Int64, DType.UInt64) {
                return corr(toSymEntry(x,int).a, toSymEntry(y,uint).a);
            }
            when (DType.Float64, DType.Float64) {
                return corr(toSymEntry(x,real).a, toSymEntry(y,real).a);
            }
            when (DType.Float64, DType.Int64) {
                return corr(toSymEntry(x,real).a, toSymEntry(y,int).a);
            }
            when (DType.Float64, DType.Bool) {
                return corr(toSymEntry(x,real).a, toSymEntry(y,bool).a);
            }
            when (DType.Float64, DType.UInt64) {
                return corr(toSymEntry(x,real).a, toSymEntry(y,uint).a);
            }
            when (DType.Bool, DType.Bool) {
                return corr(toSymEntry(x,bool).a, toSymEntry(y,bool).a);
            }
            when (DType.Bool, DType.Float64) {
                return corr(toSymEntry(x,bool).a, toSymEntry(y,real).a);
            }
            when (DType.Bool, DType.Int64) {
                return corr(toSymEntry(x,bool).a, toSymEntry(y,int).a);
            }
            when (DType.Bool, DType.UInt64) {
                return corr(toSymEntry(x,bool).a, toSymEntry(y,uint).a);
            }
            when (DType.UInt64, DType.UInt64) {
                return corr(toSymEntry(x,uint).a, toSymEntry(y,uint).a);
            }
            when (DType.UInt64, DType.Int64) {
                return corr(toSymEntry(x,uint).a, toSymEntry(y,int).a);
            }
            when (DType.UInt64, DType.Float64) {
                return corr(toSymEntry(x,uint).a, toSymEntry(y,real).a);
            }
            when (DType.UInt64, DType.Bool) {
                return corr(toSymEntry(x,uint).a, toSymEntry(y,bool).a);
            }
            otherwise {
                var errorMsg = unrecognizedTypeError(pn, "(%s,%s)".doFormat(dtype2str(x.dtype),dtype2str(y.dtype)));
                sLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                throw new owned IllegalArgumentError(errorMsg);
            }
        }
    }

    proc corrMatrixMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        var size = msgArgs.get("size").getIntValue();
        var columns = msgArgs.get("columns").getList(size);
        var dataNames = msgArgs.get("data_names").getList(size);

        var corrDict = new map(keyType=string, valType=string);
        for (col, d1, i) in zip(columns, dataNames, 0..) {
            var corrPdarray = makeDistArray(size, real);
            var d1gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(d1, st);
            forall (d2, j) in zip(dataNames, 0..) {
                var d2gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(d2, st);
                corrPdarray[j] = corrHelper(d1gEnt, d2gEnt);
            }
            var retname = st.nextName();
            st.addEntry(retname, createSymEntry(corrPdarray));
            corrDict.add(col, "created %s".doFormat(st.attrib(retname)));
        }
        var repMsg: string = formatJson(corrDict);

        sLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    // https://data-apis.org/array-api/latest/API_specification/generated/array_api.cumulative_sum.html#array_api.cumulative_sum
    @arkouda.registerND
    proc cumSumMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        const axis = msgArgs.get("axis").getPositiveIntValue(nd),
              x = msgArgs.getValueOf("x"),
              includeInitial = msgArgs.get("include_initial").getBoolValue(),
              rname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(x, st);

        proc computeCumSum(type t): MsgTuple throws
            where nd == 1
        {
            const eIn = toSymEntry(gEnt, t);
            const cs = + scan eIn.a;

            if includeInitial {
                var eOut = st.addEntry(rname, cs.size + 1, t);
                eOut.a[0] = 0:t;
                eOut.a[1..] = cs;
            } else {
                st.addEntry(rname, createSymEntry(cs));
            }

            const repMsg = "created " + st.attrib(rname);
            sLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        proc computeCumSum(type t): MsgTuple throws
            where nd > 1
        {
            const eIn = toSymEntry(gEnt, t, nd);

            var outShape = eIn.tupShape;
            if includeInitial then outShape[axis] += 1;
            var eOut = st.addEntry(rname, outShape, t);

            const DD = domOffAxis(eIn.a.domain, axis);
            for idx in DD {
                // TODO: avoid making a copy of the slice here
                const sliceDom = domOnAxis(eIn.a.domain, idx, axis),
                      slice = removeDegenRanks(eIn.a[sliceDom], 1);
                const cs = + scan slice;

                if includeInitial {
                    // put the additive identity in the first element
                    eOut.a[idx] = 0:t;

                    forall i in sliceDom {
                        var iShift = i;
                        iShift[axis] += 1;
                        eOut.a[iShift] = cs[i[axis]];
                    }
                } else {
                    forall i in sliceDom do eOut.a[i] = cs[i[axis]];
                }
            }

            const repMsg = "created " + st.attrib(rname);
            sLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        select gEnt.dtype {
            when DType.Int64 do return computeCumSum(int);
            when DType.UInt64 do return computeCumSum(uint);
            when DType.Float64 do return computeCumSum(real);
            otherwise {
                const errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
                sLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
        }
    }

    use CommandMap;
    registerFunction("cov", covMsg, getModuleName());
    registerFunction("corr",  corrMsg, getModuleName());
    registerFunction("corrMatrix",  corrMatrixMsg, getModuleName());
}
