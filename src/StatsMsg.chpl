module StatsMsg {
    use ServerConfig;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use Stats;

    use Map;

    private config const logLevel = ServerConfig.logLevel;
    const sLogger = new Logger(logLevel);

    proc meanMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;

        var x: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("x"), st);

        select (x.dtype) {
            when (DType.Int64) {
                repMsg = "float64 %.17r".format(mean(toSymEntry(x,int).a));
            }
            when (DType.Float64) {
                repMsg = "float64 %.17r".format(mean(toSymEntry(x,real).a));
            }
            when (DType.Bool) {
                repMsg = "float64 %.17r".format(mean(toSymEntry(x,bool).a));
            }
            when (DType.UInt64) {
                repMsg = "float64 %.17r".format(mean(toSymEntry(x,uint).a));
            }
            otherwise {
                var errorMsg = unrecognizedTypeError(pn, dtype2str(x.dtype));
                sLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
        sLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc varMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;

        var ddof = msgArgs.get("ddof").getIntValue();
        var x: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("x"), st);

        select (x.dtype) {
            when (DType.Int64) {
                repMsg = "float64 %.17r".format(variance(toSymEntry(x,int).a, ddof));
            }
            when (DType.Float64) {
                repMsg = "float64 %.17r".format(variance(toSymEntry(x,real).a, ddof));
            }
            when (DType.Bool) {
                repMsg = "float64 %.17r".format(variance(toSymEntry(x,bool).a, ddof));
            }
            when (DType.UInt64) {
                repMsg = "float64 %.17r".format(variance(toSymEntry(x,uint).a, ddof));
            }
            otherwise {
                var errorMsg = unrecognizedTypeError(pn, dtype2str(x.dtype));
                sLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
        sLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc stdMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;

        var ddof = msgArgs.get("ddof").getIntValue();
        var x: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("x"), st);

        select (x.dtype) {
            when (DType.Int64) {
                repMsg = "float64 %.17r".format(std(toSymEntry(x,int).a, ddof));
            }
            when (DType.Float64) {
                repMsg = "float64 %.17r".format(std(toSymEntry(x,real).a, ddof));
            }
            when (DType.Bool) {
                repMsg = "float64 %.17r".format(std(toSymEntry(x,bool).a, ddof));
            }
            when (DType.UInt64) {
                repMsg = "float64 %.17r".format(std(toSymEntry(x,uint).a, ddof));
            }
            otherwise {
                var errorMsg = unrecognizedTypeError(pn, dtype2str(x.dtype));
                sLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
        sLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
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
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.Int64, DType.Float64) {
                var eX = toSymEntry(x,int);
                var eY = toSymEntry(y,real);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.Int64, DType.Bool) {
                var eX = toSymEntry(x,int);
                var eY = toSymEntry(y,bool);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.Int64, DType.UInt64) {
                var eX = toSymEntry(x,int);
                var eY = toSymEntry(y,uint);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.Float64, DType.Float64) {
                var eX = toSymEntry(x,real);
                var eY = toSymEntry(y,real);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.Float64, DType.Int64) {
                var eX = toSymEntry(x,real);
                var eY = toSymEntry(y,int);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.Float64, DType.Bool) {
                var eX = toSymEntry(x,real);
                var eY = toSymEntry(y,bool);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.Float64, DType.UInt64) {
                var eX = toSymEntry(x,real);
                var eY = toSymEntry(y,uint);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.Bool, DType.Bool) {
                var eX = toSymEntry(x,bool);
                var eY = toSymEntry(y,bool);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.Bool, DType.Float64) {
                var eX = toSymEntry(x,bool);
                var eY = toSymEntry(y,real);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.Bool, DType.Int64) {
                var eX = toSymEntry(x,bool);
                var eY = toSymEntry(y,int);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.Bool, DType.UInt64) {
                var eX = toSymEntry(x,bool);
                var eY = toSymEntry(y,uint);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.UInt64, DType.UInt64) {
                var eX = toSymEntry(x,uint);
                var eY = toSymEntry(y,uint);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.UInt64, DType.Int64) {
                var eX = toSymEntry(x,uint);
                var eY = toSymEntry(y,int);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.UInt64, DType.Float64) {
                var eX = toSymEntry(x,uint);
                var eY = toSymEntry(y,real);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            when (DType.UInt64, DType.Bool) {
                var eX = toSymEntry(x,uint);
                var eY = toSymEntry(y,bool);
                repMsg = "float64 %.17r".format(cov(eX.a, eY.a));
            }
            otherwise {
                var errorMsg = unrecognizedTypeError(pn, "(%s,%s)".format(dtype2str(x.dtype),dtype2str(y.dtype)));
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

        var repMsg: string = "float64 %.17r".format(corrHelper(x, y));
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
                var errorMsg = unrecognizedTypeError(pn, "(%s,%s)".format(dtype2str(x.dtype),dtype2str(y.dtype)));
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
            st.addEntry(retname, new shared SymEntry(corrPdarray));
            corrDict.add(col, "created %s".format(st.attrib(retname)));
        }
        var repMsg: string = "%jt".format(corrDict);

        sLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }


    use CommandMap;
    registerFunction("mean", meanMsg, getModuleName());
    registerFunction("var", varMsg, getModuleName());
    registerFunction("std", stdMsg, getModuleName());
    registerFunction("cov", covMsg, getModuleName());
    registerFunction("corr",  corrMsg, getModuleName());
    registerFunction("corrMatrix",  corrMatrixMsg, getModuleName());
}
