module BigIntMsg {
    use ServerConfig;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use BigInteger;
    use List;


    private config const logLevel = ServerConfig.logLevel;
    const biLogger = new Logger(logLevel);

    proc bigIntCreationMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;

        var num_arrays = msgArgs.get("num_arrays").getIntValue();
        var len = msgArgs.get("len").getIntValue();
        var arrayNames = msgArgs.get("arrays").getList(num_arrays);
        var max_bits = msgArgs.get("max_bits").getIntValue();

        var bigIntArray = makeDistArray(len, bigint);
        // block size is 2**64
        var block_size = 1:bigint;
        block_size <<= 64;
        forall (name, i) in zip(arrayNames, 0..<num_arrays by -1) with (+ reduce bigIntArray) {
            var tmp = toSymEntry(getGenericTypedArrayEntry(name, st), uint).a:bigint;
            tmp <<= (64*i);
            bigIntArray += tmp;
        }
        var retname = st.nextName();

        if max_bits != -1 {
            // modBy should always be non-zero since we start at 1 and left shift
            var modBy = 1:bigint;
            modBy <<= max_bits;
            bigIntArray.mod(bigIntArray, modBy);
        }

        st.addEntry(retname, new shared SymEntry(bigIntArray, max_bits));
        var syment = toSymEntry(getGenericTypedArrayEntry(retname, st), bigint);
        repMsg = "created %s".format(st.attrib(retname));
        biLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc bigintToUintArraysMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        const name = msgArgs.getValueOf("array");
        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        select gEnt.dtype {
            when DType.BigInt {
                var e = toSymEntry(gEnt, bigint);
                var tmp = e.a;
                // take in a bigint sym entry and return list of uint64 symentries
                var retList: list(string);
                var block_size = 1:bigint;
                block_size <<= 64;
                while || reduce (tmp!=0) {
                    var low = tmp % block_size;
                    var retname = st.nextName();

                    st.addEntry(retname, new shared SymEntry(low:uint));
                    retList.append("created %s".format(st.attrib(retname)));
                    tmp /= block_size;
                }
                var repMsg = "%jt".format(retList);
                biLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            otherwise {
                var errorMsg = notImplementedError(pn, "("+dtype2str(gEnt.dtype)+")");
                biLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    use CommandMap;
    registerFunction("big_int_creation", bigIntCreationMsg, getModuleName());
    registerFunction("break_into_arrays", bigintToUintArraysMsg, getModuleName());
}