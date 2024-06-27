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
    use IOUtils;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const biLogger = new Logger(logLevel, logChannel);

    proc bigIntCreationMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;

        var num_arrays = msgArgs.get("num_arrays").getIntValue();
        var len = msgArgs.get("len").getIntValue();
        var arrayNames = msgArgs.get("arrays").getList(num_arrays);
        var max_bits = msgArgs.get("max_bits").getIntValue();

        var bigIntArray = makeDistArray(len, bigint);
        for (name, i) in zip(arrayNames, 0..<num_arrays by -1) {
            ref uintA = toSymEntry(getGenericTypedArrayEntry(name, st), uint).a;
            forall (uA, bA) in zip(uintA, bigIntArray) with (var bigUA: bigint) {
              bigUA = uA;
              bigUA <<= (64*i);
              bA += bigUA;
            }
        }

        if max_bits != -1 {
            // modBy should always be non-zero since we start at 1 and left shift
            var max_size = 1:bigint;
            max_size <<= max_bits;
            max_size -= 1;
            forall bA in bigIntArray with (var local_max_size = max_size) {
              bA &= local_max_size;
            }
        }

        var retname = st.nextName();
        st.addEntry(retname, createSymEntry(bigIntArray, max_bits));
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
                // default to false because we want to do first loop whether or not tmp is all_zero
                var all_zero = false;
                var low = makeDistArray(tmp.domain, uint);
                const ushift = 64:uint;
                while !all_zero {
                  low = tmp:uint;
                  var retname = st.nextName();
                  st.addEntry(retname, createSymEntry(low));
                  retList.pushBack("created %s".format(st.attrib(retname)));

                  all_zero = true;
                  forall t in tmp with (&& reduce all_zero) {
                    t >>= ushift;
                    all_zero &&= (t == 0 || t == -1);
                  }
                }
                var repMsg = formatJson(retList);
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

    proc getMaxBitsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        const name = msgArgs.getValueOf("array");
        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        select gEnt.dtype {
            when DType.BigInt {
                var repMsg = formatJson(toSymEntry(gEnt, bigint).max_bits);
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

    proc setMaxBitsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        const name = msgArgs.getValueOf("array");
        var max_bits = msgArgs.get("max_bits").getIntValue();
        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        select gEnt.dtype {
            when DType.BigInt {
                var e = toSymEntry(gEnt, bigint);
                ref ea = e.a;

                if max_bits != -1 {
                    // modBy should always be non-zero since we start at 1 and left shift
                    var max_size = 1:bigint;
                    max_size <<= max_bits;
                    max_size -= 1;
                    forall ei in ea with (var local_max_size = max_size) {
                      ei &= local_max_size;
                    }
                }
                e.max_bits = max_bits;
                var repMsg = "Sucessfully set max_bits";
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
    registerFunction("bigint_to_uint_list", bigintToUintArraysMsg, getModuleName());
    registerFunction("get_max_bits", getMaxBitsMsg, getModuleName());
    registerFunction("set_max_bits", setMaxBitsMsg, getModuleName());
}
