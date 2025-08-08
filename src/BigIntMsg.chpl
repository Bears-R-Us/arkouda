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

    @arkouda.instantiateAndRegister("big_int_creation")
    proc bigIntCreationMsg(cmd: string, msgArgs: borrowed MessageArgs,
                           st: borrowed SymTab, type array_dtype,
                           param array_nd: int): MsgTuple throws
        where (array_dtype == bigint) {
        param pn = Reflection.getRoutineName();
        var repMsg: string;

        var num_arrays = msgArgs.get("num_arrays").getIntValue();
        var shape = msgArgs.get("shape").toScalarTuple(int, array_nd);
        var arrayNames = msgArgs.get("arrays").getList(num_arrays);
        var max_bits = msgArgs.get("max_bits").getIntValue();

        var bigIntArray = makeDistArray((...shape), bigint);
        for (name, i) in zip(arrayNames, 0..<num_arrays by -1) {
            // We are creating a bigint array from uint arrays
            var entry = st[name]: SymEntry(uint, array_nd);
            ref uintA = entry.a;
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
        repMsg = "created %s".format(st.attrib(retname));
        biLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    @arkouda.instantiateAndRegister("bigint_to_uint_list")
    proc bigintToUintArraysMsg(cmd: string, msgArgs: borrowed MessageArgs,
                               st: borrowed SymTab, type array_dtype,
                               param array_nd: int): MsgTuple throws
        where (array_dtype == bigint) {

        param pn = Reflection.getRoutineName();
        const name = msgArgs.getValueOf("array");
        var entry = st[name]: SymEntry(array_dtype, array_nd);
        var repMsg: string = "";

        var tmp = entry.a;
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
        repMsg = formatJson(retList);

        biLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);

    }

    @arkouda.instantiateAndRegister("get_max_bits")
    proc getMaxBitsMsg(cmd: string, msgArgs: borrowed MessageArgs,
                       st: borrowed SymTab, type array_dtype,
                       param array_nd: int): MsgTuple throws
        where (array_dtype == bigint) {

        param pn = Reflection.getRoutineName();
        const name = msgArgs.getValueOf("array");

        var entry = st[name]: SymEntry(array_dtype, array_nd);
        var repMsg = formatJson(entry.max_bits);
        biLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    @arkouda.instantiateAndRegister("set_max_bits")
    proc setMaxBitsMsg(cmd: string, msgArgs: borrowed MessageArgs,
                       st: borrowed SymTab, type array_dtype,
                       param array_nd: int): MsgTuple throws
        where (array_dtype == bigint) {

        param pn = Reflection.getRoutineName();
        const name = msgArgs.getValueOf("array");
        var max_bits = msgArgs.get("max_bits").getIntValue();
        var entry = st[name]: SymEntry(array_dtype, array_nd);
        var repMsg: string = "";
        ref ea = entry.a;
        if max_bits != -1 {
            // modBy should always be non-zero since we start at 1 and left shift
            var max_size = 1:bigint;
            max_size <<= max_bits;
            max_size -= 1;
            forall ei in ea with (var local_max_size = max_size) {
                ei &= local_max_size;
            }
        }
        entry.max_bits = max_bits;
        repMsg = "Sucessfully set max_bits";

        biLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);

    }

}
