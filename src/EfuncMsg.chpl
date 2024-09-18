
module EfuncMsg
{
    use ServerConfig;

    use Time;
    use Math;
    use BitOps;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    private use SipHash;
    use UniqueMsg;
    use AryUtil;

    use CommAggregation;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const eLogger = new Logger(logLevel, logChannel);

    extern proc fmod(x: real, y: real): real;


    /* These ops are functions which take an array and produce an array.

       **Dev Note:** Do scans fit here also? I think so... vector = scanop(vector)
       parse and respond to efunc "elemental function" message
       vector = efunc(vector) 

      :arg reqMsg: request containing (cmd,efunc,name)
      :type reqMsg: string 

      :arg st: SymTab to act on
      :type st: borrowed SymTab 

      :returns: (MsgTuple)
      :throws: `UndefinedSymbolError(name)`
      */

    @arkouda.registerND
    proc efuncMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message; attributes of returned array(s) will be appended to this string
        var name = msgArgs.getValueOf("array");
        var efunc = msgArgs.getValueOf("func");
        var rname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                           "cmd: %s efunc: %s pdarray: %s".format(cmd,efunc,st.attrib(name)));

        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int, nd);
                ref ea = e.a;
                select efunc
                {
                    when "abs" {
                        st.addEntry(rname, new shared SymEntry(abs(ea)));
                    }
                    when "log" {
                        st.addEntry(rname, new shared SymEntry(log(ea)));
                    }
                    when "round" {
                        st.addEntry(rname, new shared SymEntry(ea));
                    }
                    when "sgn" {
                        st.addEntry(rname, new shared SymEntry(sgn(ea)));
                    }
                    when "exp" {
                        st.addEntry(rname, new shared SymEntry(exp(ea)));
                    }
                    when "square" {
                        st.addEntry(rname, new shared SymEntry(square(ea)));
                    }
                    when "cumsum" {
                        if nd == 1 {
                            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                            overMemLimit(numBytes(int) * e.size);
                            st.addEntry(rname, new shared SymEntry(+ scan e.a));
                        } else {
                            var errorMsg = notImplementedError(pn,efunc,gEnt.dtype,nd);
                            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                            return new MsgTuple(errorMsg, MsgType.ERROR);
                        }
                    }
                    when "cumprod" {
                        if nd == 1 {
                            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                            overMemLimit(numBytes(int) * e.size);
                            st.addEntry(rname, new shared SymEntry(* scan e.a));
                        } else {
                            var errorMsg = notImplementedError(pn,efunc,gEnt.dtype,nd);
                            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                            return new MsgTuple(errorMsg, MsgType.ERROR);
                        }
                    }
                    when "sin" {
                        st.addEntry(rname, new shared SymEntry(sin(ea)));
                    }
                    when "cos" {
                        st.addEntry(rname, new shared SymEntry(cos(ea)));
                    }
                    when "tan" {
                        st.addEntry(rname, new shared SymEntry(tan(ea)));
                    }
                    when "arcsin" {
                        st.addEntry(rname, new shared SymEntry(asin(ea)));
                    }
                    when "arccos" {
                        st.addEntry(rname, new shared SymEntry(acos(ea)));
                    }
                    when "arctan" {
                        st.addEntry(rname, new shared SymEntry(atan(ea)));
                    }
                    when "sinh" {
                        st.addEntry(rname, new shared SymEntry(sinh(ea)));
                    }
                    when "cosh" {
                        st.addEntry(rname, new shared SymEntry(cosh(ea)));
                    }
                    when "tanh" {
                        st.addEntry(rname, new shared SymEntry(tanh(ea)));
                    }
                    when "arcsinh" {
                        st.addEntry(rname, new shared SymEntry(asinh(ea)));
                    }
                    when "arccosh" {
                        st.addEntry(rname, new shared SymEntry(acosh(ea)));
                    }
                    when "arctanh" {
                        st.addEntry(rname, new shared SymEntry(atanh(ea)));
                    }
                    when "hash64" {
                        overMemLimit(numBytes(int) * e.size);
                        var a = st.addEntry(rname, e.tupShape, uint);
                        forall (ai, x) in zip(a.a, e.a) {
                            ai = sipHash64(x): uint;
                        }
                    }
                    when "hash128" {
                        overMemLimit(numBytes(int) * e.size * 2);
                        var rname2 = st.nextName();
                        var a1 = st.addEntry(rname2, e.tupShape, uint);
                        var a2 = st.addEntry(rname, e.tupShape, uint);
                        forall (a1i, a2i, x) in zip(a1.a, a2.a, e.a) {
                            (a1i, a2i) = sipHash128(x): (uint, uint);
                        }
                        // Put first array's attrib in repMsg and let common
                        // code append second array's attrib
                        repMsg += "created " + st.attrib(rname2) + "+";
                    }
                    when "popcount" {
                        st.addEntry(rname, new shared SymEntry(popCount(ea)));
                    }
                    when "parity" {
                        st.addEntry(rname, new shared SymEntry(parity(ea)));
                    }
                    when "clz" {
                        st.addEntry(rname, new shared SymEntry(clz(ea)));
                    }
                    when "ctz" {
                        st.addEntry(rname, new shared SymEntry(ctz(ea)));
                    }
                    when "not" {
                        st.addEntry(rname, new shared SymEntry(!e.a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,gEnt.dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Float64) {
                var e = toSymEntry(gEnt,real, nd);
                ref ea = e.a;
                select efunc
                {
                    when "abs" {
                        st.addEntry(rname, new shared SymEntry(abs(ea)));
                    }
                    when "ceil" {
                        st.addEntry(rname, new shared SymEntry(ceil(ea)));
                    }
                    when "floor" {
                        st.addEntry(rname, new shared SymEntry(floor(ea)));
                    }
                    when "round" {
                        st.addEntry(rname, new shared SymEntry(round(ea)));
                    }
                    when "trunc" {
                        st.addEntry(rname, new shared SymEntry(trunc(ea)));
                    }
                    when "sgn" {
                        st.addEntry(rname, new shared SymEntry(sgn(ea)));
                    }
                    when "isfinite" {
                        st.addEntry(rname, new shared SymEntry(isFinite(ea)));
                    }
                    when "isinf" {
                        st.addEntry(rname, new shared SymEntry(isInf(ea)));
                    }
                    when "isnan" {
                        st.addEntry(rname, new shared SymEntry(isNan(ea)));
                    }
                    when "log" {
                        st.addEntry(rname, new shared SymEntry(log(ea)));
                    }
                    when "log1p" {
                        st.addEntry(rname, new shared SymEntry(log1p(ea)));
                    }
                    when "log2" {
                        st.addEntry(rname, new shared SymEntry(log2(ea)));
                    }
                    when "log10" {
                        st.addEntry(rname, new shared SymEntry(log10(ea)));
                    }
                    when "exp" {
                        st.addEntry(rname, new shared SymEntry(exp(ea)));
                    }
                    when "expm1" {
                        st.addEntry(rname, new shared SymEntry(expm1(ea)));
                    }
                    when "square" {
                        st.addEntry(rname, new shared SymEntry(square(ea)));
                    }
                    when "cumsum" {
                        if nd == 1 {
                            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                            overMemLimit(numBytes(real) * e.size);
                            st.addEntry(rname, new shared SymEntry(+ scan e.a));
                        } else {
                            var errorMsg = notImplementedError(pn,efunc,gEnt.dtype,nd);
                            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                            return new MsgTuple(errorMsg, MsgType.ERROR);
                        }
                    }
                    when "cumprod" {
                        if nd == 1 {
                            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                            overMemLimit(numBytes(real) * e.size);
                            st.addEntry(rname, new shared SymEntry(* scan e.a));
                        } else {
                            var errorMsg = notImplementedError(pn,efunc,gEnt.dtype,nd);
                            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                            return new MsgTuple(errorMsg, MsgType.ERROR);
                        }
                    }
                    when "sin" {
                        st.addEntry(rname, new shared SymEntry(sin(ea)));
                    }
                    when "cos" {
                        st.addEntry(rname, new shared SymEntry(cos(ea)));
                    }
                    when "tan" {
                        st.addEntry(rname, new shared SymEntry(tan(ea)));
                    }
                    when "arcsin" {
                        st.addEntry(rname, new shared SymEntry(asin(ea)));
                    }
                    when "arccos" {
                        st.addEntry(rname, new shared SymEntry(acos(ea)));
                    }
                    when "arctan" {
                        st.addEntry(rname, new shared SymEntry(atan(ea)));
                    }
                    when "sinh" {
                        st.addEntry(rname, new shared SymEntry(sinh(ea)));
                    }
                    when "cosh" {
                        st.addEntry(rname, new shared SymEntry(cosh(ea)));
                    }
                    when "tanh" {
                        st.addEntry(rname, new shared SymEntry(tanh(ea)));
                    }
                    when "arcsinh" {
                        st.addEntry(rname, new shared SymEntry(asinh(ea)));
                    }
                    when "arccosh" {
                        st.addEntry(rname, new shared SymEntry(acosh(ea)));
                    }
                    when "arctanh" {
                        st.addEntry(rname, new shared SymEntry(atanh(ea)));
                    }
                    when "hash64" {
                        overMemLimit(numBytes(real) * e.size);
                        var a = st.addEntry(rname, e.tupShape, uint);
                        forall (ai, x) in zip(a.a, e.a) {
                            ai = sipHash64(x): uint;
                        }
                    }
                    when "hash128" {
                        overMemLimit(numBytes(real) * e.size * 2);
                        var rname2 = st.nextName();
                        var a1 = st.addEntry(rname2, e.tupShape, uint);
                        var a2 = st.addEntry(rname, e.tupShape, uint);
                        forall (a1i, a2i, x) in zip(a1.a, a2.a, e.a) {
                            (a1i, a2i) = sipHash128(x): (uint, uint);
                        }
                        // Put first array's attrib in repMsg and let common
                        // code append second array's attrib
                        repMsg += "created " + st.attrib(rname2) + "+";
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,gEnt.dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Bool) {
                var e = toSymEntry(gEnt,bool, nd);
                select efunc
                {
                    when "cumsum" {
                        if nd == 1 {
                            var ia = makeDistArray(e.a.domain, int); // make a copy of bools as ints blah!
                            ia = e.a:int;
                            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                            overMemLimit(numBytes(int) * ia.size);
                            st.addEntry(rname, new shared SymEntry(+ scan ia));
                        } else {
                            var errorMsg = notImplementedError(pn,efunc,gEnt.dtype,nd);
                            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                            return new MsgTuple(errorMsg, MsgType.ERROR);
                        }
                    }
                    when "cumprod" {
                        if nd == 1 {
                            var ia = makeDistArray(e.a.domain, int); // make a copy of bools as ints blah!
                            ia = e.a:int;
                            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                            overMemLimit(numBytes(int) * ia.size);
                            st.addEntry(rname, new shared SymEntry(* scan ia));
                        } else {
                            var errorMsg = notImplementedError(pn,efunc,gEnt.dtype,nd);
                            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                            return new MsgTuple(errorMsg, MsgType.ERROR);
                        }
                    }
                    when "not" {
                        st.addEntry(rname, new shared SymEntry(!e.a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,gEnt.dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.UInt64) {
                var e = toSymEntry(gEnt,uint, nd);
                ref ea = e.a;
                select efunc
                {
                    when "popcount" {
                        st.addEntry(rname, new shared SymEntry(popCount(ea)));
                    }
                    when "clz" {
                        st.addEntry(rname, new shared SymEntry(clz(ea)));
                    }
                    when "ctz" {
                        st.addEntry(rname, new shared SymEntry(ctz(ea)));
                    }
                    when "round" {
                        st.addEntry(rname, new shared SymEntry(ea));
                    }
                    when "sgn" {
                        st.addEntry(rname, new shared SymEntry(sgn(ea)));
                    }
                    when "cumsum" {
                        if nd == 1 {
                            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                            overMemLimit(numBytes(uint) * e.size);
                            st.addEntry(rname, new shared SymEntry(+ scan e.a));
                        } else {
                            var errorMsg = notImplementedError(pn,efunc,gEnt.dtype,nd);
                            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                            return new MsgTuple(errorMsg, MsgType.ERROR);
                        }
                    }
                    when "cumprod" {
                        if nd == 1 {
                            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                            overMemLimit(numBytes(uint) * e.size);
                            st.addEntry(rname, new shared SymEntry(* scan e.a));
                        } else {
                            var errorMsg = notImplementedError(pn,efunc,gEnt.dtype,nd);
                            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                            return new MsgTuple(errorMsg, MsgType.ERROR);
                        }
                    }
                    when "sin" {
                        st.addEntry(rname, new shared SymEntry(sin(ea)));
                    }
                    when "cos" {
                        st.addEntry(rname, new shared SymEntry(cos(ea)));
                    }
                    when "tan" {
                        st.addEntry(rname, new shared SymEntry(tan(ea)));
                    }
                    when "arcsin" {
                        st.addEntry(rname, new shared SymEntry(asin(ea)));
                    }
                    when "arccos" {
                        st.addEntry(rname, new shared SymEntry(acos(ea)));
                    }
                    when "arctan" {
                        st.addEntry(rname, new shared SymEntry(atan(ea)));
                    }
                    when "sinh" {
                        st.addEntry(rname, new shared SymEntry(sinh(ea)));
                    }
                    when "cosh" {
                        st.addEntry(rname, new shared SymEntry(cosh(ea)));
                    }
                    when "tanh" {
                        st.addEntry(rname, new shared SymEntry(tanh(ea)));
                    }
                    when "arcsinh" {
                        st.addEntry(rname, new shared SymEntry(asinh(ea)));
                    }
                    when "arccosh" {
                        st.addEntry(rname, new shared SymEntry(acosh(ea)));
                    }
                    when "arctanh" {
                        st.addEntry(rname, new shared SymEntry(atanh(ea)));
                    }
                    when "parity" {
                        st.addEntry(rname, new shared SymEntry(parity(ea)));
                    }
                    when "hash64" {
                        overMemLimit(numBytes(uint) * e.size);
                        var a = st.addEntry(rname, e.tupShape, uint);
                        forall (ai, x) in zip(a.a, e.a) {
                            ai = sipHash64(x): uint;
                        }
                    }
                    when "hash128" {
                        overMemLimit(numBytes(uint) * e.size * 2);
                        var rname2 = st.nextName();
                        var a1 = st.addEntry(rname2, e.tupShape, uint);
                        var a2 = st.addEntry(rname, e.tupShape, uint);
                        forall (a1i, a2i, x) in zip(a1.a, a2.a, e.a) {
                            (a1i, a2i) = sipHash128(x): (uint, uint);
                        }
                        // Put first array's attrib in repMsg and let common
                        // code append second array's attrib
                        repMsg += "created " + st.attrib(rname2) + "+";
                    }
                    when "log" {
                        st.addEntry(rname, new shared SymEntry(log(ea)));
                    }
                    when "exp" {
                        st.addEntry(rname, new shared SymEntry(exp(ea)));
                    }
                    when "square" {
                        st.addEntry(rname, new shared SymEntry(square(ea)));
                    }
                    when "not" {
                        st.addEntry(rname, new shared SymEntry(!e.a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,gEnt.dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            otherwise {
                var errorMsg = unrecognizedTypeError(pn, dtype2str(gEnt.dtype));
                eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
        // Append instead of assign here, to allow for 2 return arrays from hash128
        repMsg += "created " + st.attrib(rname);
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    private proc square(x) do return x * x;
    private proc log1p(x: real):real do return log(1.0 + x);
    private proc expm1(x: real):real do return exp(x) - 1.0;

    /*
        These are functions which take two arrays and produce an array.
        vector = efunc(vector, vector)
    */
    @arkouda.registerND(cmd_prefix="efunc2Arg")
    proc efunc2Msg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;
        var rname = st.nextName();
        var efunc = msgArgs.getValueOf("func");
        var aParam = msgArgs["A"];
        var bParam = msgArgs["B"];

        const (aName, aIsPdarray) = aParam.tryGetScalar(string),
              (bName, bIsPdarray) = bParam.tryGetScalar(string);

        select (aIsPdarray, bIsPdarray) {
            when (true, true) {
                var aGen: borrowed GenSymEntry = getGenericTypedArrayEntry(aName, st);
                var bGen: borrowed GenSymEntry = getGenericTypedArrayEntry(bName, st);
                if aGen.shape != bGen.shape {
                    var errorMsg = "shape mismatch in arguments to "+pn;
                    eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
                select (aGen.dtype, bGen.dtype) {
                    when (DType.Int64, DType.Int64) {
                        var aEnt = toSymEntry(aGen, int, nd);
                        var bEnt = toSymEntry(bGen, int, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bEnt.a)));
                            }
                        }
                    }
                    when (DType.Int64, DType.UInt64) {
                        var aEnt = toSymEntry(aGen, int, nd);
                        var bEnt = toSymEntry(bGen, uint, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bEnt.a)));
                            }
                        }
                    }
                    when (DType.Int64, DType.Float64) {
                        var aEnt = toSymEntry(aGen, int, nd);
                        var bEnt = toSymEntry(bGen, real, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bEnt.a)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aEnt.a, bEnt.a)));
                            }
                        }
                    }
                    when (DType.UInt64, DType.Int64) {
                        var aEnt = toSymEntry(aGen, uint, nd);
                        var bEnt = toSymEntry(bGen, int, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bEnt.a)));
                            }
                        }
                    }
                    when (DType.UInt64, DType.UInt64) {
                        var aEnt = toSymEntry(aGen, uint, nd);
                        var bEnt = toSymEntry(bGen, uint, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bEnt.a)));
                            }
                        }
                    }
                    when (DType.UInt64, DType.Float64) {
                        var aEnt = toSymEntry(aGen, uint, nd);
                        var bEnt = toSymEntry(bGen, real, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bEnt.a)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aEnt.a, bEnt.a)));
                            }
                        }
                    }
                    when (DType.Float64, DType.Int64) {
                        var aEnt = toSymEntry(aGen, real, nd);
                        var bEnt = toSymEntry(bGen, int, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bEnt.a)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aEnt.a, bEnt.a)));
                            }
                        }
                    }
                    when (DType.Float64, DType.UInt64) {
                        var aEnt = toSymEntry(aGen, real, nd);
                        var bEnt = toSymEntry(bGen, uint, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bEnt.a)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aEnt.a, bEnt.a)));
                            }
                        }
                    }
                    when (DType.Float64, DType.Float64) {
                        var aEnt = toSymEntry(aGen, real, nd);
                        var bEnt = toSymEntry(bGen, real, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bEnt.a)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aEnt.a, bEnt.a)));
                            }
                        }
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn, efunc, aGen.dtype, bGen.dtype);
                        eLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (true, false) {
                var aGen: borrowed GenSymEntry = getGenericTypedArrayEntry(aName, st);
                select (aGen.dtype, bParam.getDType()) {
                    when (DType.Int64, DType.Int64) {
                        var aEnt = toSymEntry(aGen, int, nd);
                        var bScal = bParam.getIntValue();
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bScal)));
                            }
                        }
                    }
                    when (DType.Int64, DType.UInt64) {
                        var aEnt = toSymEntry(aGen, int, nd);
                        var bScal = bParam.getUIntValue();
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bScal)));
                            }
                        }
                    }
                    when (DType.Int64, DType.Float64) {
                        var aEnt = toSymEntry(aGen, int, nd);
                        var bScal = bParam.getRealValue();
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bScal)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aEnt.a, bScal)));
                            }
                        }
                    }
                    when (DType.UInt64, DType.Int64) {
                        var aEnt = toSymEntry(aGen, uint, nd);
                        var bScal = bParam.getIntValue();
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bScal)));
                            }
                        }
                    }
                    when (DType.UInt64, DType.UInt64) {
                        var aEnt = toSymEntry(aGen, uint, nd);
                        var bScal = bParam.getUIntValue();
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bScal)));
                            }
                        }
                    }
                    when (DType.UInt64, DType.Float64) {
                        var aEnt = toSymEntry(aGen, uint, nd);
                        var bScal = bParam.getRealValue();
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bScal)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aEnt.a, bScal)));
                            }
                        }
                    }
                    when (DType.Float64, DType.Int64) {
                        var aEnt = toSymEntry(aGen, real, nd);
                        var bScal = bParam.getIntValue();
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bScal)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aEnt.a, bScal)));
                            }
                        }
                    }
                    when (DType.Float64, DType.UInt64) {
                        var aEnt = toSymEntry(aGen, real, nd);
                        var bScal = bParam.getUIntValue();
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bScal)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aEnt.a, bScal)));
                            }
                        }
                    }
                    when (DType.Float64, DType.Float64) {
                        var aEnt = toSymEntry(aGen, real, nd);
                        var bScal = bParam.getRealValue();
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aEnt.a, bScal)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aEnt.a, bScal)));
                            }
                        }
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn, efunc, aGen.dtype, bParam.getDType());
                        eLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (false, true) {
                var bGen: borrowed GenSymEntry = getGenericTypedArrayEntry(bName, st);
                select (aParam.getDType(), bGen.dtype) {
                    when (DType.Int64, DType.Int64) {
                        var aScal = aParam.getIntValue();
                        var bEnt = toSymEntry(bGen, int, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aScal, bEnt.a)));
                            }
                        }
                    }
                    when (DType.Int64, DType.UInt64) {
                        var aScal = aParam.getIntValue();
                        var bEnt = toSymEntry(bGen, uint, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aScal, bEnt.a)));
                            }
                        }
                    }
                    when (DType.Int64, DType.Float64) {
                        var aScal = aParam.getIntValue();
                        var bEnt = toSymEntry(bGen, real, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aScal, bEnt.a)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aScal, bEnt.a)));
                            }
                        }
                    }
                    when (DType.UInt64, DType.Int64) {
                        var aScal = aParam.getUIntValue();
                        var bEnt = toSymEntry(bGen, int, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aScal, bEnt.a)));
                            }
                        }
                    }
                    when (DType.UInt64, DType.UInt64) {
                        var aScal = aParam.getUIntValue();
                        var bEnt = toSymEntry(bGen, uint, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aScal, bEnt.a)));
                            }
                        }
                    }
                    when (DType.UInt64, DType.Float64) {
                        var aScal = aParam.getUIntValue();
                        var bEnt = toSymEntry(bGen, real, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aScal, bEnt.a)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aScal, bEnt.a)));
                            }
                        }
                    }
                    when (DType.Float64, DType.Int64) {
                        var aScal = aParam.getRealValue();
                        var bEnt = toSymEntry(bGen, int, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aScal, bEnt.a)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aScal, bEnt.a)));
                            }
                        }
                    }
                    when (DType.Float64, DType.UInt64) {
                        var aScal = aParam.getRealValue();
                        var bEnt = toSymEntry(bGen, uint, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aScal, bEnt.a)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aScal, bEnt.a)));
                            }
                        }
                    }
                    when (DType.Float64, DType.Float64) {
                        var aScal = aParam.getRealValue();
                        var bEnt = toSymEntry(bGen, real, nd);
                        select efunc {
                            when "arctan2" {
                                st.addEntry(rname, new shared SymEntry(atan2(aScal, bEnt.a)));
                            }
                            when "fmod" {
                                st.addEntry(rname, new shared SymEntry(fmod(aScal, bEnt.a)));
                            }
                        }
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn, efunc, aParam.getDType(), bGen.dtype);
                        eLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
        }
        repMsg = "created " + st.attrib(rname);
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /*
    These are ternary functions which take three arrays and produce an array.
    vector = efunc(vector, vector, vector)

    :arg reqMsg: request containing (cmd,efunc,name1,name2,name3)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (MsgTuple)
    :throws: `UndefinedSymbolError(name)`
    */

    // Presently, only two functions are implemented in efunc3vvMsg: where and putmask.
    // This makes it an excellent candidate for rewriting to move both where and putmask
    // into functions using the new interface.

    @arkouda.registerND
    proc efunc3vvMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; 
        var rname = st.nextName();
        
        var efunc = msgArgs.getValueOf("func");
        var g1: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("condition"), st);
        var g2: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("a"), st);
        var g3: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("b"), st);
        if g1.shape != g2.shape {    // both where and putmask require condition's shape to match 1st data input's
            var errorMsg = "shape mismatch in arguments to "+pn;
            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        if g1.dtype != DType.Bool {
            var errorMsg = "condition must be of type Bool in "+pn ;
            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        var e1 = toSymEntry(g1, bool, nd);
        select efunc {
            when "where" {
                if g2.shape != g3.shape {    // where requires all inputs to be of same shape
                    var errorMsg = "shape mismatch in arguments to "+pn;
                    eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
                select (g2.dtype,g3.dtype) {    // where allows any combination of numerical types as inputs
                   when (DType.Float64, DType.Float64) {
                       var e2 = toSymEntry(g2, real, nd);
                       var e3 = toSymEntry(g3, real, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.Float64, DType.Int64) {
                       var e2 = toSymEntry(g2, real, nd);
                       var e3 = toSymEntry(g3, int, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.Float64, DType.UInt64) {
                       var e2 = toSymEntry(g2, real, nd);
                       var e3 = toSymEntry(g3, uint, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.Float64, DType.Bool) {
                       var e2 = toSymEntry(g2, real, nd);
                       var e3 = toSymEntry(g3, bool, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   } 
                   when (DType.Int64, DType.Float64) {
                       var e2 = toSymEntry(g2, int, nd);
                       var e3 = toSymEntry(g3, real, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.Int64, DType.Int64) {
                       var e2 = toSymEntry(g2, int, nd);
                       var e3 = toSymEntry(g3, int, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.Int64, DType.UInt64) {
                       var e2 = toSymEntry(g2, int, nd);
                       var e3 = toSymEntry(g3, uint, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.Int64, DType.Bool) {
                       var e2 = toSymEntry(g2, int, nd);
                       var e3 = toSymEntry(g3, bool, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.UInt64, DType.Float64) {
                       var e2 = toSymEntry(g2, uint, nd);
                       var e3 = toSymEntry(g3, real, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.UInt64, DType.Int64) {
                       var e2 = toSymEntry(g2, uint, nd);
                       var e3 = toSymEntry(g3, int, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.UInt64, DType.UInt64) {
                       var e2 = toSymEntry(g2, uint, nd);
                       var e3 = toSymEntry(g3, uint, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.UInt64, DType.Bool) {
                       var e2 = toSymEntry(g2, uint, nd);
                       var e3 = toSymEntry(g3, bool, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.Bool, DType.Float64) {
                       var e2 = toSymEntry(g2, bool, nd);
                       var e3 = toSymEntry(g3, real, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.Bool, DType.Int64) {
                       var e2 = toSymEntry(g2, bool, nd);
                       var e3 = toSymEntry(g3, int, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.Bool, DType.UInt64) {
                       var e2 = toSymEntry(g2, bool, nd);
                       var e3 = toSymEntry(g3, uint, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   when (DType.Bool, DType.Bool) {
                       var e2 = toSymEntry(g2, bool, nd);
                       var e3 = toSymEntry(g3, bool, nd);
                       var a = where_helper_3vv(e1.a, e2.a, e3.a);
                       st.addEntry(rname, new shared SymEntry(a));
                   }
                   otherwise {
                       var errorMsg = "arg types incompatible with where in "+pn ;
                       eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                       return new MsgTuple(errorMsg, MsgType.ERROR);
                   } 
                } // end of "select" 
                repMsg = "created " + st.attrib(rname);
                eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }  // end of when where

            when "putmask" {                // putmask only requires original values and mask to be ofsame shape,
                if nd != 1 {
                    var errorMsg = "multi-dim putmask is not yet enabled";
                    eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
                select (g2.dtype,g3.dtype) {    // and allows all of the data type combinations herein.
                    when (DType.Float64, DType.Float64) {
                        var e2 = toSymEntry(g2, real, nd);
                        var e3 = toSymEntry(g3, real, nd);
                        putmask_helper(e1.a,e2.a,e3.a);
                    }
                    when (DType.Float64, DType.Int64) {
                        var e2 = toSymEntry(g2, real, nd);
                        var e3 = toSymEntry(g3, int, nd);
                        putmask_helper(e1.a,e2.a,e3.a);
                    }
                    when (DType.Float64, DType.UInt64) {
                        var e2 = toSymEntry(g2, real, nd);
                        var e3 = toSymEntry(g3, uint, nd);
                        putmask_helper(e1.a,e2.a,e3.a);
                    }
                    when (DType.Float64, DType.Bool) {
                        var e2 = toSymEntry(g2, real, nd);
                        var e3 = toSymEntry(g3, bool, nd);
                        putmask_helper(e1.a,e2.a,e3.a);
                    }
                    when (DType.Int64, DType.Int64) {
                        var e2 = toSymEntry(g2, int, nd);
                        var e3 = toSymEntry(g3, int, nd);
                        putmask_helper(e1.a,e2.a,e3.a);
                    }
                    when (DType.Int64, DType.UInt64) {
                        var e2 = toSymEntry(g2, int, nd);
                        var e3 = toSymEntry(g3, uint, nd);
                        putmask_helper(e1.a,e2.a,e3.a);
                    }
                    when (DType.Int64, DType.Bool) {
                        var e2 = toSymEntry(g2, int, nd);
                        var e3 = toSymEntry(g3, bool, nd);
                        putmask_helper(e1.a,e2.a,e3.a);
                    }
                    when (DType.UInt64, DType.UInt64) {
                        var e2 = toSymEntry(g2, uint, nd);
                        var e3 = toSymEntry(g3, uint, nd);
                        putmask_helper(e1.a,e2.a,e3.a);
                    }
                    when (DType.UInt64, DType.Bool) {
                        var e2 = toSymEntry(g2, uint, nd);
                        var e3 = toSymEntry(g3, bool, nd);
                        putmask_helper(e1.a,e2.a,e3.a);
                    }
                    when (DType.Bool, DType.Bool) {
                        var e2 = toSymEntry(g2, bool, nd);
                        var e3 = toSymEntry(g3, bool, nd);
                        putmask_helper(e1.a,e2.a,e3.a);
                    }
                    otherwise {
                        var errorMsg = "arg types incompatible with putmask in "+pn ;
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    } 
                }  // end of select
                return MsgTuple.success();  // putmask does not return a new pdarray; it writes in place
            } // end of when putmask
            otherwise { // neither where nor putmask
                var errorMsg = notImplementedError(pn,efunc,g1.dtype,g2.dtype,g3.dtype);
                eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);       
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }  // end of select efunc
        repMsg = "created " + st.attrib(rname);
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
        return new MsgTuple(repMsg, MsgType.NORMAL); 
    } // end of efunc3vvMsg

    @arkouda.registerND
    proc efunc3vsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var efunc = msgArgs.getValueOf("func");
        var dtype = str2dtype(msgArgs.getValueOf("dtype"));
        var rname = st.nextName();

        var name1 = msgArgs.getValueOf("condition");
        var name2 = msgArgs.getValueOf("a");
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s efunc: %s scalar: %s dtype: %s name1: %s name2: %s rname: %s".format(
             cmd,efunc,msgArgs.getValueOf("scalar"),dtype,name1,name2,rname));

        var g1: borrowed GenSymEntry = getGenericTypedArrayEntry(name1, st);
        var g2: borrowed GenSymEntry = getGenericTypedArrayEntry(name2, st);
        if !(g1.shape == g2.shape) {
            var errorMsg = "shape mismatch in arguments to "+pn;
            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);  
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        select (g1.dtype, g2.dtype, dtype) {
            when (DType.Bool, DType.Int64, DType.Int64) {
               var e1 = toSymEntry(g1, bool, nd);
               var e2 = toSymEntry(g2, int, nd);
               var val = msgArgs.get("scalar").getIntValue();
               select efunc {
                  when "where" {
                      var a = where_helper(e1.a, e2.a, val, 1);
                      st.addEntry(rname, new shared SymEntry(a));
                  }
                  otherwise {
                      var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                         g2.dtype,dtype);
                      eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                      return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
               } 
            }
            when (DType.Bool, DType.UInt64, DType.UInt64) {
               var e1 = toSymEntry(g1, bool, nd);
               var e2 = toSymEntry(g2, uint, nd);
               var val = msgArgs.get("scalar").getUIntValue();
               select efunc {
                  when "where" {
                      var a = where_helper(e1.a, e2.a, val, 1);
                      st.addEntry(rname, new shared SymEntry(a));
                  }
                  otherwise {
                      var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                         g2.dtype,dtype);
                      eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                      return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
               } 
            }
            when (DType.Bool, DType.Float64, DType.Float64) {
                var e1 = toSymEntry(g1, bool, nd);
                var e2 = toSymEntry(g2, real, nd);
                var val = msgArgs.get("scalar").getRealValue();
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, e2.a, val, 1);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                          g2.dtype,dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            } 
            when (DType.Bool, DType.Bool, DType.Bool) {
                var e1 = toSymEntry(g1, bool, nd);
                var e2 = toSymEntry(g2, bool, nd);
                var val = msgArgs.get("scalar").getBoolValue();
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, e2.a, val, 1);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                           g2.dtype,dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                         
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                } 
            }
            otherwise {
                var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                   g2.dtype,dtype);
                eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                return new MsgTuple(errorMsg, MsgType.ERROR);            
            }
        }

        repMsg = "created " + st.attrib(rname);
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
        return new MsgTuple(repMsg, MsgType.NORMAL); 
    }

    /*
    vector = efunc(vector, scalar, vector)

    :arg reqMsg: request containing (cmd,efunc,name1,dtype,value,name2)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (MsgTuple)
    :throws: `UndefinedSymbolError(name)`
    */
    @arkouda.registerND
    proc efunc3svMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var efunc = msgArgs.getValueOf("func");
        var dtype = str2dtype(msgArgs.getValueOf("dtype"));
        var rname = st.nextName();

        var name1 = msgArgs.getValueOf("condition");
        var name2 = msgArgs.getValueOf("b");
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s efunc: %s scalar: %s dtype: %s name1: %s name2: %s rname: %s".format(
             cmd,efunc,msgArgs.getValueOf("scalar"),dtype,name1,name2,rname));

        var g1: borrowed GenSymEntry = getGenericTypedArrayEntry(name1, st);
        var g2: borrowed GenSymEntry = getGenericTypedArrayEntry(name2, st);
        if !(g1.shape == g2.shape) {
            var errorMsg = "shape mismatch in arguments to "+pn;
            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);            
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        select (g1.dtype, dtype, g2.dtype) {
            when (DType.Bool, DType.Int64, DType.Int64) {
                var e1 = toSymEntry(g1, bool, nd);
                var val = msgArgs.get("scalar").getIntValue();
                var e2 = toSymEntry(g2, int, nd);
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, val, e2.a, 2);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                           dtype,g2.dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                  
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }   
               } 
            }
            when (DType.Bool, DType.UInt64, DType.UInt64) {
                var e1 = toSymEntry(g1, bool, nd);
                var val = msgArgs.get("scalar").getUIntValue();
                var e2 = toSymEntry(g2, uint, nd);
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, val, e2.a, 2);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                           dtype,g2.dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                  
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }   
               } 
            }
            when (DType.Bool, DType.Float64, DType.Float64) {
                var e1 = toSymEntry(g1, bool, nd);
                var val = msgArgs.get("scalar").getRealValue();
                var e2 = toSymEntry(g2, real, nd);
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, val, e2.a, 2);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                      var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                           dtype,g2.dtype);
                      eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                      return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                } 
            }
            when (DType.Bool, DType.Bool, DType.Bool) {
                var e1 = toSymEntry(g1, bool, nd);
                var val = msgArgs.get("scalar").getBoolValue();
                var e2 = toSymEntry(g2, bool, nd);
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, val, e2.a, 2);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                           dtype,g2.dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg, MsgType.ERROR);                    
                    }
               } 
            }
            otherwise {
                var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                   dtype,g2.dtype);
                eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                                                 
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }

        repMsg = "created " + st.attrib(rname);
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
        return new MsgTuple(repMsg, MsgType.NORMAL); 
    }

    /*
    vector = efunc(vector, scalar, scalar)
    
    :arg reqMsg: request containing (cmd,efunc,name1,dtype1,value1,dtype2,value2)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (MsgTuple)
    :throws: `UndefinedSymbolError(name)`
    */
    @arkouda.registerND
    proc efunc3ssMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var dtype = str2dtype(msgArgs.getValueOf("dtype"));
        var efunc = msgArgs.getValueOf("func");
        var rname = st.nextName();
        
        var name1 = msgArgs.getValueOf("condition");

        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s efunc: %s scalar1: %s dtype1: %s scalar2: %s name: %s rname: %s".format(
             cmd,efunc,msgArgs.getValueOf("a"),dtype,msgArgs.getValueOf("b"),name1,rname));

        var g1: borrowed GenSymEntry = getGenericTypedArrayEntry(name1, st);
        select (g1.dtype, dtype) {
            when (DType.Bool, DType.Int64) {
                var e1 = toSymEntry(g1, bool, nd);
                var val1 = msgArgs.get("a").getIntValue();
                var val2 = msgArgs.get("b").getIntValue();
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, val1, val2, 3);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                      dtype, dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                } 
            }
            when (DType.Bool, DType.UInt64) {
                var e1 = toSymEntry(g1, bool, nd);
                var val1 = msgArgs.get("a").getUIntValue();
                var val2 = msgArgs.get("b").getUIntValue();
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, val1, val2, 3);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                      dtype, dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                } 
            }
            when (DType.Bool, DType.Float64) {
                var e1 = toSymEntry(g1, bool, nd);
                var val1 = msgArgs.get("a").getRealValue();
                var val2 = msgArgs.get("b").getRealValue();
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, val1, val2, 3);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                        dtype, dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg, MsgType.ERROR);                                                     
                    }
                } 
            }
            when (DType.Bool, DType.Bool) {
                var e1 = toSymEntry(g1, bool, nd);
                var val1 = msgArgs.get("a").getBoolValue();
                var val2 = msgArgs.get("b").getBoolValue();
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, val1, val2, 3);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                       dtype, dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg, MsgType.ERROR);      
                   }
               } 
            }
            otherwise {
                var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                               dtype, dtype);
                eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                return new MsgTuple(errorMsg, MsgType.ERROR);                                             
            }
        }

        repMsg = "created " + st.attrib(rname);
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
        return new MsgTuple(repMsg, MsgType.NORMAL); 
    }

        // The 3vv version of where_helper has been pulled out as a separate function,
        // to ease the transition to matching the data types as allowed by numpy.
    proc where_helper_3vv(cond:[?D] bool, A:[D] ?ta, B:[D] ?tb) : [D] throws {
    type resultType = compute_result_type(ta,tb) ;
        var C = makeDistArray(D, resultType);
        forall (ch, a, b, c) in zip(cond, A, B, C) {
          c = if ch then a:resultType else b:resultType;
        }
        return C;
    }

    // This proc gives the return type for ak.where, given the input types.
    // cribbed from LinalgMsg.chpl, with uint(8) changed to uint(64)
    proc compute_result_type (type t1, type t2) type {
        if t1 == real || t2 == real then return real;
        if t1 == int || t2 == int then return int;
        if t1 == uint(64) || t2 == uint(64) then return uint(64);
        return bool;
    }

    /* The 'where' function takes a boolean array and two other arguments A and B, and 
       returns an array with A where the boolean is true and B where it is false. A and B
       can be vectors or scalars. 
       Dev Note: I would like to be able to write these functions without
       the param kind and just let the compiler choose, but it complains about an
       ambiguous call. 
       
       :arg cond:
       :type cond: [?D] bool

       :arg A:
       :type A: [D] ?t

       :arg B: 
       :type B: [D] t

       :arg kind:
       :type kind: param
       */

    proc where_helper(cond:[?D] bool, A:[D] ?t, B:[D] t, param kind):[D] t throws where (kind == 0) {
      var C = makeDistArray(D, t);
      forall (ch, a, b, c) in zip(cond, A, B, C) {
        c = if ch then a else b;
      }
      return C;
    }

    /*

    :arg cond:
    :type cond: [?D] bool

    :arg A:
    :type A: [D] ?t

    :arg B: 
    :type B: t

    :arg kind:
    :type kind: param
    */
    proc where_helper(cond:[?D] bool, A:[D] ?t, b:t, param kind):[D] t throws where (kind == 1) {
      var C = makeDistArray(D, t);
      forall (ch, a, c) in zip(cond, A, C) {
        c = if ch then a else b;
      }
      return C;
    }

    /*

    :arg cond:
    :type cond: [?D] bool

    :arg a:
    :type a: ?t

    :arg B: 
    :type B: [D] t

    :arg kind:
    :type kind: param
    */
    proc where_helper(cond:[?D] bool, a:?t, B:[D] t, param kind):[D] t throws where (kind == 2) {
      var C = makeDistArray(D, t);
      forall (ch, b, c) in zip(cond, B, C) {
        c = if ch then a else b;
      }
      return C;
    }

    /*
    
    :arg cond:
    :type cond: [?D] bool

    :arg a:
    :type a: ?t

    :arg b: 
    :type b: t

    :arg kind:
    :type kind: param
    */
    proc where_helper(cond:[?D] bool, a:?t, b:t, param kind):[D] t throws where (kind == 3) {
      var C = makeDistArray(D, t);
      forall (ch, c) in zip(cond, C) {
        c = if ch then a else b;
      }
      return C;
    }
//}
//}
    proc putmask_helper (mask : [?D1] bool, ref A : [D1] ?ta, Values : [?D2] ?tv) {

//  Note, added casting to the output, to accomodate the various combinations of
//  types that are allowed by np.putmask.  Invalid pairings of types ta, tv must be 
//  caught and flagged by the calling function.  This proc does no type checking.

      if A.size == Values.size { // then their distributions match
         forall element in A.domain {
             if mask[element] then A[element] = Values[element]:ta ;
         }    

      } else if A.size < Values.size { // then prune Values so the distributions match
         var Pruned_Values = Values[0..A.size-1] ;
         forall element in A.domain {
             if mask[element] then A[element] = Pruned_Values[element]:ta ;
         }    

//  For the remaining cases, we know Values.size < A.size

      } else {
         var A_subset_size = (A.size/numLocales):int ;
         if A.size%numLocales == 0 then A_subset_size += 1 ;
 
//  If Values.size < A_subset_size, then we may as well make a copy of Values on each locale

         if Values.size < A_subset_size {
             coforall loc in Locales do on loc {
                 var local_Values : [0..Values.size-1] tv = Values ;
                 forall element in A.localSubdomain() {
                     if mask[element] then A[element] = (local_Values[element%local_Values.size]):ta ;
                 }
             }

//  Finally, we have the case where Values.size > A_subset_size, which means Values is distributed
//  but probably not to our liking.  Use an aggregator to get the right values in the right places.

         } else {
             coforall loc in Locales do on loc {
                var local_Values : [0..A.localSubdomain().size-1] tv;
                forall (element, item) in zip (A.localSubdomain(),local_Values) with (var agg = newSrcAggregator(tv)) {
                    agg.copy(item,Values[element%Values.size]) ;
                }
                forall (element, item) in zip (A.localSubdomain(),local_Values) {
                    if mask[element] then A[element] = item:ta ;
                }
             }    
         }
      }
      return ;
    }
}
