
module EfuncMsg
{

    use ServerConfig;

    use Time;
    use Math;
    use BitOps;
    use Reflection;
    use ServerErrors;
    use List;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    private use SipHash;
    use SortMsg;
    use UniqueMsg;
    use AryUtil;
    use CTypes;
    use OS.POSIX;

    use CommAggregation;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const eLogger = new Logger(logLevel, logChannel);

    extern proc fmod(x: real, y: real): real;

    // These ops are functions which take an array and produce an array.

    @arkouda.registerCommand(name="sin")
    proc ak_sin (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    { 
        return sin(x);
    }
       
    @arkouda.registerCommand(name="cos")
    proc ak_cos (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return cos(x);
    }
       
    @arkouda.registerCommand(name="tan")
    proc ak_tan (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return tan(x);
    }
       
    @arkouda.registerCommand()
    proc arcsin (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return asin(x);
    }
       
    @arkouda.registerCommand()
    proc arccos (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return acos(x);
    }
       
    @arkouda.registerCommand()
    proc arctan (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return atan(x);
    }
       
    @arkouda.registerCommand(name="sinh")
    proc ak_sinh (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return sinh(x);
    }
       
    @arkouda.registerCommand(name="cosh")
    proc ak_cosh (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return cosh(x);
    }
       
    @arkouda.registerCommand(name="tanh")
    proc ak_tanh (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return tanh(x);
    }
       
    @arkouda.registerCommand()
    proc arcsinh (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return asinh(x);
    }
       
    @arkouda.registerCommand()
    proc arccosh (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return acosh(x);
    }
       
    @arkouda.registerCommand()
    proc arctanh (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return atanh(x);
    }
       
    @arkouda.registerCommand(name="abs")
    proc ak_abs (const ref pda : [?d] ?t) : [d] t throws
        where (t==int || t==real) // TODO maybe: allow uint also
    {
        return abs(pda);
    }

    @arkouda.registerCommand(name="square")
    proc ak_square (const ref x : [?d] ?t) : [d] t throws
        where (t==int || t==real || t==uint)
    {
        return square(x);
    }

    @arkouda.registerCommand(name="exp")
    proc ak_exp (const ref pda : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return exp(pda);
    }

    @arkouda.registerCommand(name="expm1")
    proc ak_expm1 (const ref pda : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return expm1(pda);
    }

    @arkouda.registerCommand(name="log")
    proc ak_log (const ref pda : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return log(pda);
    }

    @arkouda.registerCommand(name="log1p")
    proc ak_log1p (const ref pda : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return log1p(pda);
    }

    //  chapel log2 returns ints when given ints, so the input has been cast to real.

    @arkouda.registerCommand(name="log2")
    proc ak_log2 (const ref pda : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return log2(pda:real);
    }

    @arkouda.registerCommand(name="log10")
    proc ak_log10 (const ref pda : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return log10(pda);
    }

    @arkouda.registerCommand()
    proc isinf (pda : [?d] real) : [d] bool
    {
        return (isInf(pda));
    }

    @arkouda.registerCommand()
    proc isnan (pda : [?d] real) : [d] bool
    {
        return (isNan(pda));
    }

    @arkouda.registerCommand()
    proc isfinite (pda : [?d] real) : [d] bool
    {
        return (isFinite(pda));
    }

    @arkouda.registerCommand(name="floor")
    proc ak_floor (x : [?d] ?t) : [d] real throws
        where (t==real)
    {
        return floor(x);
    }

    @arkouda.registerCommand(name="ceil")
    proc ak_ceil (x : [?d] ?t) : [d] real throws
        where (t==real)
    {
        return ceil(x);
    }

    @arkouda.registerCommand(name="round")
    proc ak_round (x : [?d] ?t) : [d] real throws
        where (t==real)
    {
        return round(x);
    }

    @arkouda.registerCommand(name="trunc")
    proc ak_trunc (x : [?d] ?t) : [d] real throws
        where (t==real)
    {
        return trunc(x);
    }

    @arkouda.registerCommand()
    proc popcount (pda : [?d] ?t) : [d] t throws
        where (t==int || t==uint)
    {
        return popCount(pda);
    }

    @arkouda.registerCommand(name="parity")
    proc ak_parity (pda : [?d] ?t) : [d] t throws
        where (t==int || t==uint)
    {
        return parity(pda);
    }

    @arkouda.registerCommand(name="clz")
    proc ak_clz (pda : [?d] ?t) : [d] t throws
        where (t==int || t==uint)
    {
        return clz(pda);
    }

    @arkouda.registerCommand(name="ctz")
    proc ak_ctz (pda : [?d] ?t) : [d] t throws
        where (t==int || t==uint)
    {
        return ctz(pda);
    }

    @arkouda.registerCommand()
    proc not (pda : [?d] ?t) : [d] bool throws
        where (t==int || t==uint || t==bool)
    {
        return (!pda);
    }

    @arkouda.registerCommand()
    proc nextafter (x1: [?d] real(64), x2: [d] real(64)): [d] real(64) throws
    {
        var outArray: [d] real;
        forall outIdx in outArray.domain {
            if isNan(x1[outIdx]) || isNan(x2[outIdx]) {
                outArray[outIdx] = nan;
                continue;
            }
            if x1[outIdx] == 0.0 && x2[outIdx] != 0.0 {
                outArray[outIdx] = if x2[outIdx] > 0.0 then 5e-324 else -5e-324;
                continue;
            }

            // You might say, "Well, this looks silly." But really, I'm handling positive and negative zero here.
            if x1[outIdx] == 0.0 && x2[outIdx] == 0.0 {
                outArray[outIdx] = x2[outIdx];
                continue;
            }
            var intValue: int(64);
            var realValueRef = x1[outIdx];
            memcpy(c_ptrTo(intValue), c_ptrTo(realValueRef), c_sizeof(real(64)));
            if ((x1[outIdx] > 0 && x1[outIdx] < x2[outIdx]) || (x1[outIdx] < 0 && x1[outIdx] > x2[outIdx])) {
                intValue += 1;
            }
            if ((x1[outIdx] > 0 && x1[outIdx] > x2[outIdx]) || (x1[outIdx] < 0 && x1[outIdx] < x2[outIdx])) {
                intValue -= 1;
            }
            var nextRealValue: real(64);
            var intValueRef = intValue;
            memcpy(c_ptrTo(nextRealValue), c_ptrTo(intValueRef), c_sizeof(int(64)));
            outArray[outIdx] = nextRealValue;
        }
        return outArray;
    }

    //  cumsum and cumprod -- the below helper function gives return type

    proc cumspReturnType(type t) type
      do return if t == bool then int else t;

    // Implements + reduction over numeric data, converting all elements to int before summing.
    // See https://chapel-lang.org/docs/technotes/reduceIntents.html#readme-reduceintents-interface

    class PlusIntReduceOp: ReduceScanOp {
        type eltType;
        var value: int;
        proc identity      do return 0: int;
        proc accumulate(elm)  { value = value + elm:int; }
        proc accumulateOntoState(ref state, elm)  { state = state + elm:int; }
        proc initialAccumulate(outerVar) { value = value + outerVar: int; }
        proc combine(other: borrowed PlusIntReduceOp(?))   { value = value + other.value; }
        proc generate()    do return value;
        proc clone()       do return new unmanaged PlusIntReduceOp(eltType=eltType);
    }

    @arkouda.registerCommand()
    proc cumsum(x : [?d] ?t) : [d] cumspReturnType(t) throws
        where (t==int || t==real || t==uint || t==bool) && (d.rank==1)
    {
        overMemLimit(numBytes(int) * x.size) ;
        if t == bool {
            return (PlusIntReduceOp scan x);
        } else {
            return (+ scan x) ;
        }
    }

    @arkouda.registerCommand()
    proc cumprod(x : [?d] ?t) : [d] cumspReturnType(t) throws
        where (t==int || t==real || t==uint || t==bool) && (d.rank==1)
    {
        overMemLimit(numBytes(int) * x.size) ;
        if t == bool {
            return (&& scan x);
        } else {
            return (*scan x) ;
        }
    }

    // sgn is a special case.  It is the only thing that returns int(8).

    @arkouda.registerCommand(name="sgn")
    proc ak_sgn (pda : [?d] ?t) : [d] t throws
        where (t==int || t==real)
    {
        return (sgn(pda));
    }

    // Hashes are more of a challenge to unhook from the old interface, but they
    // have been pulled out into their own functions.

    @arkouda.instantiateAndRegister
    proc hash64 (cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int) : MsgTuple throws 
        where ((array_dtype==real || array_dtype==int || array_dtype==uint) && array_nd==1)
    {
        const efunc = msgArgs.getValueOf("x"),
            e = st[msgArgs["x"]]: SymEntry(array_dtype,array_nd);
        const rname = st.nextName();
        overMemLimit(numBytes(array_dtype)*e.size);
        var a = st.addEntry(rname, e.tupShape, uint);
        forall (ai, x) in zip (a.a, e.a) {
            ai = sipHash64(x) : uint ;
        }
        var repMsg = "created " + st.attrib(rname);
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    @arkouda.instantiateAndRegister
    proc hash128 (cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int) : MsgTuple throws 
        where ((array_dtype==real || array_dtype==int || array_dtype==uint) && array_nd==1)
    {
        const efunc = msgArgs.getValueOf("x"),
            e = st[msgArgs["x"]]: SymEntry(array_dtype,array_nd);
        const rname = st.nextName();
        var rname2 = st.nextName();
        overMemLimit(numBytes(array_dtype) * e.size * 2);
        var a1 = st.addEntry(rname2, e.tupShape, uint);
        var a2 = st.addEntry(rname, e.tupShape, uint);
        forall (a1i, a2i, x) in zip(a1.a, a2.a, e.a) {
            (a1i, a2i) = sipHash128(x): (uint, uint);
        }
        var repMsg = "created " + st.attrib(rname2) + "+";
        repMsg += "created " + st.attrib(rname);
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    //  The functions fmod, arctan2, and where require some special handling, since the 
    //  arkouda interface doesn't work well with unknown scalar types.

    //  with two 2 vector inputs, both types can be unknown.

    @arkouda.registerCommand()
    proc fmod2vv ( a: [?d] ?ta, b : [d] ?tb) : [d] real throws
        where ((ta==real && (tb==int || tb == uint || tb==real)) ||
               (ta==int  && tb==real) ||
               (ta==uint && tb==real))
    {
        return (fmod(a,b));
    }

    //  1 vector and 1 scalar (or 1 scalar and 1 vector) requires the scalar type be
    //  specified.

    @arkouda.registerCommand(name="fmod2vs_float64")
    proc fmod2vsr ( a: [?d] ?ta, b : real) : [d] real throws
        where (ta==int || ta==uint || ta==real)
    {
        return (fmod(a,b));
    }

    @arkouda.registerCommand(name="fmod2vs_int64")
    proc fmod2vsi ( a: [?d] ?ta, b : int) : [d] real throws
        where (ta==int || ta==uint || ta==real)
    {
        return (fmod(a,b));
    }

    @arkouda.registerCommand(name="fmod2vs_uint64")
    proc fmod2vsu ( a: [?d] ?ta, b : uint) : [d] real throws
        where (ta==int || ta==uint || ta==real)
    {
        return (fmod(a,b));
    }

    @arkouda.registerCommand(name="fmod2sv_float64")
    proc fmod2svr ( a: real, b : [?d] ?tb) : [d] real throws
        where (tb==int || tb==uint || tb==real)
    {
        return (fmod(a,b));
    }

    @arkouda.registerCommand(name="fmod2sv_int64")
    proc fmod2svi ( a: int, b : [?d] ?tb) : [d] real throws
        where (tb==int || tb==uint || tb==real)
    {
        return (fmod(a,b));
    }

    @arkouda.registerCommand(name="fmod2sv_uint64")
    proc fmod2svu ( a: uint, b : [?d] ?tb) : [d] real throws
        where (tb==int || tb==uint || tb==real)
    {
        return (fmod(a,b));
    }

    //  The above comment re scalar types applies to arctan2 as well.

    @arkouda.registerCommand()
    proc arctan2vv (a : [?d] ?ta, b : [d] ?tb) : [d] real throws
        where ( (ta==real && (tb==real || tb==int || tb==uint)) ||
                (ta==int  && (tb==real || tb==int || tb==uint)) ||
                (ta==uint && (tb==real || tb==int || tb==uint)) ) {
            return (atan2(a,b));
        }

    @arkouda.registerCommand(name="arctan2vs_float64")
    proc arctan2vsr (a : [?d] ?ta, b : real) : [d] real throws
        where (ta==int || ta==uint || ta==real) {
            return (atan2(a,b));
        }

    @arkouda.registerCommand(name="arctan2vs_int64")
    proc arctan2vsi (a : [?d] ?ta, b : int) : [d] real throws
        where (ta==int || ta==uint || ta==real) {
            return (atan2(a,b));
        }

    @arkouda.registerCommand(name="arctan2vs_uint64")
    proc arctan2vsu (a : [?d] ?ta, b : uint) : [d] real throws
        where (ta==int || ta==uint || ta==real) {
            return (atan2(a,b));
        }

    @arkouda.registerCommand(name="arctan2sv_float64")
    proc arctan2svr (a : real, b : [?d] ?tb) : [d] real throws
        where (tb==int || tb==uint || tb==real) {
            return (atan2(a,b));
        }

    @arkouda.registerCommand(name="arctan2sv_int64")
    proc arctan2svi (a : int, b : [?d] ?tb) : [d] real throws
        where (tb==int || tb==uint || tb==real) {
            return (atan2(a,b));
        }

    @arkouda.registerCommand(name="arctan2sv_uint64")
    proc arctan2svu (a : uint, b : [?d] ?tb) : [d] real throws
        where (tb==int || tb==uint || tb==real) {
            return (atan2(a,b));
        }

    // The above comment re scalar types (in fmod) allso applies to "where"

    // where's return type depends on the type of its two inputs.  The function below provides
    // the return type.  The (int,uint) -> real may look odd, but it is done to match numpy.

    proc whereReturnType(type ta, type tb) type throws {
        if ( (ta==real || tb==real) || (ta==int && tb==uint) || (ta==uint && tb==int) ) {
            return real ;
        } else if (ta==int || tb==int) {
            return int ;
        } else if (ta==uint || tb==uint) {
            return uint ;
        } else if (ta==bool || tb==bool) {
            return bool ;
        } else {
          throw new Error ("where does not support types %s %s".format(type2str(ta),type2str(tb))) ; 
        }
    }

    @arkouda.registerCommand()
    proc wherevv ( condition: [?d] bool, a : [d] ?ta, b : [d] ?tb) : [d] whereReturnType(ta,tb) throws
    where ((ta==real || ta==int || ta==uint || ta==bool) && (tb==real || tb==int || tb==uint || tb==bool))
    {
        var c = makeDistArray(d, whereReturnType(ta,tb));
        if numLocales == 1 { 
           forall (ch, A, B, C) in zip(condition, a, b, c) { 
                C = if ch then A:whereReturnType(ta,tb) else B:whereReturnType(ta,tb) ;
           }    
        } else {
            coforall loc in Locales do on loc {
                var cLD = c.localSubdomain() ;   // they all have the same domain
                forall idx in cLD { 
                   c[idx] = if condition[idx] then a[idx]:whereReturnType(ta,tb)
                                              else b[idx]:whereReturnType(ta,tb) ;
                }
            }
        }
        return c;
    }

    // The latest change to where eliminates the needs for wherevs, wheresv, and wheress.
    // So as a test, I'm commenting them all out.

    // The wherevs (vector, scalar), wheresv (scalar, vector) and wheress (scalar, scalar)
    // implementations all call corresponding helper functions.  This is done because the
    // arkouda interface requires the scalar types to be specified, but the helper functions,
    // since they're not registered as commands, can be more generic.

 
 /*
    proc wherevsHelper (condition: [?d] bool, a : [d] ?ta, b : ?tb) : [d] whereReturnType(ta,tb) throws
    {
        var c = makeDistArray(d, whereReturnType(ta,tb));
        if numLocales == 1 { 
           forall (ch, A, C) in zip(condition, a, c) { 
                C = if ch then A:whereReturnType(ta,tb) else b:whereReturnType(ta,tb) ;
           }    
        } else {
            coforall loc in Locales do on loc {
                var cLD = c.localSubdomain() ;   // they all have the same domain
                forall idx in cLD { 
                   c[idx] = if condition[idx] then a[idx]:whereReturnType(ta,tb)
                                              else b:whereReturnType(ta,tb) ;
                }
            }
        }
        return c;
    }

    @arkouda.registerCommand(name="wherevs_float64")
    proc wherevsr ( condition: [?d] bool, a : [d] ?ta, b : real) : [d] real throws
    where (ta==real || ta==int || ta==uint || ta==bool)
    {
        return wherevsHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wherevs_int64")
    proc wherevsi ( condition: [?d] bool, a : [d] ?ta, b : int) : [d] whereReturnType(ta,int) throws 
    where (ta==real || ta==int || ta==uint || ta==bool)
    {
        return wherevsHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wherevs_uint64")
    proc wherevsu ( condition: [?d] bool, a : [d] ?ta, b : uint) : [d] whereReturnType(ta,uint) throws 
    where (ta==real || ta==int || ta==uint || ta==bool)
    {
        return wherevsHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wherevs_bool")
    proc wherevsb ( condition: [?d] bool, a : [d] ?ta, b : bool) : [d] whereReturnType(ta,bool) throws 
    where (ta==real || ta==int || ta==uint || ta==bool)
    {
        return wherevsHelper (condition, a, b);
    }

    proc wheresvHelper (condition: [?d] bool, a : ?ta, b : [d] ?tb) : [d] whereReturnType(ta,tb) throws
    {
        var c = makeDistArray(d, whereReturnType(ta,tb));
        if numLocales == 1 { 
           forall (ch, B, C) in zip(condition, b, c) { 
                C = if ch then a:whereReturnType(ta,tb) else B:whereReturnType(ta,tb) ;
           }    
        } else {
            coforall loc in Locales do on loc {
                var cLD = c.localSubdomain() ;   // they all have the same domain
                forall idx in cLD { 
                   c[idx] = if condition[idx] then a:whereReturnType(ta,tb)
                                              else b[idx]:whereReturnType(ta,tb) ;
                }
            }
        }
        return c;
    }

    @arkouda.registerCommand(name="wheresv_float64")
    proc wheresvr ( condition: [?d] bool, a : real, b : [d] ?tb) : [d] real throws 
    where (tb==real || tb==int || tb==uint || tb==bool)
    {
        return wheresvHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheresv_int64")
    proc wheresvi ( condition: [?d] bool, a : int, b : [d] ?tb) : [d] whereReturnType(tb,int) throws 
    where (tb==real || tb==int || tb==uint || tb==bool)
    {
        return wheresvHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheresv_uint64")
    proc wheresvu ( condition: [?d] bool, a : uint, b : [d] ?tb) : [d] whereReturnType(tb,uint) throws 
    where (tb==real || tb==int || tb==uint || tb==bool)
    {
        return wheresvHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheresv_bool")
    proc wheresvb ( condition: [?d] bool, a : bool, b : [d] ?tb) : [d] whereReturnType(tb,bool) throws 
    where (tb==real || tb==int || tb==uint || tb==bool)
    {
        return wheresvHelper (condition, a, b);
    }

    proc wheressHelper (condition: [?d] bool, a : ?ta, b : ?tb) : [d] whereReturnType(ta,tb) throws
    {
        var c = makeDistArray(d, whereReturnType(ta,tb));
        if numLocales == 1 { 
           forall (ch, C) in zip(condition, c) { 
                C = if ch then a:whereReturnType(ta,tb) else b:whereReturnType(ta,tb) ;
           }    
        } else {
            coforall loc in Locales do on loc {
                var cLD = c.localSubdomain() ;   // they all have the same domain
                forall idx in cLD { 
                   c[idx] = if condition[idx] then a:whereReturnType(ta,tb)
                                              else b:whereReturnType(ta,tb) ;
                }
            }
        }
        return c;
    }

    @arkouda.registerCommand(name="wheress_float64_float64")
    proc wheressrr ( condition: [?d] bool, a : real, b : real) : [d] real throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_float64_int64")
    proc wheressri ( condition: [?d] bool, a : real, b : int) : [d] real throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_float64_uint64")
    proc wheressru ( condition: [?d] bool, a : real, b : uint) : [d] real throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_float64_bool")
    proc wheressrb ( condition: [?d] bool, a : real, b : bool) : [d] real throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_int64_float64")
    proc wheressir ( condition: [?d] bool, a : int, b : real) : [d] real throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_int64_int64") 
    proc wheressii ( condition: [?d] bool, a : int, b : int) : [d] int throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_int64_uint64") 
    proc wheressiu ( condition: [?d] bool, a : int, b : uint) : [d] real throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_int64_bool") 
    proc wheressib ( condition: [?d] bool, a : int, b : bool) : [d] int throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_uint64_float64") 
    proc wheressur ( condition: [?d] bool, a : uint, b : real) : [d] real throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_uint64_int64")
    proc wheressui ( condition: [?d] bool, a : uint, b : int) : [d] real throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_uint64_uint64")
    proc wheressuu ( condition: [?d] bool, a : uint, b : uint) : [d] uint throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_uint64_bool")
    proc wheressub ( condition: [?d] bool, a : uint, b : bool) : [d] uint throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_bool_float64")
    proc wheressbr ( condition: [?d] bool, a : bool, b : real) : [d] real throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_bool_int64")
    proc wheressbi ( condition: [?d] bool, a : bool, b : int) : [d] int throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_bool_uint64")
    proc wheressbu ( condition: [?d] bool, a : bool, b : uint) : [d] uint throws
    {
        return wheressHelper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_bool_bool")
    proc wheressbb ( condition: [?d] bool, a : bool, b : bool) : [d] bool throws
    {
        return wheressHelper (condition, a, b);
    }

  */  

    //  putmask has been rewritten for both the new interface and the multi-dimensional case.
    //  The specifics are based on the observation that np.putmask behaves as if multi-dim
    //  inputs were flattened before the operation, and then de-flattened after.

   @arkouda.registerCommand()
   proc putmask (mask : [?d1] bool, ref a : [d1] ?ta, v : [?d2] ?tv ) throws
       where   (ta==real ||
               (ta==int  && (tv==int  || tv==uint || tv==bool)) ||
               (ta==uint && (tv==uint || tv==bool)) ||
               (ta==bool &&  tv==bool ) ) {

        // By using indexToOrder, below, this proc is generic for any rank matrices.

        // Re the "de-flattened" comment above:

        //    "np.putmask behaves as if multi-dim inputs were flattened before the operation."
        //    Multi-dimensional a, mask, and v are stepped through as if one-dimensional.

        //    i.e., the nth element of mask determines whether the nth element of a will be
        //    overwritten with the nth element of v (v doesn't actually have to be the same
        //    size as a and mask, but hold that thought for now).

        //    As with np.putmask, we require a and mask to have the same domain, so that
        //    they'll be distributed identically. We don't know what that distribution
        //    will be, and it may vary from locale to locale.

        //    Further comments are embedded in the code.

       if mask.shape != a.shape {
           throw new Error ("mask and a pdarrays must be of same shape in putmask.") ;
       }

       var aR = new orderer (d1.shape) ;  // gives us indexToOrder for a

       // If there's only one locale, then there's no sense worrying about distributions

       // The loop below will use indexToOrder and orderToIndex to step through a (and mask)
       // in lock-step with v.

       if numLocales == 1 {
           forall element in d1 {
               if mask(element) {
                   const idx = aR.indexToOrder(element)%v.size ;
                   a(element) = v(d2.orderToIndex(idx)):ta ;
               }
           }

       // But if there's more than one locale, it's a bit more involved.  Decisions about
       // copying and/or aggregating v must be made on a per-locale basis.

       } else {
           coforall loc in Locales do on loc {
              var aLD = a.localSubdomain() ;

              // if all of v is smaller than aLD, just bring all of v over to this locale

              if v.size <= aLD.size {
                 var lv : [0..v.size-1] tv ; // a 1-D local v simplifies things in this case
                 forall entry in 0..v.size-1 with (var agg = new SrcAggregator(tv)) {
                    agg.copy (lv(entry),v(d2.orderToIndex(entry))) ; // lv = local copy of all v
                 }
                 forall element in aLD {
                    if mask(element) {      // indexToOrder is used just as in the 1-D case
                        a(element) = lv(aR.indexToOrder(element)%v.size):ta ;
                    }
                 }

              // but if v is larger than the local subdomain of a, just bring the part we need.
              // Since this subset of v will be the same size as the local part of a, we can
              // arbitrarily give it the same domain as aLD.

              } else {
                 var lv : [aLD] tv ;  // the local part of v
                 forall element in aLD with (var agg = newSrcAggregator(tv)) {
                    const vdx = aR.indexToOrder(element)%v.size ;
                    agg.copy (lv(element),v(d2.orderToIndex(vdx))) ;
                 }
                 forall element in aLD { if mask(element) then a(element) = lv(element):ta ; }
              }
           }
       }
                 
       return ;
   }

    private proc square(x) do return x * x;
    private proc log1p(x: real):real do return log(1.0 + x);
    private proc expm1(x: real):real do return exp(x) - 1.0;

   // Several functions below (m, indx, eclipse and gee) are needed in quantile & percentile,
   // to support the calculation of the indices j and j1, that are then used to compute the result.

   // "nearest" is not implemented because the numpy version doesn't appear to match its
   // documentation.  It's included here, in comments, to show that we are aware of it.

   proc m (method: string, q: real) : real {
        select (method) {
            when "inverted_cdf"              { return 0.0; }
            when "averaged_inverted_cdf"     { return 0.0; }
            when "closest_observation"       { return -0.5; }
            when "interpolated_inverted_cdf" { return 0.0; }
            when "hazen"                     { return 0.5; }
            when "weibull"                   { return q; }
            when "linear"                    { return 1 - q; }
            when "median_unbiased"           { return q/3.0 + 1.0/3.0; }
            when "normal_unbiased"           { return q/4.0 + 0.375; }
            when "lower"                     { return 1 - q; }
            when "midpoint"                  { return 1 - q; }
            when "higher"                    { return 1 - q;}
            // when "nearest"                { return 1 - q; } // not implementing this one yet
            otherwise                        { return 1 - q; }
        }
    }

    proc indx(q : real, n : int, m : real) : real {
        return q*n + m - 1;
    }

    proc frac (a: real) : real {
        return a - floor(a) ;
    }

    proc g(method: string, q: real, n: int, j:int) : real {
        var emm = m(method,q);
        select (method) {
            when "inverted_cdf"             { return (indx(q,n,emm) - j > 0):int;  }
            when "averaged_inverted_cdf"    { var interim = (indx(q,n,emm) - j > 0):int; return (1.0 + interim)/2.0; }
            when "closest_observation"      { return 1.0 - ((indx(q,n,emm) == j) & (j%2==1)):int; }
            when "interplated_inverted_cdf" { return frac(q*n + emm - 1.0); }
            when "hazen"                    { return frac(q*n + emm - 1.0); }
            when "weibull"                  { return frac(q*n + emm - 1.0); }
            when "linear"                   { return frac(q*n + emm - 1.0); }
            when "median_unbiased"          { return frac(q*n + emm - 1.0); }
            when "normal_unbiased"          { return frac(q*n + emm - 1.0); }
            when "lower"                    { return 0.0; }
            when "midpoint"                 { return 0.5; }
            when "higher"                   { return 1.0; }
            // when "nearest"               { return 1 - q; } // not implementing this one yet
            otherwise                       { return frac(q*n + emm -1.0); }
        }
    }

   proc eclipse (a : ?, lo : ?, high : ?) : real {
            return if a < lo then lo else if a > high then high else a;
        }

   proc supports_integers(method: string) : bool {
        select (method) {
            when "inverted_cdf"             { return false; }
            when "averaged_inverted_cdf"    { return false; }
            when "closest_observation"      { return false; }
            otherwise                       { return true; }
        }
    }
        
   proc discontinuous(method: string) : bool {
        select (method) {
            when "lower"       { return true; }
            when "midpoint"    { return true; }
            // when "nearest"  { return true; }  // not implementing this one yet
            when "higher"      { return true; }
            otherwise          { return false; }
        }
    }

   //  quantile and percentile can be called with q as a scalar or an array, and with/without
   //  axes specified along which to slice.

   //  Regardless of which instance was called (scalar, array, slice, no slice), the computation
   //  is done in quantile_helper.  The pdarray a must already be sorted and flattened before
   //  quantile_helper is invoked.

   //  The specifics come from a combination of studying the np.quantile documentation
   //  and reverse engineering what it actually does.

   proc quantile_helper (in a : [?d] ?t, q : real, method : string) : real throws
        where (d.rank == 1 && (t==real || t== int || t==uint)) {
        const n = d.size;
        var emm = m(method,q);
        var idx = indx(q,n,emm);
        var interim_j = q*n + emm - 1.0;
        if discontinuous (method) then interim_j = q*(n-1.0);
        var j = eclipse(interim_j,0,n-1):int;
        var mess = interim_j ;
        if supports_integers(method) {
            mess += (interim_j != floor(interim_j)):int ;
        } else {
            mess += 1.0;
        } 
        var j1 = eclipse(mess,0,n-1):int;
        var gee = g(method,q,n,j) ;
        return (1.0 - gee)*a[j] + gee*a[j1];
    }
        

   //  1st case is q scalar, no axis slicing.  The result is a scalar.

   @arkouda.registerCommand 
   proc quantile_scalar_no_axis (in a : [?d] ?t, q : real, method : string) : real throws
      where ((t==real || t==int || t==uint)) {
        return quantile_helper(sort(flatten(a)),q,method);
    }

   // 2nd case is q a pdarray, still no axis slicing.  The result is a pdarray of the same
   // shape as q.

   @arkouda.registerCommand
   proc quantile_array_no_axis (in a : [?d] ?t, q : [?dq] real, method : string) : [dq] real throws
      where ((t==real || t==int || t==uint)) {
        var a_ = sort(flatten(a));
        var return_value = makeDistArray(dq,real);
        forall dqidx in dq {
            return_value[dqidx] = quantile_helper(a_,q[dqidx],method);
        }
        return return_value;
    }

    // 3rd case is q scalar, with axis slicing.  The result is a pdarray of rank 1 less than the
    // rank of a.  A temporary result will be passed of the same shape as a.  The degenerate rank
    // will be removed python-side using the squeeze function.

    @arkouda.registerCommand
    proc quantile_scalar_with_axis (in a: [?d] ?t, q : real, axis: list(int), method : string) : [] real throws
        where ((t==real || t==int || t==uint)) {
        const (valid, axes) = validateNegativeAxes(axis, d.rank);
        if !valid {
            throw new Error("Invalid axis value(s) '%?' in quantile reduction".format(axis));
        } else {
            const outShape = reducedShape(a.shape, axes);
            var ret = makeDistArray((...outShape), real);
            for (sliceDom,sliceIdx) in axisSlices(d, axes) {

                var holder = makeDistArray(sliceDom.size,t);
                forall idx in holder.domain with (var agg = new DstAggregator(t)) {
                   agg.copy (holder(idx),a[sliceDom.orderToIndex(idx)]) ;
                }
                ret [sliceIdx] = quantile_helper(sort(flatten(holder)),q,method);
            }
            return ret;
        }
    }

    // 4th case is where q is a pdarray, with axis slicing.  This will return a pdarray of rank
    // q.ndim + a.ndim, and shape (q.shape,a.shape).  The sliced axes will be removed python-side
    // using the squeeze function.

    // growShape will append one shape to another.  This is needed only in this
    // specific case.

    proc growShape(qShape: ?Nq*int, aShape : ?Na*int): (Na+Nq)*int
        where Na >= 1 && Nq >= 1 {
        var grownShape : (Na+Nq)*int;
        for i in 0..(Nq-1) do grownShape[i] = qShape[i];
        for i in 0..(Na-1) do grownShape[Nq+i] = aShape[i];
        return grownShape;
    }

    @arkouda.registerCommand
    proc quantile_array_with_axis (in a: [?d] ?t, q: [?dq] real, axis: list(int), method : string) : [] real throws
        where ((t==real || t==int || t==uint)) {
        const (valid, axes) = validateNegativeAxes(axis, d.rank);
        if !valid {
            throw new Error("Invalid axis value(s) '%?' in quantile reduction".format(axis));
        } else {

            const tmpShape = reducedShape(a.shape, axes);
            var tmpret = makeDistArray((...tmpShape), real);
            var ret = makeDistArray((...growShape(q.shape,tmpShape)),real);
            for (sliceDom,sliceIdx) in axisSlices(d, axes) {

                var holder = makeDistArray(sliceDom.size,t);
                forall idx in holder.domain with (var agg = new DstAggregator(t)) {
                    agg.copy (holder(idx),a[sliceDom.orderToIndex(idx)]) ;
                }

                // The "holder" vector (the slice) was formed before looping over q.
                // This makes for less computation than calculating the vector slice within
                // the loop over q.

                for qIdx in q.domain {
                    var retIdx : (ret.rank)*int; // combination of qIdx and sliceIdx
                    if q.rank == 1 {
                        retIdx[0] = qIdx;
                    } else {
                        for i in 0..q.rank-1 do retIdx[i] = qIdx[i];
                    }
                    if sliceDom.rank == 1 {
                        retIdx [q.rank] = sliceIdx;
                    } else {
                        for i in 0..sliceDom.rank-1 do retIdx[q.rank+i] = sliceIdx[i];
                    }
                    ret [retIdx] = quantile_helper(sort(holder),q[qIdx],method);
                }
            }
            return ret ;
        }
    }

}

