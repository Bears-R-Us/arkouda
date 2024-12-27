
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

    // These ops are functions which take an array and produce an array.

    @arkouda.registerCommand (name="sin")
    proc sine (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    { 
        return sin(x);
    }
       
    @arkouda.registerCommand (name="cos")
    proc cosine (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return cos(x);
    }
       
    @arkouda.registerCommand (name="tan")
    proc tangent (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return tan(x);
    }
       
    @arkouda.registerCommand (name="arcsin")
    proc arcsine (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return asin(x);
    }
       
    @arkouda.registerCommand (name="arccos")
    proc arccosine (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return acos(x);
    }
       
    @arkouda.registerCommand (name="arctan")
    proc arctangent (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return atan(x);
    }
       
    @arkouda.registerCommand (name="sinh")
    proc hypsine (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return sinh(x);
    }
       
    @arkouda.registerCommand (name="cosh")
    proc hypcosine (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return cosh(x);
    }
       
    @arkouda.registerCommand (name="tanh")
    proc hyptangent (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return tanh(x);
    }
       
    @arkouda.registerCommand (name="arcsinh")
    proc archypsine (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return asinh(x);
    }
       
    @arkouda.registerCommand (name="arccosh")
    proc archypcosine (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return acosh(x);
    }
       
    @arkouda.registerCommand (name="arctanh")
    proc archyptangent (x : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return atanh(x);
    }
       
    @arkouda.registerCommand(name="abs")
    proc absolut (const ref pda : [?d] ?t) : [d] t throws
        where (t==int || t==real) // TODO maybe: allow uint also
    {
        return abs(pda);
    }

    @arkouda.registerCommand(name="square")
    proc boxy (const ref pda : [?d] ?t) : [d] t throws
        where (t==int || t==real || t==uint)
    {
        return square(pda);
    }

    @arkouda.registerCommand(name="exp")
    proc expo (const ref pda : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return exp(pda);
    }

    @arkouda.registerCommand(name="expm1")
    proc expom (const ref pda : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return expm1(pda);
    }

    @arkouda.registerCommand(name="log")
    proc log_e (const ref pda : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return log(pda);
    }

    @arkouda.registerCommand(name="log1p")
    proc log_1p (const ref pda : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return log1p(pda);
    }

    //  chapel log2 returns ints when given ints, so the input has been cast to real.

    @arkouda.registerCommand(name="log2")
    proc log_2 (const ref pda : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return log2(pda:real);
    }

    @arkouda.registerCommand(name="log10")
    proc log_10 (const ref pda : [?d] ?t) : [d] real throws
        where (t==int || t==real || t==uint)
    {
        return log10(pda);
    }

    @arkouda.registerCommand(name="isinf")
    proc isinf_ (pda : [?d] real) : [d] bool
    {
        return (isInf(pda));
    }

    @arkouda.registerCommand(name="isnan")
    proc isnan_ (pda : [?d] real) : [d] bool
    {
        return (isNan(pda));
    }

    @arkouda.registerCommand(name="isfinite")
    proc isfinite_ (pda : [?d] real) : [d] bool
    {
        return (isFinite(pda));
    }

    @arkouda.registerCommand (name="floor")
    proc floor_ (pda : [?d] ?t) : [d] real throws
        where (t==real)
    {
        return floor(pda);
    }

    @arkouda.registerCommand (name="ceil")
    proc ceil_ (pda : [?d] ?t) : [d] real throws
        where (t==real)
    {
        return ceil(pda);
    }

    @arkouda.registerCommand (name="round")
    proc round_ (pda : [?d] ?t) : [d] real throws
        where (t==real)
    {
        return round(pda);
    }

    @arkouda.registerCommand (name="trunc")
    proc trunc_ (pda : [?d] ?t) : [d] real throws
        where (t==real)
    {
        return trunc(pda);
    }

    @arkouda.registerCommand (name="popcount")
    proc popcount_ (pda : [?d] ?t) : [d] t throws
        where (t==int || t==uint)
    {
        return popCount(pda);
    }

    @arkouda.registerCommand (name="parity")
    proc parity_ (pda : [?d] ?t) : [d] t throws
        where (t==int || t==uint)
    {
        return parity(pda);
    }

    @arkouda.registerCommand (name="clz")
    proc clz_ (pda : [?d] ?t) : [d] t throws
        where (t==int || t==uint)
    {
        return clz(pda);
    }

    @arkouda.registerCommand (name="ctz")
    proc ctz_ (pda : [?d] ?t) : [d] t throws
        where (t==int || t==uint)
    {
        return ctz(pda);
    }

    @arkouda.registerCommand(name="not")
    proc not_ (pda : [?d] ?t) : [d] bool throws
        where (t==int || t==uint || t==bool)
    {
        return (!pda);
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

    @arkouda.registerCommand(name="cumsum")
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

    @arkouda.registerCommand(name="cumprod")
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
    proc sign (pda : [?d] ?t) : [d] int(8) throws
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

    @arkouda.registerCommand(name="fmod2vv")
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

    @arkouda.registerCommand(name="arctan2vv")
    proc arctangent2vv (a : [?d] ?ta, b : [d] ?tb) : [d] real throws
        where ( (ta==real && (tb==real || tb==int || tb==uint)) ||
                (ta==int  && (tb==real || tb==int || tb==uint)) ||
                (ta==uint && (tb==real || tb==int || tb==uint)) ) {
            return (atan2(a,b));
        }

    @arkouda.registerCommand(name="arctan2vs_float64")
    proc arctangent2vsr (a : [?d] ?ta, b : real) : [d] real throws
        where (ta==int || ta==uint || ta==real) {
            return (atan2(a,b));
        }

    @arkouda.registerCommand(name="arctan2vs_int64")
    proc arctangent2vsi (a : [?d] ?ta, b : int) : [d] real throws
        where (ta==int || ta==uint || ta==real) {
            return (atan2(a,b));
        }

    @arkouda.registerCommand(name="arctan2vs_uint64")
    proc arctangent2vsu (a : [?d] ?ta, b : uint) : [d] real throws
        where (ta==int || ta==uint || ta==real) {
            return (atan2(a,b));
        }

    @arkouda.registerCommand(name="arctan2sv_float64")
    proc arctangent2svr (a : real, b : [?d] ?tb) : [d] real throws
        where (tb==int || tb==uint || tb==real) {
            return (atan2(a,b));
        }

    @arkouda.registerCommand(name="arctan2sv_int64")
    proc arctangent2svi (a : int, b : [?d] ?tb) : [d] real throws
        where (tb==int || tb==uint || tb==real) {
            return (atan2(a,b));
        }

    @arkouda.registerCommand(name="arctan2sv_uint64")
    proc arctangent2svu (a : uint, b : [?d] ?tb) : [d] real throws
        where (tb==int || tb==uint || tb==real) {
            return (atan2(a,b));
        }

    // The above comment re scalar types allso applies to "where"

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

    @arkouda.registerCommand(name="wherevv")
    proc wherevv_ ( condition: [?d] bool, a : [d] ?ta, b : [d] ?tb) : [d] whereReturnType(ta,tb) throws
    where ((ta==real || ta==int || ta==uint || ta==bool) && (tb==real || tb==int || tb==uint || tb==bool))
    {
         var c = makeDistArray(d, whereReturnType(ta,tb));
         forall (ch, A, B, C) in zip(condition, a, b, c) {
             C = if ch then A:whereReturnType(ta,tb) else B:whereReturnType(ta,tb) ;
         }
         return c;
    }

    // The wherevs (vector, scalar), wheresv (scalar, vector) and wheress (scalar, scalar)
    // implementations all call corresponding helper functions.  This is done because the
    // arkouda interface requires the scalar types to be specified, but the helper functions,
    // since they're not registered as commands, can be more generic.

    proc wherevs_helper (condition: [?d] bool, a : [d] ?ta, b : ?tb) : [d] whereReturnType(ta,tb) throws
    {
        var c = makeDistArray(d, whereReturnType(ta,tb));
        forall (ch, A, C) in zip(condition, a, c) {
            C = if ch then A:whereReturnType(ta,tb) else b:whereReturnType(ta,tb);
        }
        return c;
    }

    @arkouda.registerCommand(name="wherevs_float64")
    proc wherevsr_ ( condition: [?d] bool, a : [d] ?ta, b : real) : [d] real throws
    where (ta==real || ta==int || ta==uint || ta==bool)
    {
        return wherevs_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wherevs_int64")
    proc wherevsi_ ( condition: [?d] bool, a : [d] ?ta, b : int) : [d] whereReturnType(ta,int) throws 
    where (ta==real || ta==int || ta==uint || ta==bool)
    {
        return wherevs_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wherevs_uint64")
    proc wherevsu_ ( condition: [?d] bool, a : [d] ?ta, b : uint) : [d] whereReturnType(ta,uint) throws 
    where (ta==real || ta==int || ta==uint || ta==bool)
    {
        return wherevs_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wherevs_bool")
    proc wherevsb_ ( condition: [?d] bool, a : [d] ?ta, b : bool) : [d] whereReturnType(ta,bool) throws 
    where (ta==real || ta==int || ta==uint || ta==bool)
    {
        return wherevs_helper (condition, a, b);
    }

    proc wheresv_helper (condition: [?d] bool, a : ?ta, b : [d] ?tb) : [d] whereReturnType(ta,tb) throws
    {
        var c = makeDistArray(d, whereReturnType(ta,tb));
        forall (ch, B, C) in zip(condition, b, c) {
            C = if ch then a:whereReturnType(ta,tb) else B:whereReturnType(ta,tb);
        }
        return c;
    }

    @arkouda.registerCommand(name="wheresv_float64")
    proc wheresvr_ ( condition: [?d] bool, a : real, b : [d] ?tb) : [d] real throws 
    where (tb==real || tb==int || tb==uint || tb==bool)
    {
        return wheresv_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheresv_int64")
    proc wheresvi_ ( condition: [?d] bool, a : int, b : [d] ?tb) : [d] whereReturnType(tb,int) throws 
    where (tb==real || tb==int || tb==uint || tb==bool)
    {
        return wheresv_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheresv_uint64")
    proc wheresvu_ ( condition: [?d] bool, a : uint, b : [d] ?tb) : [d] whereReturnType(tb,uint) throws 
    where (tb==real || tb==int || tb==uint || tb==bool)
    {
        return wheresv_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheresv_bool")
    proc wheresvb_ ( condition: [?d] bool, a : bool, b : [d] ?tb) : [d] whereReturnType(tb,bool) throws 
    where (tb==real || tb==int || tb==uint || tb==bool)
    {
        return wheresv_helper (condition, a, b);
    }

    proc wheress_helper (condition: [?d] bool, a : ?ta, b : ?tb) : [d] whereReturnType(ta,tb) throws
    {
        var c = makeDistArray(d, whereReturnType(ta,tb));
        forall (ch, C) in zip(condition, c) {
            C = if ch then a:whereReturnType(ta,tb) else b:whereReturnType(ta,tb);
        }
        return c;
    }

    @arkouda.registerCommand(name="wheress_float64_float64")
    proc wheress_rr ( condition: [?d] bool, a : real, b : real) : [d] real throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_float64_int64")
    proc wheress_ri ( condition: [?d] bool, a : real, b : int) : [d] real throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_float64_uint64")
    proc wheress_ru ( condition: [?d] bool, a : real, b : uint) : [d] real throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_float64_bool")
    proc wheress_rb ( condition: [?d] bool, a : real, b : bool) : [d] real throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_int64_float64")
    proc wheress_ir ( condition: [?d] bool, a : int, b : real) : [d] real throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_int64_int64") 
    proc wheress_ii ( condition: [?d] bool, a : int, b : int) : [d] int throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_int64_uint64") 
    proc wheress_iu ( condition: [?d] bool, a : int, b : uint) : [d] real throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_int64_bool") 
    proc wheress_ib ( condition: [?d] bool, a : int, b : bool) : [d] int throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_uint64_float64") 
    proc wheress_ur ( condition: [?d] bool, a : uint, b : real) : [d] real throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_uint64_int64")
    proc wheress_ui ( condition: [?d] bool, a : uint, b : int) : [d] real throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_uint64_uint64")
    proc wheress_uu ( condition: [?d] bool, a : uint, b : uint) : [d] uint throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_uint64_bool")
    proc wheress_ub ( condition: [?d] bool, a : uint, b : bool) : [d] uint throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_bool_float64")
    proc wheress_br ( condition: [?d] bool, a : bool, b : real) : [d] real throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_bool_int64")
    proc wheress_bi ( condition: [?d] bool, a : bool, b : int) : [d] int throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_bool_uint64")
    proc wheress_bu ( condition: [?d] bool, a : bool, b : uint) : [d] uint throws
    {
        return wheress_helper (condition, a, b);
    }

    @arkouda.registerCommand(name="wheress_bool_bool")
    proc wheress_bb ( condition: [?d] bool, a : bool, b : bool) : [d] bool throws
    {
        return wheress_helper (condition, a, b);
    }

    //  putmask has been rewritten for both the new interface and the multi-dimensional case.
    //  The specifics are based on the observation that np.putmask behaves as if multi-dim
    //  inputs were flattened before the operation, and then de-flattened after.

   @arkouda.registerCommand(name="putmask")
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

        //    In a departure from np.putmask, we require a and mask to have the same domain,
        //    so that they'll be distributed identically. We don't know what that distribution
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

}
