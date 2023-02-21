
module EfuncMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
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

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const eLogger = new Logger(logLevel, logChannel);
    
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

    proc efuncMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
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
                var e = toSymEntry(gEnt,int);
                select efunc
                {
                    when "abs" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = abs(e.a);
                    }
                    when "log" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = log(e.a);
                    }
                    when "exp" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = exp(e.a);
                    }
                    when "cumsum" {
                        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                        overMemLimit(numBytes(int) * e.size);
                        st.addEntry(rname, new shared SymEntry(+ scan e.a));
                    }
                    when "cumprod" {
                        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                        overMemLimit(numBytes(int) * e.size);
                        st.addEntry(rname, new shared SymEntry(* scan e.a));
                    }
                    when "sin" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = sin(e.a);
                    }
                    when "cos" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = cos(e.a);
                    }
                    when "hash64" {
                        overMemLimit(numBytes(int) * e.size);
                        var a = st.addEntry(rname, e.size, uint);
                        forall (ai, x) in zip(a.a, e.a) {
                            ai = sipHash64(x): uint;
                        }
                    }
                    when "hash128" {
                        overMemLimit(numBytes(int) * e.size * 2);
                        var rname2 = st.nextName();
                        var a1 = st.addEntry(rname2, e.size, uint);
                        var a2 = st.addEntry(rname, e.size, uint);
                        forall (a1i, a2i, x) in zip(a1.a, a2.a, e.a) {
                            (a1i, a2i) = sipHash128(x): (uint, uint);
                        }
                        // Put first array's attrib in repMsg and let common
                        // code append second array's attrib
                        repMsg += "created " + st.attrib(rname2) + "+";
                    }
                    when "popcount" {
                        var a = st.addEntry(rname, e.size, int);
                        a.a = popcount(e.a);
                    }
                    when "parity" {
                        var a = st.addEntry(rname, e.size, int);
                        a.a = parity(e.a);
                    }
                    when "clz" {
                        var a = st.addEntry(rname, e.size, int);
                        a.a = clz(e.a);
                    }
                    when "ctz" {
                        var a = st.addEntry(rname, e.size, int);
                        a.a = ctz(e.a);
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,gEnt.dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                                               
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Float64) {
                var e = toSymEntry(gEnt,real);
                select efunc
                {
                    when "abs" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = abs(e.a);
                    }
                    when "log" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = log(e.a);
                    }
                    when "exp" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = exp(e.a);
                    }
                    when "cumsum" {
                        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                        overMemLimit(numBytes(real) * e.size);
                        st.addEntry(rname, new shared SymEntry(+ scan e.a));
                    }
                    when "cumprod" {
                        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                        overMemLimit(numBytes(real) * e.size);
                        st.addEntry(rname, new shared SymEntry(* scan e.a));
                    }
                    when "sin" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = sin(e.a);
                    }
                    when "cos" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = cos(e.a);
                    }
                    when "isnan" {
                        var a = st.addEntry(rname, e.size, bool);
                        a.a = isnan(e.a);
                    }
                    when "hash64" {
                        overMemLimit(numBytes(real) * e.size);
                        var a = st.addEntry(rname, e.size, uint);
                        forall (ai, x) in zip(a.a, e.a) {
                            ai = sipHash64(x): uint;
                        }
                    }
                    when "hash128" {
                        overMemLimit(numBytes(real) * e.size * 2);
                        var rname2 = st.nextName();
                        var a1 = st.addEntry(rname2, e.size, uint);
                        var a2 = st.addEntry(rname, e.size, uint);
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
                var e = toSymEntry(gEnt,bool);
                select efunc
                {
                    when "cumsum" {
                        var ia: [e.a.domain] int = (e.a:int); // make a copy of bools as ints blah!
                        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                        overMemLimit(numBytes(int) * ia.size);
                        st.addEntry(rname, new shared SymEntry(+ scan ia));
                    }
                    when "cumprod" {
                        var ia: [e.a.domain] int = (e.a:int); // make a copy of bools as ints blah!
                        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                        overMemLimit(numBytes(int) * ia.size);
                        st.addEntry(rname, new shared SymEntry(* scan ia));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,gEnt.dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                        
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.UInt64) {
                var e = toSymEntry(gEnt,uint);
                select efunc
                {
                    when "popcount" {
                        var a = st.addEntry(rname, e.size, uint);
                        a.a = popcount(e.a);
                    }
                    when "clz" {
                        var a = st.addEntry(rname, e.size, uint);
                        a.a = clz(e.a);
                    }
                    when "ctz" {
                        var a = st.addEntry(rname, e.size, uint);
                        a.a = ctz(e.a);
                    }
                    when "cumsum" {
                        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                        overMemLimit(numBytes(uint) * e.size);
                        st.addEntry(rname, new shared SymEntry(+ scan e.a));
                    }
                    when "cumprod" {
                        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                        overMemLimit(numBytes(uint) * e.size);
                        st.addEntry(rname, new shared SymEntry(* scan e.a));
                    }
                    when "sin" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = sin(e.a);
                    }
                    when "cos" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = cos(e.a);
                    }
                    when "parity" {
                        var a = st.addEntry(rname, e.size, uint);
                        a.a = parity(e.a);
                    }
                    when "hash64" {
                        overMemLimit(numBytes(uint) * e.size);
                        var a = st.addEntry(rname, e.size, uint);
                        forall (ai, x) in zip(a.a, e.a) {
                            ai = sipHash64(x): uint;
                        }
                    }
                    when "hash128" {
                        overMemLimit(numBytes(uint) * e.size * 2);
                        var rname2 = st.nextName();
                        var a1 = st.addEntry(rname2, e.size, uint);
                        var a2 = st.addEntry(rname, e.size, uint);
                        forall (a1i, a2i, x) in zip(a1.a, a2.a, e.a) {
                            (a1i, a2i) = sipHash128(x): (uint, uint);
                        }
                        // Put first array's attrib in repMsg and let common
                        // code append second array's attrib
                        repMsg += "created " + st.attrib(rname2) + "+";
                    }
                    when "log" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = log(e.a);
                    }
                    when "exp" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = exp(e.a);
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
    proc efunc3vvMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var rname = st.nextName();
        
        var efunc = msgArgs.getValueOf("func");
        var g1: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("condition"), st);
        var g2: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("a"), st);
        var g3: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("b"), st);
        if !((g1.size == g2.size) && (g2.size == g3.size)) {
            var errorMsg = "size mismatch in arguments to "+pn;
            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        }
        select (g1.dtype, g2.dtype, g3.dtype) {
            when (DType.Bool, DType.Int64, DType.Int64) {
                var e1 = toSymEntry(g1, bool);
                var e2 = toSymEntry(g2, int);
                var e3 = toSymEntry(g3, int);
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, e2.a, e3.a, 0);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                           g2.dtype,g3.dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg, MsgType.ERROR); 
                    }                
                } 
            }
            when (DType.Bool, DType.UInt64, DType.UInt64) {
                var e1 = toSymEntry(g1, bool);
                var e2 = toSymEntry(g2, uint);
                var e3 = toSymEntry(g3, uint);
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, e2.a, e3.a, 0);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                           g2.dtype,g3.dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg, MsgType.ERROR); 
                    }                
                } 
            }
            when (DType.Bool, DType.Float64, DType.Float64) {
                var e1 = toSymEntry(g1, bool);
                var e2 = toSymEntry(g2, real);
                var e3 = toSymEntry(g3, real);
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, e2.a, e3.a, 0);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                       g2.dtype,g3.dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                } 
            }
            when (DType.Bool, DType.Bool, DType.Bool) {
                var e1 = toSymEntry(g1, bool);
                var e2 = toSymEntry(g2, bool);
                var e3 = toSymEntry(g3, bool);
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, e2.a, e3.a, 0);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                       g2.dtype,g3.dtype);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                                                      
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                } 
            }
            otherwise {
               var errorMsg = notImplementedError(pn,efunc,g1.dtype,g2.dtype,g3.dtype);
               eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);       
               return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }

        repMsg = "created " + st.attrib(rname);
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
        return new MsgTuple(repMsg, MsgType.NORMAL); 
    }

    /*
    vector = efunc(vector, vector, scalar)

    :arg reqMsg: request containing (cmd,efunc,name1,name2,dtype,value)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (MsgTuple)
    :throws: `UndefinedSymbolError(name)`
    */
    proc efunc3vsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
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
        if !(g1.size == g2.size) {
            var errorMsg = "size mismatch in arguments to "+pn;
            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);  
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        select (g1.dtype, g2.dtype, dtype) {
            when (DType.Bool, DType.Int64, DType.Int64) {
               var e1 = toSymEntry(g1, bool);
               var e2 = toSymEntry(g2, int);
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
               var e1 = toSymEntry(g1, bool);
               var e2 = toSymEntry(g2, uint);
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
                var e1 = toSymEntry(g1, bool);
                var e2 = toSymEntry(g2, real);
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
                var e1 = toSymEntry(g1, bool);
                var e2 = toSymEntry(g2, bool);
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
    proc efunc3svMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
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
        if !(g1.size == g2.size) {
            var errorMsg = "size mismatch in arguments to "+pn;
            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);            
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        select (g1.dtype, dtype, g2.dtype) {
            when (DType.Bool, DType.Int64, DType.Int64) {
                var e1 = toSymEntry(g1, bool);
                var val = msgArgs.get("scalar").getIntValue();
                var e2 = toSymEntry(g2, int);
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
                var e1 = toSymEntry(g1, bool);
                var val = msgArgs.get("scalar").getUIntValue();
                var e2 = toSymEntry(g2, uint);
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
                var e1 = toSymEntry(g1, bool);
                var val = msgArgs.get("scalar").getRealValue();
                var e2 = toSymEntry(g2, real);
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
                var e1 = toSymEntry(g1, bool);
                var val = msgArgs.get("scalar").getBoolValue();
                var e2 = toSymEntry(g2, bool);
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
    proc efunc3ssMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
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
                var e1 = toSymEntry(g1, bool);
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
                var e1 = toSymEntry(g1, bool);
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
                var e1 = toSymEntry(g1, bool);
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
                var e1 = toSymEntry(g1, bool);
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

    proc efuncArrMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var n = msgArgs.get("length").getIntValue();
        var s = msgArgs.get("size").getIntValue();
        var namesList = msgArgs.get("nameslist").getList(n);
        var typesList = msgArgs.get("typeslist").getList(n);
        var (size, hasStr, names, types) = validateArraysSameLength(n, namesList, typesList, st);

        // Call hashArrays on list of given array names
        var hashes = hashArrays(size, names, types, st);

        // Init upper and lower 64-bit entries
        var upper = new shared SymEntry(s, uint);
        var lower = new shared SymEntry(s, uint);

        // Assign upper and lower bit values to their respective entries
        forall (i, (up, low)) in zip(0..#s, hashes) {
            upper.a[i] = up;
            lower.a[i] = low;
        }

        var upperName = st.nextName();
        st.addEntry(upperName, upper);
        var lowerName = st.nextName();
        st.addEntry(lowerName, lower);

        var repMsg = "created %s+created %s".format(st.attrib(upperName), st.attrib(lowerName));
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
        return new MsgTuple(repMsg, MsgType.NORMAL);
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
    proc where_helper(cond:[?D] bool, A:[D] ?t, B:[D] t, param kind):[D] t where (kind == 0) {
      var C:[D] t;
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
    proc where_helper(cond:[?D] bool, A:[D] ?t, b:t, param kind):[D] t where (kind == 1) {
      var C:[D] t;
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
    proc where_helper(cond:[?D] bool, a:?t, B:[D] t, param kind):[D] t where (kind == 2) {
      var C:[D] t;
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
    proc where_helper(cond:[?D] bool, a:?t, b:t, param kind):[D] t where (kind == 3) {
      var C:[D] t;
      forall (ch, c) in zip(cond, C) {
        c = if ch then a else b;
      }
      return C;
    }    

    use CommandMap;
    registerFunction("efunc", efuncMsg, getModuleName());
    registerFunction("efunc3vv", efunc3vvMsg, getModuleName());
    registerFunction("efunc3vs", efunc3vsMsg, getModuleName());
    registerFunction("efunc3sv", efunc3svMsg, getModuleName());
    registerFunction("efunc3ss", efunc3ssMsg, getModuleName());
    registerFunction("efuncArr", efuncArrMsg, getModuleName());
}