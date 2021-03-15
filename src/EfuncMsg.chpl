
module EfuncMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    use Reflection;
    use Errors;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    
    use AryUtil;
    
    const eLogger = new Logger();

    if v {
        eLogger.level = LogLevel.DEBUG;
    } else {
        eLogger.level = LogLevel.INFO;
    }
    
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

    proc efuncMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (efunc, name) = payload.splitMsgToTuple(2);
        var rname = st.nextName();
        
        var gEnt: borrowed GenSymEntry = st.lookup(name);
        
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                           "cmd: %s efunc: %s pdarray: %s".format(cmd,efunc,st.attrib(name)));
       
        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                select efunc
                {
                    when "abs" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = Math.abs(e.a);                      
                    }
                    when "log" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = Math.log(e.a);
                    }
                    when "exp" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = Math.exp(e.a);
                    }
                    when "cumsum" {
                        var a = st.addEntry(rname, e.size, int);
                        a.a = + scan e.a;
                    }
                    when "cumprod" {
                        var a= st.addEntry(rname, e.size, int);
                        a.a = * scan e.a;
                    }
                    when "sin" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = Math.sin(e.a);
                    }
                    when "cos" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = Math.cos(e.a);
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
                        a.a = Math.abs(e.a);
                    }
                    when "log" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = Math.log(e.a);
                    }
                    when "exp" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = Math.exp(e.a);
                    }
                    when "cumsum" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = + scan e.a;
                    }
                    when "cumprod" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = * scan e.a;
                    }
                    when "sin" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = Math.sin(e.a);
                    }
                    when "cos" {
                        var a = st.addEntry(rname, e.size, real);
                        a.a = Math.cos(e.a);
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
                        var ia: [e.aD] int = (e.a:int); // make a copy of bools as ints blah!
                        var a = st.addEntry(rname, e.size, int);
                        a.a = + scan ia;
                    }
                    when "cumprod" {
                        var ia: [e.aD] int = (e.a:int); // make a copy of bools as ints blah!
                        var a = st.addEntry(rname, e.size, int);
                        a.a = * scan ia;
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
    proc efunc3vvMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (efunc, name1, name2, name3) = payload.splitMsgToTuple(4);
        var rname = st.nextName();

        var g1: borrowed GenSymEntry = st.lookup(name1);
        var g2: borrowed GenSymEntry = st.lookup(name2);
        var g3: borrowed GenSymEntry = st.lookup(name3);
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
    proc efunc3vsMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var (efunc, name1, name2, dtypestr, value)
              = payload.splitMsgToTuple(5); // split request into fields
        var dtype = str2dtype(dtypestr);
        var rname = st.nextName();

        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s efunc: %s scalar: %s dtype: %s name1: %s name2: %s rname: %s".format(
             cmd,efunc,value,dtype,name1,name2,rname));

        var g1: borrowed GenSymEntry = st.lookup(name1);
        var g2: borrowed GenSymEntry = st.lookup(name2);
        if !(g1.size == g2.size) {
            var errorMsg = "size mismatch in arguments to "+pn;
            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);  
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        select (g1.dtype, g2.dtype, dtype) {
            when (DType.Bool, DType.Int64, DType.Int64) {
               var e1 = toSymEntry(g1, bool);
               var e2 = toSymEntry(g2, int);
               var val = try! value:int;
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
                var val = try! value:real;
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
                var val = try! value.toLower():bool;
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
    proc efunc3svMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var (efunc, name1, dtypestr, value, name2)
              = payload.splitMsgToTuple(5); // split request into fields
        var dtype = str2dtype(dtypestr);
        var rname = st.nextName();

        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s efunc: %s scalar: %s dtype: %s name1: %s name2: %s rname: %s".format(
             cmd,efunc,value,dtype,name1,name2,rname));

        var g1: borrowed GenSymEntry = st.lookup(name1);
        var g2: borrowed GenSymEntry = st.lookup(name2);
        if !(g1.size == g2.size) {
            var errorMsg = "size mismatch in arguments to "+pn;
            eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);            
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        select (g1.dtype, dtype, g2.dtype) {
            when (DType.Bool, DType.Int64, DType.Int64) {
                var e1 = toSymEntry(g1, bool);
                var val = try! value:int;
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
            when (DType.Bool, DType.Float64, DType.Float64) {
                var e1 = toSymEntry(g1, bool);
                var val = try! value:real;
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
                var val = try! value.toLower():bool;
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
    proc efunc3ssMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var (efunc, name1, dtype1str, value1, dtype2str, value2)
              = payload.splitMsgToTuple(6); // split request into fields
        var dtype1 = str2dtype(dtype1str);
        var dtype2 = str2dtype(dtype2str);
        var rname = st.nextName();
        
        eLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s efunc: %s scalar1: %s dtype1: %s scalar2: %s dtype2: %s name: %s rname: %s".format(
             cmd,efunc,value1,dtype1,value2,dtype2,name1,rname));

        var g1: borrowed GenSymEntry = st.lookup(name1);
        select (g1.dtype, dtype1, dtype1) {
            when (DType.Bool, DType.Int64, DType.Int64) {
                var e1 = toSymEntry(g1, bool);
                var val1 = try! value1:int;
                var val2 = try! value2:int;
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, val1, val2, 3);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                      dtype1,dtype2);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                } 
            }
            when (DType.Bool, DType.Float64, DType.Float64) {
                var e1 = toSymEntry(g1, bool);
                var val1 = try! value1:real;
                var val2 = try! value2:real;
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, val1, val2, 3);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                        dtype1,dtype2);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg, MsgType.ERROR);                                                     
                    }
                } 
            }
            when (DType.Bool, DType.Bool, DType.Bool) {
                var e1 = toSymEntry(g1, bool);
                var val1 = try! value1.toLower():bool;
                var val2 = try! value2.toLower():bool;
                select efunc {
                    when "where" {
                        var a = where_helper(e1.a, val1, val2, 3);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                                       dtype1,dtype2);
                        eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg, MsgType.ERROR);      
                   }
               } 
            }
            otherwise {
                var errorMsg = notImplementedError(pn,efunc,g1.dtype,
                                               dtype1,dtype2);
                eLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                return new MsgTuple(errorMsg, MsgType.ERROR);                                             
            }
        }

        repMsg = "created " + st.attrib(rname);
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

}
