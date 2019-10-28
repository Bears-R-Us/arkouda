
module EfuncMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    use Reflection only;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    
    use AryUtil;
    
    /* These ops are functions which take an array and produce an array.
       
       **Dev Note:** Do scans fit here also? I think so... vector = scanop(vector)
       parse and respond to efunc "elemental function" message
       vector = efunc(vector) 
       
      :arg reqMsg: request containing (cmd,efunc,name)
      :type reqMsg: string 

      :arg st: SymTab to act on
      :type st: borrowed SymTab 

      :returns: (string)
      */

    proc efuncMsg(reqMsg: string, st: borrowed SymTab): string {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var efunc = fields[2];
        var name = fields[3];
        var rname = st.nextName();
        if v {try! writeln("%s %s %s : %s".format(cmd,efunc,name,rname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError(pn,name);}
       
        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                select efunc
                {
                    when "abs" {
                        var a = Math.abs(e.a);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "log" {
                        var a = Math.log(e.a);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "exp" {
                        var a = Math.exp(e.a);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "cumsum" {
                        var a: [e.aD] int = + scan e.a; //try! writeln((a.type):string,(a.domain.type):string); try! stdout.flush();
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "cumprod" {
                        var a: [e.aD] int = * scan e.a; //try! writeln((a.type):string,(a.domain.type):string); try! stdout.flush();
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "sin" {
                        var a = Math.sin(e.a);
                        st.addEntry(rname,new shared SymEntry(a));
                    }
                    when "cos" {
                        var a = Math.cos(e.a);
                        st.addEntry(rname,new shared SymEntry(a));
                    }
                    otherwise {return notImplementedError(pn,efunc,gEnt.dtype);}
                }
            }
            when (DType.Float64) {
                var e = toSymEntry(gEnt,real);
                select efunc
                {
                    when "abs" {
                        var a = Math.abs(e.a);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "log" {
                        var a = Math.log(e.a);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "exp" {
                        var a = Math.exp(e.a);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "cumsum" {
                        var a: [e.aD] real = + scan e.a;
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "cumprod" {
                        var a: [e.aD] real = * scan e.a;
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "sin" {
                        var a = Math.sin(e.a);
                        st.addEntry(rname,new shared SymEntry(a));
                    }
                    when "cos" {
                        var a = Math.cos(e.a);
                        st.addEntry(rname,new shared SymEntry(a));
                    }
                    otherwise {return notImplementedError(pn,efunc,gEnt.dtype);}
                }
            }
            when (DType.Bool) {
                var e = toSymEntry(gEnt,bool);
                select efunc
                {
                    when "cumsum" {
                        var ia: [e.aD] int = (e.a:int); // make a copy of bools as ints blah!
                        var a: [e.aD] int = + scan ia;
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "cumprod" {
                        var ia: [e.aD] int = (e.a:int); // make a copy of bools as ints blah!
                        var a: [e.aD] int = * scan ia;
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {return notImplementedError(pn,efunc,gEnt.dtype);}
                }
            }
            otherwise {return unrecognizedTypeError(pn, dtype2str(gEnt.dtype));}
        }
        return try! "created " + st.attrib(rname);
    }

    /*
    These are ternary functions which take three arrays and produce an array.
    vector = efunc(vector, vector, vector)

    :arg reqMsg: request containing (cmd,efunc,name1,name2,name3)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string)
    */
    proc efunc3vvMsg(reqMsg: string, st: borrowed SymTab): string {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var efunc = fields[2];
        var name1 = fields[3];
	var name2 = fields[4];
	var name3 = fields[5];
        var rname = st.nextName();
	if v {try! writeln("%s %s %s %s %s %s : %s".format(cmd,efunc,name1,name2,name3,rname));try! stdout.flush();}

        var g1: borrowed GenSymEntry = st.lookup(name1);
        if (g1 == nil) {return unknownSymbolError(pn,name1);}
	var g2: borrowed GenSymEntry = st.lookup(name2);
	if (g2 == nil) {return unknownSymbolError(pn,name2);}
	var g3: borrowed GenSymEntry = st.lookup(name3);
	if (g3 == nil) {return unknownSymbolError(pn,name3);}
	if !((g1.size == g2.size) && (g2.size == g3.size)) {
	  return "Error: size mismatch in arguments to "+pn;
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
	    otherwise {return notImplementedError(pn,efunc,g1.dtype,g2.dtype,g3.dtype);}
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
	    otherwise {return notImplementedError(pn,efunc,g1.dtype,g2.dtype,g3.dtype);}
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
	    otherwise {return notImplementedError(pn,efunc,g1.dtype,g2.dtype,g3.dtype);}
	    } 
	}
	otherwise {return notImplementedError(pn,efunc,g1.dtype,g2.dtype,g3.dtype);}
	}
	return try! "created " + st.attrib(rname);
    }

    /*
    vector = efunc(vector, vector, scalar)

    :arg reqMsg: request containing (cmd,efunc,name1,name2,dtype,value)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string)
    */
    proc efunc3vsMsg(reqMsg: string, st: borrowed SymTab): string {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var efunc = fields[2];
        var name1 = fields[3];
	var name2 = fields[4];
	var dtype = str2dtype(fields[5]);
	var value = fields[6];
        var rname = st.nextName();
	if v {try! writeln("%s %s %s %s %s %s %s : %s".format(cmd,efunc,name1,name2,dtype,value,rname));try! stdout.flush();}

        var g1: borrowed GenSymEntry = st.lookup(name1);
        if (g1 == nil) {return unknownSymbolError(pn,name1);}
	var g2: borrowed GenSymEntry = st.lookup(name2);
	if (g2 == nil) {return unknownSymbolError(pn,name2);}
	if !(g1.size == g2.size) {
	  return "Error: size mismatch in arguments to "+pn;
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
	    otherwise {return notImplementedError(pn,efunc,g1.dtype,g2.dtype,dtype);}
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
	    otherwise {return notImplementedError(pn,efunc,g1.dtype,g2.dtype,dtype);}
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
	    otherwise {return notImplementedError(pn,efunc,g1.dtype,g2.dtype,dtype);}
	    } 
	}
	otherwise {return notImplementedError(pn,efunc,g1.dtype,g2.dtype,dtype);}
	}
	return try! "created " + st.attrib(rname);
    }

    /*
    vector = efunc(vector, scalar, vector)

    :arg reqMsg: request containing (cmd,efunc,name1,dtype,value,name2)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string)
    */
    proc efunc3svMsg(reqMsg: string, st: borrowed SymTab): string {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var efunc = fields[2];
        var name1 = fields[3];
	var dtype = str2dtype(fields[4]);
	var value = fields[5];
	var name2 = fields[6];
        var rname = st.nextName();
	if v {try! writeln("%s %s %s %s %s %s %s : %s".format(cmd,efunc,name1,dtype,value,name2,rname));try! stdout.flush();}

        var g1: borrowed GenSymEntry = st.lookup(name1);
        if (g1 == nil) {return unknownSymbolError(pn,name1);}
	var g2: borrowed GenSymEntry = st.lookup(name2);
	if (g2 == nil) {return unknownSymbolError(pn,name2);}
	if !(g1.size == g2.size) {
	  return "Error: size mismatch in arguments to "+pn;
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
	    otherwise {return notImplementedError(pn,efunc,g1.dtype,dtype,g2.dtype);}
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
	    otherwise {return notImplementedError(pn,efunc,g1.dtype,dtype,g2.dtype);}
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
	    otherwise {return notImplementedError(pn,efunc,g1.dtype,dtype,g2.dtype);}
	    } 
	}
	otherwise {return notImplementedError(pn,efunc,g1.dtype,dtype,g2.dtype);}
	}
	return try! "created " + st.attrib(rname);
    }

    /*
    vector = efunc(vector, scalar, scalar)
    
    :arg reqMsg: request containing (cmd,efunc,name1,dtype1,value1,dtype2,value2)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string)
    */
    proc efunc3ssMsg(reqMsg: string, st: borrowed SymTab): string {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var efunc = fields[2];
        var name1 = fields[3];
	var dtype1 = str2dtype(fields[4]);
	var value1 = fields[5];
	var dtype2 = str2dtype(fields[6]);
	var value2 = fields[7];
        var rname = st.nextName();
	if v {try! writeln("%s %s %s %s %s %s %s %s : %s".format(cmd,efunc,name1,dtype1,value1,dtype2,value2,rname));try! stdout.flush();}

        var g1: borrowed GenSymEntry = st.lookup(name1);
        if (g1 == nil) {return unknownSymbolError(pn,name1);}
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
	    otherwise {return notImplementedError(pn,efunc,g1.dtype,dtype1,dtype2);}
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
	    otherwise {return notImplementedError(pn,efunc,g1.dtype,dtype1,dtype2);}
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
	    otherwise {return notImplementedError(pn,efunc,g1.dtype,dtype1,dtype2);}
	    } 
	}
	otherwise {return notImplementedError(pn,efunc,g1.dtype,dtype1,dtype2);}
	}
	return try! "created " + st.attrib(rname);
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
