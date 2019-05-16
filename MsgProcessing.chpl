
module MsgProcessing
{
    use ServerConfig;

    use Time only;
    use Math only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use AryUtil;
    
    use OperatorMsg;
    use RandMsg;
    use IndexingMsg;
    use UniqueMsg;
    use In1dMsg;
    use HistogramMsg;
    use ArgSortMsg;
    use ReductionMsg;
    
    // parse, execute, and respond to create message
    proc createMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var dtype = str2dtype(fields[2]);
        var size = try! fields[3]:int;

        // get next symbol name
        var rname = st.nextName();
        
        // if verbose print action
        if v {try! writeln("%s %s %i : %s".format(cmd,dtype2str(dtype),size,rname)); try! stdout.flush();}
        // create and add entry to symbol table
        st.addEntry(rname, size, dtype);
        // response message
        return try! "created " + st.attrib(rname);
    }

    // parse, execute, and respond to delete message
    proc deleteMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        if v {try! writeln("%s %s".format(cmd,name));try! stdout.flush();}
        // delete entry from symbol table
        st.deleteEntry(name);
        return try! "deleted %s".format(name);
    }

    // info header only
    proc infoMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        if v {try! writeln("%s %s".format(cmd,name));try! stdout.flush();}
        // if name == "__AllSymbols__" passes back info on all symbols
        return st.info(name);
    }
    
    // dump info and values
    proc dumpMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        if v {try! writeln("%s %s".format(cmd,name));try! stdout.flush();}
        // if name == "__AllSymbols__" passes back dump on all symbols
        return st.dump(name);
    }

    // response to __str__ method in python
    // str convert array data to string
    proc strMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var printThresh = try! fields[3]:int;
        if v {try! writeln("%s %s %i".format(cmd,name,printThresh));try! stdout.flush();}
        return st.datastr(name,printThresh);
    }

    // response to __repr__ method in python
    // repr convert array data to string
    proc reprMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var printThresh = try! fields[3]:int;
        if v {try! writeln("%s %s %i".format(cmd,name,printThresh));try! stdout.flush();}
        return st.datarepr(name,printThresh);
    }

    proc arangeMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var start = try! fields[2]:int;
        var stop = try! fields[3]:int;
        var stride = try! fields[4]:int;
        // compute length
        var len = (stop - start + stride - 1) / stride;
        // get next symbol name
        var rname = st.nextName();
        if v {try! writeln("%s %i %i %i : %i , %s".format(cmd, start, stop, stride, len, rname));try! stdout.flush();}
        
        var t1 = Time.getCurrentTime();
        var aD = makeDistDom(len);
        var a = makeDistArray(len, int);
        writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        t1 = Time.getCurrentTime();
        forall i in aD {
            a[i] = start + (i * stride);
        }
        writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        st.addEntry(rname, new shared SymEntry(a));
        return try! "created " + st.attrib(rname);
    }            

    proc linspaceMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var start = try! fields[2]:real;
        var stop = try! fields[3]:real;
        var len = try! fields[4]:int;
        // compute stride
        var stride = (stop - start) / (len-1);
        // get next symbol name
        var rname = st.nextName();
        if v {try! writeln("%s %r %r %i : %r , %s".format(cmd, start, stop, len, stride, rname));try! stdout.flush();}

        var t1 = Time.getCurrentTime();
        var aD = makeDistDom(len);
        var a = makeDistArray(len, real);
        writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        t1 = Time.getCurrentTime();
        forall i in aD {
            a[i] = start + (i * stride);
        }
        a[0] = start;
        a[len-1] = stop;
        writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        st.addEntry(rname, new shared SymEntry(a));
        return try! "created " + st.attrib(rname);
    }

    // sets all elements in array to a value (broadcast)
    proc setMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var dtype = str2dtype(fields[3]);
        var value = fields[4];

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError("set",name);}

        select (gEnt.dtype, dtype) {
            when (DType.Int64, DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var val: int = try! value:int;
                if v {try! writeln("%s %s to %t".format(cmd,name,val));try! stdout.flush();}
                e.a = val;
                repMsg = try! "set %s to %t".format(name, val);
            }
            when (DType.Int64, DType.Float64) {
                var e = toSymEntry(gEnt,int);
                var val: real = try! value:real;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:int));try! stdout.flush();}
                e.a = val:int;
                repMsg = try! "set %s to %t".format(name, val:int);
            }
            when (DType.Int64, DType.Bool) {
                var e = toSymEntry(gEnt,int);
                value = value.replace("True","true");
                value = value.replace("False","false");
                var val: bool = try! value:bool;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:int));try! stdout.flush();}
                e.a = val:int;
                repMsg = try! "set %s to %t".format(name, val:int);
            }
            when (DType.Float64, DType.Int64) {
                var e = toSymEntry(gEnt,real);
                var val: int = try! value:int;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:real));try! stdout.flush();}
                e.a = val:real;
                repMsg = try! "set %s to %t".format(name, val:real);
            }
            when (DType.Float64, DType.Float64) {
                var e = toSymEntry(gEnt,real);
                var val: real = try! value:real;
                if v {try! writeln("%s %s to %t".format(cmd,name,val));try! stdout.flush();}
                e.a = val;
                repMsg = try! "set %s to %t".format(name, val);
            }
            when (DType.Float64, DType.Bool) {
                var e = toSymEntry(gEnt,real);
                value = value.replace("True","true");
                value = value.replace("False","false");                
                var val: bool = try! value:bool;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:real));try! stdout.flush();}
                e.a = val:real;
                repMsg = try! "set %s to %t".format(name, val:real);
            }
            when (DType.Bool, DType.Int64) {
                var e = toSymEntry(gEnt,bool);
                var val: int = try! value:int;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:bool));try! stdout.flush();}
                e.a = val:bool;
                repMsg = try! "set %s to %t".format(name, val:bool);
            }
            when (DType.Bool, DType.Float64) {
                var e = toSymEntry(gEnt,int);
                var val: real = try! value:real;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:bool));try! stdout.flush();}
                e.a = val:bool;
                repMsg = try! "set %s to %t".format(name, val:bool);
            }
            when (DType.Bool, DType.Bool) {
                var e = toSymEntry(gEnt,bool);
                value = value.replace("True","true");
                value = value.replace("False","false");
                var val: bool = try! value:bool;
                if v {try! writeln("%s %s to %t".format(cmd,name,val));try! stdout.flush();}
                e.a = val;
                repMsg = try! "set %s to %t".format(name, val);
            }
            otherwise {return unrecognizedTypeError("set",fields[3]);}
        }
        return repMsg;
    }
    
    // these ops are functions which take an array and produce and array
    // do scans fit here also? I think so... vector = scanop(vector)
    // parse and respond to efunc "elemental function" message
    // vector = efunc(vector)
    proc efuncMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var efunc = fields[2];
        var name = fields[3];
        var rname = st.nextName();
        if v {try! writeln("%s %s %s : %s".format(cmd,efunc,name,rname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError("efunc",name);}
       
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
                    otherwise {return notImplementedError("efunc",efunc,gEnt.dtype);}
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
                    otherwise {return notImplementedError("efunc",efunc,gEnt.dtype);}
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
                    otherwise {return notImplementedError("efunc",efunc,gEnt.dtype);}
                }
            }
            otherwise {return unrecognizedTypeError("efunc", dtype2str(gEnt.dtype));}
        }
        return try! "created " + st.attrib(rname);
    }

    // these ternary functions which take three arrays and produce and array
    // vector = efunc(vector, vector, vector)
    proc efunc3vvMsg(reqMsg: string, st: borrowed SymTab): string {
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
        if (g1 == nil) {return unknownSymbolError("efunc",name1);}
	var g2: borrowed GenSymEntry = st.lookup(name2);
	if (g2 == nil) {return unknownSymbolError("efunc",name2);}
	var g3: borrowed GenSymEntry = st.lookup(name3);
	if (g3 == nil) {return unknownSymbolError("efunc",name3);}
	if !((g1.size == g2.size) && (g2.size == g3.size)) {
	  return "Error: size mismatch in arguments to efunc3vv";
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
	    otherwise {return notImplementedError("efunc3vv",efunc,g1.dtype,g2.dtype,g3.dtype);}
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
	    otherwise {return notImplementedError("efunc3vv",efunc,g1.dtype,g2.dtype,g3.dtype);}
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
	    otherwise {return notImplementedError("efunc3vv",efunc,g1.dtype,g2.dtype,g3.dtype);}
	    } 
	}
	otherwise {return notImplementedError("efunc3vv",efunc,g1.dtype,g2.dtype,g3.dtype);}
	}
	return try! "created " + st.attrib(rname);
    }

    // vector = efunc(vector, vector, scalar)
    proc efunc3vsMsg(reqMsg: string, st: borrowed SymTab): string {
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
        if (g1 == nil) {return unknownSymbolError("efunc",name1);}
	var g2: borrowed GenSymEntry = st.lookup(name2);
	if (g2 == nil) {return unknownSymbolError("efunc",name2);}
	if !(g1.size == g2.size) {
	  return "Error: size mismatch in arguments to efunc3vs";
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
	    otherwise {return notImplementedError("efunc3vs",efunc,g1.dtype,g2.dtype,dtype);}
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
	    otherwise {return notImplementedError("efunc3vs",efunc,g1.dtype,g2.dtype,dtype);}
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
	    otherwise {return notImplementedError("efunc3vs",efunc,g1.dtype,g2.dtype,dtype);}
	    } 
	}
	otherwise {return notImplementedError("efunc3vs",efunc,g1.dtype,g2.dtype,dtype);}
	}
	return try! "created " + st.attrib(rname);
    }

    // vector = efunc(vector, scalar, vector)
    proc efunc3svMsg(reqMsg: string, st: borrowed SymTab): string {
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
        if (g1 == nil) {return unknownSymbolError("efunc",name1);}
	var g2: borrowed GenSymEntry = st.lookup(name2);
	if (g2 == nil) {return unknownSymbolError("efunc",name2);}
	if !(g1.size == g2.size) {
	  return "Error: size mismatch in arguments to efunc3sv";
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
	    otherwise {return notImplementedError("efunc3sv",efunc,g1.dtype,dtype,g2.dtype);}
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
	    otherwise {return notImplementedError("efunc3sv",efunc,g1.dtype,dtype,g2.dtype);}
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
	    otherwise {return notImplementedError("efunc3sv",efunc,g1.dtype,dtype,g2.dtype);}
	    } 
	}
	otherwise {return notImplementedError("efunc3sv",efunc,g1.dtype,dtype,g2.dtype);}
	}
	return try! "created " + st.attrib(rname);
    }

    // vector = efunc(vector, scalar, scalar)
    proc efunc3ssMsg(reqMsg: string, st: borrowed SymTab): string {
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
        if (g1 == nil) {return unknownSymbolError("efunc",name1);}
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
	    otherwise {return notImplementedError("efunc3ss",efunc,g1.dtype,dtype1,dtype2);}
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
	    otherwise {return notImplementedError("efunc3ss",efunc,g1.dtype,dtype1,dtype2);}
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
	    otherwise {return notImplementedError("efunc3ss",efunc,g1.dtype,dtype1,dtype2);}
	    } 
	}
	otherwise {return notImplementedError("efunc3sv",efunc,g1.dtype,dtype1,dtype2);}
	}
	return try! "created " + st.attrib(rname);
    }

    /* The 'where' function takes a boolean array and two other arguments A and B, and 
       returns an array with A where the boolean is true and B where it is false. A and B
       can be vectors or scalars. I would like to be able to write these functions without
       the param kind and just let the compiler choose, but it complains about an
       ambiguous call. */
    proc where_helper(cond:[?D] bool, A:[D] ?t, B:[D] t, param kind):[D] t where (kind == 0) {
      var C:[D] t;
      forall (ch, a, b, c) in zip(cond, A, B, C) {
	c = if ch then a else b;
      }
      return C;
    }

    proc where_helper(cond:[?D] bool, A:[D] ?t, b:t, param kind):[D] t where (kind == 1) {
      var C:[D] t;
      forall (ch, a, c) in zip(cond, A, C) {
	c = if ch then a else b;
      }
      return C;
    }

    proc where_helper(cond:[?D] bool, a:?t, B:[D] t, param kind):[D] t where (kind == 2) {
      var C:[D] t;
      forall (ch, b, c) in zip(cond, B, C) {
	c = if ch then a else b;
      }
      return C;
    }

    proc where_helper(cond:[?D] bool, a:?t, b:t, param kind):[D] t where (kind == 3) {
      var C:[D] t;
      forall (ch, c) in zip(cond, C) {
	c = if ch then a else b;
      }
      return C;
    }
}
