
module OperatorMsg
{
    use ServerConfig;

    use Time;
    use Math;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use MsgProcessing;
    /*
    Parse and respond to binopvv message.
    vv == vector op vector

    :arg reqMsg: request containing (cmd,op,aname,bname,rname)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) 
    */
    proc binopvvMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var op = fields[2];
        var aname = fields[3];
        var bname = fields[4];
        var rname = st.nextName();
        if v {try! writeln("%s %s %s %s : %s".format(cmd,op,aname,bname,rname));try! stdout.flush();}

        var left: borrowed GenSymEntry = st.lookup(aname);
        if (left == nil) {return try! "Error: binopvv: unkown symbol %s".format(aname);}
        var right: borrowed GenSymEntry = st.lookup(bname);
        if (right == nil) {return try! "Error: binopvv: unkown symbol %s".format(bname);}

        select (left.dtype, right.dtype) {
            when (DType.Int64, DType.Int64) {
                var l = toSymEntry(left,int);
                var r = toSymEntry(right,int);
                select op
                {
                    when "+" {
                        // new way: no extra copy!
                        // 1) create a new symbol table entry
                        st.addEntry(rname, l.size, int);
                        // 2) lookup and cast to the correct result type
                        var e = toSymEntry(st.lookup(rname), int);
                        // 3) do operation writing the result inplace into the symbol table entry's array
                        e.a = l.a + r.a;
                        
                        /* // old way: extra copy! */
                        /* // 1) does operation creating a new array result */
                        /* var a = l.a + r.a; */
                        /* // 2) copies result array into a new symbol table entry array */
                        /* st.addEntry(rname, new shared SymEntry(a)); */
                    }
                    when "-" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a - r.a;
                    }
                    when "*" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a * r.a;
                    }
                    when "/" { // truediv
		        // Chapel will halt if any element of r.a is zero
		        if || reduce (r.a == 0) {
			  return "Error: Attempt to divide by zero";
			}
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);                        
                        e.a = l.a:real / r.a:real;
                    } 
                    when "//" { // floordiv
		        // Chapel will halt if any element of r.a is zero
		        if || reduce (r.a == 0) {
			  return "Error: Attempt to divide by zero";
			}
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a / r.a;
                    }
		    when "%" { // modulo
		        // Chapel will halt if any element of r.a is zero
		        if || reduce (r.a == 0) {
			  return "Error: Attempt to compute a modulus by zero";
			}
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
		        e.a = l.a % r.a;
		    }
                    when "<" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a < r.a;
                    }
                    when ">" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a > r.a;
                    }
                    when "<=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a <= r.a;
                    }
                    when ">=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a >= r.a;
                    }
                    when "==" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a == r.a;
                    }
                    when "!=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a != r.a;
                    }
                    when "<<" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a << r.a;
                    }                    
                    when ">>" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a >> r.a;
                    }                    
                    when "&" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a & r.a;
                    }                    
                    when "|" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a | r.a;
                    }                    
                    when "^" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a ^ r.a;
                    }    
                    when "**" { //POTENTIALL BROKE 
                        var a = l.a**r.a;
                        st.addEntry(rname, new shared SymEntry(a));
                    } //end rsdange experiment     
                    otherwise {return notImplementedError("binopvv",left.dtype,op,right.dtype);}
                }
            }
            when (DType.Int64, DType.Float64) {
                var l = toSymEntry(left,int);
                var r = toSymEntry(right,real);
                select op
                {
                    when "+" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a + r.a;
                    }
                    when "-" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a - r.a;
                    }
                    when "*" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a * r.a;
                    }
                    when "/" { // truediv
                        // add check for division by zero!
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a:real / r.a;
                    } 
                    when "//" { // floordiv
                        // add check for division by zero!
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = floor((l.a / r.a));
                    }
                    when "<" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a < r.a;
                    }
                    when ">" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a > r.a;
                    }
                    when "<=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a <= r.a;
                    }
                    when ">=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a >= r.a;
                    }
                    when "==" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a == r.a;
                    }
                    when "!=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a != r.a;
                    }
                    when "**" { //POTENTIALL BROKE 
                        var a = l.a**r.a;
                        st.addEntry(rname, new shared SymEntry(a));
                    } //end rsdange experiment     
                    otherwise {return notImplementedError("binopvv",left.dtype,op,right.dtype);}
                }
            }
            when (DType.Float64, DType.Int64) {
                var l = toSymEntry(left,real);
                var r = toSymEntry(right,int);
                select op
                {
                    when "+" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a + r.a;
                    }
                    when "-" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a - r.a;
                    }
                    when "*" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a * r.a;
                    }
                    when "/" { // truediv
                        // add check for division by zero!
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a / r.a:real;
                    } 
                    when "//" { // floordiv
                        // add check for division by zero!
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = floor(l.a / r.a);
                    }
                    when "<" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a < r.a;
                    }
                    when ">" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a > r.a;
                    }
                    when "<=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a <= r.a;
                    }
                    when ">=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a >= r.a;
                    }
                    when "==" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a == r.a;
                    }
                    when "!=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a != r.a;
                    }
                    when "**" { //POTENTIALL BROKE 
                       var a = l.a**r.a;
                        st.addEntry(rname, new shared SymEntry(a));
                    } //end rsdange experiment     
                    otherwise {return notImplementedError("binopvv",left.dtype,op,right.dtype);}
                }
            }
            when (DType.Float64, DType.Float64) {
                var l = toSymEntry(left,real);
                var r = toSymEntry(right,real);
                select op
                {
                    when "+" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a + r.a;
                    }
                    when "-" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a - r.a;
                    }
                    when "*" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a * r.a;
                    }
                    when "/" { // truediv
                        // add check for division by zero!
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a / r.a;
                    } 
                    when "//" { // floordiv
                        // add check for division by zero!
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = floor(l.a / r.a);
                    }
                    when "<" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a < r.a;
                    }
                    when ">" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a > r.a;
                    }
                    when "<=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a <= r.a;
                    }
                    when ">=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a >= r.a;
                    }
                    when "==" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a == r.a;
                    }
                    when "!=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a != r.a;
                    }
                    when "**" { //POTENTIALL BROKE 
                        var a = l.a**r.a;
                        st.addEntry(rname, new shared SymEntry(a));
                    } //end rsdange experiment     
                    otherwise {return notImplementedError("binopvv",left.dtype,op,right.dtype);}
                }
            }
	    when (DType.Bool, DType.Bool) {
	        var l = toSymEntry(left, bool);
		var r = toSymEntry(right, bool);
		select op {
		    when "|" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
		        e.a = l.a | r.a;
		    }
		    when "&" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
		        e.a = l.a & r.a;
		    }
		    when "^" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
		        e.a = l.a ^ r.a;
		    }
		    otherwise {return notImplementedError("binopvv",left.dtype,op,right.dtype);}
		}
	    }
	    when (DType.Bool, DType.Int64) {
	        var l = toSymEntry(left, bool);
		var r = toSymEntry(right, int);
		select op {
		    when "+" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
		        e.a = l.a:int + r.a;
		    }
		    when "-" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
		        e.a = l.a:int - r.a;
		    }
		    when "*" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
		        e.a = l.a:int * r.a;
		    }
		    otherwise {return notImplementedError("binopvv",left.dtype,op,right.dtype);}
		}
	    }
	    when (DType.Int64, DType.Bool) {
	        var l = toSymEntry(left, int);
		var r = toSymEntry(right, bool);
		select op {
		    when "+" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
		        e.a = l.a + r.a:int;
		    }
		    when "-" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
		        e.a = l.a - r.a:int;
		    }
		    when "*" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
		        e.a = l.a * r.a:int;
		    }
		    otherwise {return notImplementedError("binopvv",left.dtype,op,right.dtype);}
		}
	    }
	    when (DType.Bool, DType.Float64) {
	        var l = toSymEntry(left, bool);
		var r = toSymEntry(right, real);
		select op {
		    when "+" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);                        
		        e.a = l.a:real + r.a;
		    }
		    when "-" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);                        
		        e.a = l.a:real - r.a;
		    }
		    when "*" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);                        
		        e.a = l.a:real * r.a;
		    }
		    otherwise {return notImplementedError("binopvv",left.dtype,op,right.dtype);}
		}
	    }
	    when (DType.Float64, DType.Bool) {
	        var l = toSymEntry(left, real);
		var r = toSymEntry(right, bool);
		select op {
		    when "+" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);                        
		        e.a = l.a + r.a:real;
		    }
		    when "-" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);                        
		        e.a = l.a - r.a:real;
		    }
		    when "*" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);                        
		        e.a = l.a * r.a:real;
		    }
		    otherwise {return notImplementedError("binopvv",left.dtype,op,right.dtype);}
		}
	    }
            otherwise {return unrecognizedTypeError("binopvv",
                                                    "("+dtype2str(left.dtype)+","+dtype2str(right.dtype)+")");}
        }
        return try! "created " + st.attrib(rname);
    }
    /*
    Parse and respond to binopvs message.
    vs == vector op scalar

    :arg reqMsg: request containing (cmd,op,aname,dtype,value)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) 
    */
    proc binopvsMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string = ""; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var op = fields[2];
        var aname = fields[3];
        var dtype = str2dtype(fields[4]);
        var value = fields[5];
        var rname = st.nextName();
        if v {try! writeln("%s %s %s %s %s : %s".format(cmd,op,aname,dtype2str(dtype),value,rname));try! stdout.flush();}

        var left: borrowed GenSymEntry = st.lookup(aname);
        if (left == nil) {return try! "Error: binopvs: unkown symbol %s".format(aname);}
        select (left.dtype, dtype) {
            when (DType.Int64, DType.Int64) {
                var l = toSymEntry(left,int);
                var val = try! value:int;
                select op
                {
                    when "+" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a + val;
                    }
                    when "-" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a - val;
                    }
                    when "*" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a * val;
                    }
                    when "/" { // truediv
		        // Chapel will halt if val is zero
		        if (val == 0) {
			  return "Error: Attempt to divide by zero";
			}
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a:real / val:real;
                    } 
                    when "//" { // floordiv
		        // Chapel will halt if any element of r.a is zero
		        if (val == 0) {
			  return "Error: Attempt to divide by zero";
			}
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a / val;
                    }
		    when "%" { // modulo
		      // Chapel will halt if val is zero
		      if (val == 0) {
			return "Error: Attempt to compute a modulus by zero";
		      }
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
		        e.a = l.a % val;
		    } 
                    when "<" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a < val;
                    }
                    when ">" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a > val;
                    }
                    when "<=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a <= val;
                    }
                    when ">=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a >= val;
                    }
                    when "==" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a == val;
                    }
                    when "!=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a != val;
                    }
                    when "<<" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a << val;
                    }
                    when ">>" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a >> val;
                    }
                    when "&" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a & val;
                    }
                    when "|" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a | val;
                    }
                    when "^" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a ^ val;
                    }
                    when "**" { 
                        var a =l.a**val;
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    otherwise {return notImplementedError("binopvs",left.dtype,op,dtype);}
                }
            }
            when (DType.Int64, DType.Float64) {
                var l = toSymEntry(left,int);
                var val = try! value:real;
                select op
                {
                    when "+" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a + val;
                    }
                    when "-" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a - val;
                    }
                    when "*" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a * val;
                    }
                    when "/" { // truediv
                        // add check for division by zero!
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a:real / val;
                    } 
                    when "//" { // floordiv
                        // add check for division by zero!
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = floor(l.a:real / val);
                    }
                    when "<" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a < val;
                    }
                    when ">" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a > val;
                    }
                    when "<=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a <= val;
                    }
                    when ">=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a >= val;
                    }
                    when "==" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a == val;
                    }
                    when "!=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a != val;
                    }
                    when "**" { //this doesnt seem elegant...?
                        var a =l.a**val;
                        st.addEntry(rname, new shared SymEntry(a));
                    }//end rsdange experiment
                    otherwise {return notImplementedError("binopvs",left.dtype,op,dtype);}
                }
            }
            when (DType.Float64, DType.Int64) {
                var l = toSymEntry(left,real);
                var val = try! value:int;
                select op
                {
                    when "+" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a + val;
                    }
                    when "-" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a - val;
                    }
                    when "*" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a * val;
                    }
                    when "/" { // truediv
                        // add check for division by zero!
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a / val:real;
                    } 
                    when "//" { // floordiv
                        // add check for division by zero!
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = floor(l.a / val:real);
                    }
                    when "<" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a < val;
                    }
                    when ">" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a > val;
                    }
                    when "<=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a <= val;
                    }
                    when ">=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a >= val;
                    }
                    when "==" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a == val;
                    }
                    when "!=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a != val;
                    }
                    when "**" { //this doesnt seem elegant...?
                        var a =l.a**val;
                        st.addEntry(rname, new shared SymEntry(a));
                    }//end rsdange experiment
                    otherwise {return notImplementedError("binopvs",left.dtype,op,dtype);}
                }
            }
            when (DType.Float64, DType.Float64) {
                var l = toSymEntry(left,real);
                var val = try! value:real;
                select op
                {
                    when "+" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a + val;
                    }
                    when "-" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a - val;
                    }
                    when "*" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a * val;
                    }
                    when "/" { // truediv
                        // add check for division by zero!
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a / val;
                    } 
                    when "//" { // floordiv
                        // add check for division by zero!
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = floor(l.a / val);
                    }
                    when "<" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a < val;
                    }
                    when ">" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a > val;
                    }
                    when "<=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a <= val;
                    }
                    when ">=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a >= val;
                    }
                    when "==" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a == val;
                    }
                    when "!=" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = l.a != val;
                    }
                    when "**" { //this doesnt seem elegant...?
                        var a =l.a**val;
                        st.addEntry(rname, new shared SymEntry(a));
                    }//end rsdange experiment
                    otherwise {return notImplementedError("binopvs",left.dtype,op,dtype);}
                }
            }
	    when (DType.Bool, DType.Bool) {
	        var l = toSymEntry(left, bool);
		var val = try! value.toLower():bool;
		select op {
		    when "|" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
		        e.a = l.a | val;
		    }
		    when "&" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
		        e.a = l.a & val;
		    }
		    when "^" {
                        st.addEntry(rname, l.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
		        e.a = l.a ^ val;
		    }
                    otherwise {return notImplementedError("binopvs",left.dtype,op,dtype);}
		}
	    }
	    when (DType.Bool, DType.Int64) {
	        var l = toSymEntry(left, bool);
		var val = try! value:int;
		select op {
		    when "+" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a:int + val;
		    }
		    when "-" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a:int - val;
		    }
		    when "*" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a:int * val;
		    }
                    otherwise {return notImplementedError("binopvs",left.dtype,op,dtype);}
		}
	    }
	    when (DType.Int64, DType.Bool) {
	        var l = toSymEntry(left, int);
		var val = try! value.toLower():bool;
		select op {
		    when "+" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a + val:int;
		    }
		    when "-" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a - val:int;
		    }
		    when "*" {
                        st.addEntry(rname, l.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = l.a * val:int;
		    }
                    otherwise {return notImplementedError("binopvs",left.dtype,op,dtype);}
		}
	    }
	    when (DType.Bool, DType.Float64) {
	        var l = toSymEntry(left, bool);
		var val = try! value:real;
		select op {
		    when "+" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname),real);
                        e.a = l.a:real + val;
		    }
		    when "-" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a:real - val;
		    }
		    when "*" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a:real * val;
		    }
                    otherwise {return notImplementedError("binopvs",left.dtype,op,dtype);}
		}
	    }
	    when (DType.Float64, DType.Bool) {
	        var l = toSymEntry(left, real);
		var val = try! value.toLower():bool;
		select op {
		    when "+" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a + val:real;
		    }
		    when "-" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a - val:real;
		    }
		    when "*" {
                        st.addEntry(rname, l.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = l.a * val:real;
		    }
                    otherwise {return notImplementedError("binopvs",left.dtype,op,dtype);}
		}
	    }
            otherwise {return unrecognizedTypeError("binopvs",
                                                    "("+dtype2str(left.dtype)+","+dtype2str(dtype)+")");}
        }
        return try! "created " + st.attrib(rname);
    }

    /*
    Parse and respond to binopsv message.
    sv == scalar op vector

    :arg reqMsg: request containing (cmd,op,dtype,value,aname)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) 
    */
    proc binopsvMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string = ""; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var op = fields[2];
        var dtype = str2dtype(fields[3]);
        var value = fields[4];
        var aname = fields[5];
        var rname = st.nextName();
        if v {try! writeln("%s %s %s %s %s : %s".format(cmd,op,dtype2str(dtype),value,aname,rname));try! stdout.flush();}

        var right: borrowed GenSymEntry = st.lookup(aname);
        if (right == nil) {return try! "Error: binopsv: unkown symbol %s".format(aname);}
        select (dtype, right.dtype) {
            when (DType.Int64, DType.Int64) {
                var val = try! value:int;
                var r = toSymEntry(right,int);
                select op
                {
                    when "+" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val + r.a;
                    }
                    when "-" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val - r.a;
                    }
                    when "*" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val * r.a;
                    }
                    when "/" { // truediv
		        // Chapel will halt if any element of r.a is zero
		        if || reduce (r.a == 0) {
			  return "Error: Attempt to divide by zero";
			}
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a =  val:real / r.a:real;
                    } 
                    when "//" { // floordiv
		        // Chapel will halt if any element of r.a is zero
		        if || reduce (r.a == 0) {
			  return "Error: Attempt to divide by zero";
			}
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val / r.a;
                    }
		    when "%" { // modulo
		        // Chapel will halt if any element of r.a is zero
		        if || reduce (r.a == 0) {
			  return "Error: Attempt to compute a modulus by zero";
			}
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
		        e.a = val % r.a;
		    }
                    when "<" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val < r.a;
                    }
                    when ">" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val > r.a;
                    }
                    when "<=" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val <= r.a;
                    }
                    when ">=" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val >= r.a;
                    }
                    when "==" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val == r.a;
                    }
                    when "!=" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val != r.a;
                    }
                    when "<<" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val << r.a;
                    }
                    when ">>" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val >> r.a;
                    }
                    when "&" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val & r.a;
                    }
                    when "|" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val | r.a;
                    }
                    when "^" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val ^ r.a;
                    }
                    when "**" { //this doesnt seem elegant...?
                        var a =val**r.a;
                        st.addEntry(rname, new shared SymEntry(a));
                    }//end rsdange experiment
                    otherwise {return notImplementedError("binopsv",dtype,op,right.dtype);}
                }
            }
            when (DType.Int64, DType.Float64) {
                var val = try! value:int;
                var r = toSymEntry(right,real);
                select op
                {
                    when "+" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val + r.a;
                    }
                    when "-" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val - r.a;
                    }
                    when "*" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val * r.a;
                    }
                    when "/" { // truediv
                        // add check for division by zero!
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val:real / r.a;
                    }
                    when "//" { // floordiv
                        // add check for division by zero!
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = floor(val:real / r.a);
                    }
                    when "<" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val < r.a;
                    }
                    when ">" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val > r.a;
                    }
                    when "<=" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val <= r.a;
                    }
                    when ">=" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val >= r.a;
                    }
                    when "==" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val == r.a;
                    }
                    when "!=" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val != r.a;
                    }
                    when "**" { //this doesnt seem elegant...?
                        var a =val**r.a;
                        st.addEntry(rname, new shared SymEntry(a));
                    }//end rsdange experiment
                    otherwise {return notImplementedError("binopsv",dtype,op,right.dtype);}
                }
            }
            when (DType.Float64, DType.Int64) {
                var val = try! value:real;
                var r = toSymEntry(right,int);
                select op
                {
                    when "+" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val + r.a;
                    }
                    when "-" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val - r.a;
                    }
                    when "*" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val * r.a;
                    }
                    when "/" { // truediv
                        // add check for division by zero!
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val / r.a:real;
                    } 
                    when "//" { // floordiv
                        // add check for division by zero!
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = floor(val / r.a:real);
                    }
                    when "<" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val < r.a;
                    }
                    when ">" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val > r.a;
                    }
                    when "<=" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val <= r.a;
                    }
                    when ">=" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val >= r.a;
                    }
                    when "==" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val == r.a;
                    }
                    when "!=" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val != r.a;
                    }
                    when "**" { //this doesnt seem elegant...?
                        var a =val**r.a;
                        st.addEntry(rname, new shared SymEntry(a));
                    }//end rsdange experiment
                    otherwise {return notImplementedError("binopsv",dtype,op,right.dtype);}
                }
            }
            when (DType.Float64, DType.Float64) {
                var val = try! value:real;
                var r = toSymEntry(right,real);
                select op
                {
                    when "+" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val + r.a;
                    }
                    when "-" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val - r.a;
                    }
                    when "*" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val * r.a;
                    }
                    when "/" { // truediv
                        // add check for division by zero!
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val / r.a;
                    } 
                    when "//" { // floordiv
                        // add check for division by zero!
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = floor(val / r.a);
                    }
                    when "<" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val < r.a;
                    }
                    when ">" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val > r.a;
                    }
                    when "<=" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val <= r.a;
                    }
                    when ">=" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val >= r.a;
                    }
                    when "==" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val == r.a;
                    }
                    when "!=" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
                        e.a = val != r.a;
                    }
                    when "**" { //this doesnt seem elegant...?
                        var a =val**r.a;
                        st.addEntry(rname, new shared SymEntry(a));
                    }//end rsdange experiment
                    otherwise {return notImplementedError("binopsv",dtype,op,right.dtype);}
                }
            }
	    when (DType.Bool, DType.Bool) {
	        var val = try! value.toLower():bool;
		var r = toSymEntry(right, bool);
		select op {
		    when "|" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
 		        e.a = val | r.a;
		    }
		    when "&" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
 		        e.a = val & r.a;
		    }
		    when "^" {
                        st.addEntry(rname, r.size, bool);
                        var e = toSymEntry(st.lookup(rname), bool);
 		        e.a = val ^ r.a;
		    }
                    otherwise {return notImplementedError("binopsv",dtype,op,right.dtype);}
		}
	    }
	    when (DType.Bool, DType.Int64) {
	        var val = try! value.toLower():bool;
		var r = toSymEntry(right, int);
		select op {
		    when "+" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val:int + r.a;
		    }
		    when "-" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val:int - r.a;
		    }
		    when "*" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val:int * r.a;
		    }
                    otherwise {return notImplementedError("binopsv",dtype,op,right.dtype);}
		}
	    }
	    when (DType.Int64, DType.Bool) {
		var val = try! value:int;
		var r = toSymEntry(right, bool);
		select op {
		    when "+" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val + r.a:int;
		    }
		    when "-" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val - r.a:int;
		    }
		    when "*" {
                        st.addEntry(rname, r.size, int);
                        var e = toSymEntry(st.lookup(rname), int);
                        e.a = val * r.a:int;
		    }
                    otherwise {return notImplementedError("binopsv",dtype,op,right.dtype);}
		}
	    }
	    when (DType.Bool, DType.Float64) {
	        var val = try! value.toLower():bool;
		var r = toSymEntry(right, real);
		select op {
		    when "+" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val:real + r.a;
		    }
		    when "-" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val:real - r.a;
		    }
		    when "*" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val:real * r.a;
		    }
                    otherwise {return notImplementedError("binopsv",dtype,op,right.dtype);}
		}
	    }
	    when (DType.Float64, DType.Bool) {
		var val = try! value:real;
		var r = toSymEntry(right, bool);
		select op {
		    when "+" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val + r.a:real;
		    }
		    when "-" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val - r.a:real;
		    }
		    when "*" {
                        st.addEntry(rname, r.size, real);
                        var e = toSymEntry(st.lookup(rname), real);
                        e.a = val * r.a:real;
		    }
                    otherwise {return notImplementedError("binopsv",dtype,op,right.dtype);}
		}
	    }
            otherwise {return unrecognizedTypeError("binopsv",
                                                    "("+dtype2str(dtype)+","+dtype2str(right.dtype)+")");}
        }
        return try! "created " + st.attrib(rname);
    }

    /*
    Parse and respond to opeqvv message.
    vector op= vector

    :arg reqMsg: request containing (cmd,op,aname,bname)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) 
    */
    proc opeqvvMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var op = fields[2];
        var aname = fields[3];
        var bname = fields[4];
        if v {try! writeln("%s %s %s %s".format(cmd,op,aname,bname));try! stdout.flush();}
        
        var left: borrowed GenSymEntry = st.lookup(aname);
        if (left == nil) {return unknownSymbolError("opeqvv",aname);}
        var right: borrowed GenSymEntry = st.lookup(bname);
        if (right == nil) {return unknownSymbolError("opeqvv",bname);}
        select (left.dtype, right.dtype) {
            when (DType.Int64, DType.Int64) {
                var l = toSymEntry(left,int);
                var r = toSymEntry(right,int);
                select op
                {
                    when "+=" { l.a += r.a; }
                    when "-=" { l.a -= r.a; }
                    when "*=" { l.a *= r.a; }
                    when "//=" { l.a /= r.a; }//floordiv
                    when "**=" { 
                        //do a +scan and iter thru res
                        //if any i+1 val is < i val, then set neg flag to true.
                        var neg: atomic bool;
                        neg.write(false);
                        //(there exists an el in r.a thats<0)
                        forall i in r.a{
                            if(i<0){
                                neg.write(true);
                            }
                        }
                        if(neg.read()){
                            
                            // cmd = "create"; //create an array of reals.
                            // var len = l.a.size;
                            // var dtype = DType.Float64;
                            // var reqMsg = try! "%s %s %i".format(cmd, dtype2str(dtype), len);
                            // repMsg = createMsg(reqMsg, st);
                            // if v { writeln(repMsg); }
                            // fields = repMsg.split(); 
                            // aname= fields[2];
                        
                            // var left2: borrowed GenSymEntry = st.lookup(aname);
                            // var l2=toSymEntry(left2,real);

                            // l2.a=[i in l.a] i:real; //transfer and cast vals
                            // l2.a **= r.a; //perform the desired operation
                            writeln("Negative exponent not supported in vv int64 int64");
                            return notImplementedError("opeqvv",left.dtype,op,right.dtype);
                        }//negative exponent
                        else{ l.a **= r.a; }
                    }
                    otherwise {return notImplementedError("opeqvv",left.dtype,op,right.dtype);}
                }
            }
            when (DType.Int64, DType.Float64) {
                var l = toSymEntry(left,int);
                var r = toSymEntry(right,real);
                return notImplementedError("opeqvv",left.dtype,op,right.dtype);
            }
            when (DType.Float64, DType.Int64) {
                var l = toSymEntry(left,real);
                var r = toSymEntry(right,int);
                select op
                {
                    when "+=" {l.a += r.a;}
                    when "-=" {l.a -= r.a;}
                    when "*=" {l.a *= r.a;}
                    when "/=" {l.a /= r.a:real;} //truediv
                    when "//=" {l.a = floor(l.a / r.a);} //floordiv
                    when "**=" { l.a **= r.a; }
                    otherwise {return notImplementedError("opeqvv",left.dtype,op,right.dtype);}
                }
            }
            when (DType.Float64, DType.Float64) {
                var l = toSymEntry(left,real);
                var r = toSymEntry(right,real);
                select op
                {
                    when "+=" {l.a += r.a;}
                    when "-=" {l.a -= r.a;}
                    when "*=" {l.a *= r.a;}
                    when "/=" {l.a /= r.a;}//truediv
                    when "//=" {l.a = floor(l.a / r.a);}//floordiv
                    when "**=" { l.a **= r.a; }
                    otherwise {return notImplementedError("opeqvv",left.dtype,op,right.dtype);}
                }
            }
	    when (DType.Bool, DType.Bool) {
	        var l = toSymEntry(left, bool);
		var r = toSymEntry(right, bool);
		select op
		{
		    when "|=" {l.a |= r.a;}
		    when "&=" {l.a &= r.a;}
		    when "^=" {l.a ^= r.a;}
		    otherwise {return notImplementedError("opeqvv",left.dtype,op,right.dtype);}
		}
	    }
	    when (DType.Int64, DType.Bool) {
	        var l = toSymEntry(left, int);
		var r = toSymEntry(right, bool);
		select op
		{ 
		    when "+=" {l.a += r.a:int;}
		    when "-=" {l.a -= r.a:int;}
		    when "*=" {l.a *= r.a:int;}
  		    otherwise {return notImplementedError("opeqvv",left.dtype,op,right.dtype);}
		}
	    }
	    when (DType.Float64, DType.Bool) {
	        var l = toSymEntry(left, real);
		var r = toSymEntry(right, bool);
		select op
		{
		    when "+=" {l.a += r.a:real;}
		    when "-=" {l.a -= r.a:real;}
		    when "*=" {l.a *= r.a:real;}
  		    otherwise {return notImplementedError("opeqvv",left.dtype,op,right.dtype);}
		}
	    }
            otherwise {return unrecognizedTypeError("opeqvv",
                                                    "("+dtype2str(left.dtype)+","+dtype2str(right.dtype)+")");}
        }
        return "opeqvv success";
    }

    /*
    Parse and respond to opeqvs message.
    vector op= scalar

    :arg reqMsg: request containing (cmd,op,aname,bname,rname)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) 

    */
    proc opeqvsMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var op = fields[2];
        var aname = fields[3];
        var dtype = str2dtype(fields[4]);
        var value = fields[5];
        if v {try! writeln("%s %s %s %s %s".format(cmd,op,aname,dtype2str(dtype),value));try! stdout.flush();}

        var left: borrowed GenSymEntry = st.lookup(aname);
        if (left == nil) {return unknownSymbolError("opeqvs",aname);}

        select (left.dtype, dtype) {
            when (DType.Int64, DType.Int64) {
                var l = toSymEntry(left,int);
                var val = try! value:int;
                select op
                {
                    when "+=" { l.a += val; }
                    when "-=" { l.a -= val; }
                    when "*=" { l.a *= val; }
                    when "//=" { l.a /= val; }//floordiv
                    when "**=" { 
                        if(val<0){
                            
                            // cmd = "create"; //create an array of reals.
                            // var len = l.a.size;
                            // dtype = DType.Float64;
                            // var reqMsg = try! "%s %s %i".format(cmd, dtype2str(dtype), len);
                            // repMsg = createMsg(reqMsg, st);
                            // if v { writeln(repMsg); }
                            // var fields = repMsg.split(); 
                            // aname= fields[2];
                        
                            // var left2: borrowed GenSymEntry = st.lookup(aname);
                            // var l2=toSymEntry(left2,real);

                            // l2.a=[i in l.a] i:real; //transfer and cast vals
                            // l2.a **= val; //perform the desired operation
                            writeln("Negative exponent not supported in vs int64 int64");
                            return notImplementedError("opeqvs",left.dtype,op,dtype);
                        }//negative exponent
                        else{ l.a **= val; }

                    }
                    otherwise {return notImplementedError("opeqvs",left.dtype,op,dtype);}
                }
            }
            when (DType.Int64, DType.Float64) {
                var l = toSymEntry(left,int);
                var val = try! value:real;
                return notImplementedError("opeqvs",left.dtype,op,dtype);
            }
            when (DType.Float64, DType.Int64) {
                var l = toSymEntry(left,real);
                var val = try! value:int;
                select op
                {
                    when "+=" {l.a += val;}
                    when "-=" {l.a -= val;}
                    when "*=" {l.a *= val;}
                    when "/=" {l.a /= val:real;} //truediv
                    when "//=" {l.a = floor(l.a / val);} //floordiv
                    when "**=" { l.a **= val; }
                    otherwise {return notImplementedError("opeqvs",left.dtype,op,dtype);}
                }
            }
            when (DType.Float64, DType.Float64) {
                var l = toSymEntry(left,real);
                var val = try! value:real;
                select op
                {
                    when "+=" {l.a += val;}
                    when "-=" {l.a -= val;}
                    when "*=" {l.a *= val;}
                    when "/=" {l.a /= val;}//truediv
                    when "//=" {l.a = floor(l.a / val);}//floordiv
                    when "**=" { l.a **= val; }
                    otherwise {return notImplementedError("opeqvs",left.dtype,op,dtype);}
                }
            }
	    when (DType.Int64, DType.Bool) {
	        var l = toSymEntry(left, int);
		var val = try! value: bool;
		select op {
		    when "+=" {l.a += val:int;}
		    when "-=" {l.a -= val:int;}
		    when "*=" {l.a *= val:int;}
		    otherwise {return notImplementedError("opeqvs",left.dtype,op,dtype);}
		}
	    }
	    when (DType.Float64, DType.Bool) {
	        var l = toSymEntry(left, real);
		var val = try! value: bool;
		select op {
		    when "+=" {l.a += val:real;}
		    when "-=" {l.a -= val:real;}
		    when "*=" {l.a *= val:real;}
		    otherwise {return notImplementedError("opeqvs",left.dtype,op,dtype);}
		}
	    }
            otherwise {return unrecognizedTypeError("opeqvs",
                                                    "("+dtype2str(left.dtype)+","+dtype2str(dtype)+")");}
        }
        return "opeqvs success";
    }

}
