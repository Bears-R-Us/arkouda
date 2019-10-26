
module OperatorMsg
{
    use ServerConfig;

    use Time;
    use Math;
    use Reflection only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

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
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var op = fields[2];
        var aname = fields[3];
        var bname = fields[4];
        var rname = st.nextName();
        if v {try! writeln("%s %s %s %s : %s".format(cmd,op,aname,bname,rname));try! stdout.flush();}

        var left: borrowed GenSymEntry = st.lookup(aname);
        if (left == nil) {return unknownSymbolError(pn, aname);}
        var right: borrowed GenSymEntry = st.lookup(bname);
        if (right == nil) {return unknownSymbolError(pn, bname);}

        select (left.dtype, right.dtype) {
            when (DType.Int64, DType.Int64) {
                var l = toSymEntry(left,int);
                var r = toSymEntry(right,int);
                select op
                {
                    when "+" {
                        // new way: no extra copy!
                        // 1) insert new entry into generic symbol table and
                        //    return a borrow to that entry in the table
                        var e = st.addEntry(rname, l.size, int);
                        // 2) do operation writing the result inplace into the symbol table entry's array
                        e.a = l.a + r.a;
                        
                        /* // old way: extra copy! */
                        /* // 1) does operation creating a new array result */
                        /* var a = l.a + r.a; */
                        /* // 2) copies result array into a new symbol table entry array */
                        /* st.addEntry(rname, new shared SymEntry(a)); */
                    }
                    when "-" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a - r.a;
                    }
                    when "*" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a * r.a;
                    }
                    when "/" { // truediv
		        // Chapel will halt if any element of r.a is zero
		                if || reduce (r.a == 0) {
			                return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a:real / r.a:real;
                    } 
                    when "//" { // floordiv
		        // Chapel will halt if any element of r.a is zero
		        if || reduce (r.a == 0) {
			  return "Error: Attempt to divide by zero";
			}
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a / r.a;
                    }
		    when "%" { // modulo
		        // Chapel will halt if any element of r.a is zero
		        if || reduce (r.a == 0) {
			  return "Error: Attempt to compute a modulus by zero";
			}
                        var e = st.addEntry(rname, l.size, int);
		        e.a = l.a % r.a;
		    }
                    when "<" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a < r.a;
                    }
                    when ">" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a > r.a;
                    }
                    when "<=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a <= r.a;
                    }
                    when ">=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a >= r.a;
                    }
                    when "==" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a == r.a;
                    }
                    when "!=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a != r.a;
                    }
                    when "<<" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a << r.a;
                    }                    
                    when ">>" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a >> r.a;
                    }                    
                    when "&" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a & r.a;
                    }                    
                    when "|" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a | r.a;
                    }                    
                    when "^" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a ^ r.a;
                    }    
                    when "**" { 
                        if || reduce (r.a<0){
                            //instead of error, could we paste the below code but of type float?
                            return "Error: Attempt to exponentiate base of type Int64 to negative exponent";
                        }
                        var e = st.addEntry(rname, l.size, int);
                        e.a= l.a**r.a;
                    }     
                    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
                }
            }
            when (DType.Int64, DType.Float64) {
                var l = toSymEntry(left,int);
                var r = toSymEntry(right,real);
                select op
                {
                    when "+" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a + r.a;
                    }
                    when "-" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a - r.a;
                    }
                    when "*" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a * r.a;
                    }
                    when "/" { // truediv
                        if || reduce (r.a == 0) {
			                return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a:real / r.a;
                    } 
                    when "//" { // floordiv
                        if || reduce (r.a == 0) {
			                return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, l.size, real);
                        e.a = floor((l.a / r.a));
                    }
                    when "<" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a < r.a;
                    }
                    when ">" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a > r.a;
                    }
                    when "<=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a <= r.a;
                    }
                    when ">=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a >= r.a;
                    }
                    when "==" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a == r.a;
                    }
                    when "!=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a != r.a;
                    }
                    when "**" { 
                        var e = st.addEntry(rname, l.size, real);
                        e.a= l.a**r.a;
                    }    
                    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
                }
            }
            when (DType.Float64, DType.Int64) {
                var l = toSymEntry(left,real);
                var r = toSymEntry(right,int);
                select op
                {
                    when "+" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a + r.a;
                    }
                    when "-" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a - r.a;
                    }
                    when "*" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a * r.a;
                    }
                    when "/" { // truediv
                        if || reduce (r.a == 0) {
			                return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a / r.a:real;
                    } 
                    when "//" { // floordiv
                        if || reduce (r.a == 0) {
			                return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, l.size, real);
                        e.a = floor(l.a / r.a);
                    }
                    when "<" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a < r.a;
                    }
                    when ">" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a > r.a;
                    }
                    when "<=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a <= r.a;
                    }
                    when ">=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a >= r.a;
                    }
                    when "==" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a == r.a;
                    }
                    when "!=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a != r.a;
                    }
                    when "**" { 
                        var e = st.addEntry(rname, l.size, real);
                        e.a= l.a**r.a;
                    }      
                    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
                }
            }
            when (DType.Float64, DType.Float64) {
                var l = toSymEntry(left,real);
                var r = toSymEntry(right,real);
                select op
                {
                    when "+" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a + r.a;
                    }
                    when "-" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a - r.a;
                    }
                    when "*" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a * r.a;
                    }
                    when "/" { // truediv
                        if || reduce (r.a == 0) {
			                return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a / r.a;
                    } 
                    when "//" { // floordiv
                        if || reduce (r.a == 0) {
			                return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, l.size, real);
                        e.a = floor(l.a / r.a);
                    }
                    when "<" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a < r.a;
                    }
                    when ">" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a > r.a;
                    }
                    when "<=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a <= r.a;
                    }
                    when ">=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a >= r.a;
                    }
                    when "==" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a == r.a;
                    }
                    when "!=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a != r.a;
                    }
                    when "**" { 
                        var e = st.addEntry(rname, l.size, real);
                        e.a= l.a**r.a;
                    }     
                    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
                }
            }
	    when (DType.Bool, DType.Bool) {
	        var l = toSymEntry(left, bool);
		var r = toSymEntry(right, bool);
		select op {
		    when "|" {
                        var e = st.addEntry(rname, l.size, bool);
		        e.a = l.a | r.a;
		    }
		    when "&" {
                        var e = st.addEntry(rname, l.size, bool);
		        e.a = l.a & r.a;
		    }
		    when "^" {
                        var e = st.addEntry(rname, l.size, bool);
		        e.a = l.a ^ r.a;
		    }
		    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
		}
	    }
	    when (DType.Bool, DType.Int64) {
	        var l = toSymEntry(left, bool);
		var r = toSymEntry(right, int);
		select op {
		    when "+" {
                        var e = st.addEntry(rname, l.size, int);
		        e.a = l.a:int + r.a;
		    }
		    when "-" {
                        var e = st.addEntry(rname, l.size, int);
		        e.a = l.a:int - r.a;
		    }
		    when "*" {
                        var e = st.addEntry(rname, l.size, int);
		        e.a = l.a:int * r.a;
		    }
		    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
		}
	    }
	    when (DType.Int64, DType.Bool) {
	        var l = toSymEntry(left, int);
		var r = toSymEntry(right, bool);
		select op {
		    when "+" {
                        var e = st.addEntry(rname, l.size, int);
		        e.a = l.a + r.a:int;
		    }
		    when "-" {
                        var e = st.addEntry(rname, l.size, int);
		        e.a = l.a - r.a:int;
		    }
		    when "*" {
                        var e = st.addEntry(rname, l.size, int);
		        e.a = l.a * r.a:int;
		    }
		    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
		}
	    }
	    when (DType.Bool, DType.Float64) {
	        var l = toSymEntry(left, bool);
		var r = toSymEntry(right, real);
		select op {
		    when "+" {
                        var e = st.addEntry(rname, l.size, real);
		        e.a = l.a:real + r.a;
		    }
		    when "-" {
                        var e = st.addEntry(rname, l.size, real);
		        e.a = l.a:real - r.a;
		    }
		    when "*" {
                        var e = st.addEntry(rname, l.size, real);
		        e.a = l.a:real * r.a;
		    }
		    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
		}
	    }
	    when (DType.Float64, DType.Bool) {
	        var l = toSymEntry(left, real);
		var r = toSymEntry(right, bool);
		select op {
		    when "+" {
                        var e = st.addEntry(rname, l.size, real);
		        e.a = l.a + r.a:real;
		    }
		    when "-" {
                        var e = st.addEntry(rname, l.size, real);
		        e.a = l.a - r.a:real;
		    }
		    when "*" {
                        var e = st.addEntry(rname, l.size, real);
		        e.a = l.a * r.a:real;
		    }
		    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
		}
	    }
            otherwise {return unrecognizedTypeError(pn,
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
        param pn = Reflection.getRoutineName();
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
        if (left == nil) {return unknownSymbolError(pn, aname);}
        select (left.dtype, dtype) {
            when (DType.Int64, DType.Int64) {
                var l = toSymEntry(left,int);
                var val = try! value:int;
                select op
                {
                    when "+" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a + val;
                    }
                    when "-" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a - val;
                    }
                    when "*" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a * val;
                    }
                    when "/" { // truediv
		        // Chapel will halt if val is zero
		        if (val == 0) {
			  return "Error: Attempt to divide by zero";
			}
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a:real / val:real;
                    } 
                    when "//" { // floordiv
		        // Chapel will halt if any element of r.a is zero
		        if (val == 0) {
			  return "Error: Attempt to divide by zero";
			}
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a / val;
                    }
		    when "%" { // modulo
		      // Chapel will halt if val is zero
		      if (val == 0) {
			return "Error: Attempt to compute a modulus by zero";
		      }
                        var e = st.addEntry(rname, l.size, int);
		        e.a = l.a % val;
		    } 
                    when "<" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a < val;
                    }
                    when ">" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a > val;
                    }
                    when "<=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a <= val;
                    }
                    when ">=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a >= val;
                    }
                    when "==" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a == val;
                    }
                    when "!=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a != val;
                    }
                    when "<<" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a << val;
                    }
                    when ">>" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a >> val;
                    }
                    when "&" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a & val;
                    }
                    when "|" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a | val;
                    }
                    when "^" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a ^ val;
                    }
                    when "**" { 
                        if (val<0){
                            return "Error: Attempt to exponentiate base of type Int64 to negative exponent";
                        }
                        var e = st.addEntry(rname, l.size, int);
                        e.a= l.a**val;
                    }
                    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
                }
            }
            when (DType.Int64, DType.Float64) {
                var l = toSymEntry(left,int);
                var val = try! value:real;
                select op
                {
                    when "+" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a + val;
                    }
                    when "-" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a - val;
                    }
                    when "*" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a * val;
                    }
                    when "/" { // truediv
                        // add check for division by zero!
                        if (val == 0) {
			              return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a:real / val;
                    } 
                    when "//" { // floordiv
                        // add check for division by zero!
                        if (val == 0) {
			              return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, l.size, real);
                        e.a = floor(l.a:real / val);
                    }
                    when "<" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a < val;
                    }
                    when ">" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a > val;
                    }
                    when "<=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a <= val;
                    }
                    when ">=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a >= val;
                    }
                    when "==" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a == val;
                    }
                    when "!=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a != val;
                    }
                    when "**" { 
                        var e = st.addEntry(rname, l.size, real);
                        e.a= l.a**val;
                    }
                    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
                }
            }
            when (DType.Float64, DType.Int64) {
                var l = toSymEntry(left,real);
                var val = try! value:int;
                select op
                {
                    when "+" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a + val;
                    }
                    when "-" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a - val;
                    }
                    when "*" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a * val;
                    }
                    when "/" { // truediv
                        // add check for division by zero!
                        if (val == 0) {
			              return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a / val:real;
                    } 
                    when "//" { // floordiv
                        // add check for division by zero!
                        if (val == 0) {
			              return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, l.size, real);
                        e.a = floor(l.a / val:real);
                    }
                    when "<" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a < val;
                    }
                    when ">" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a > val;
                    }
                    when "<=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a <= val;
                    }
                    when ">=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a >= val;
                    }
                    when "==" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a == val;
                    }
                    when "!=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a != val;
                    }
                    when "**" { 
                        var e = st.addEntry(rname, l.size, real);
                        e.a= l.a**val;
                    }
                    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
                }
            }
            when (DType.Float64, DType.Float64) {
                var l = toSymEntry(left,real);
                var val = try! value:real;
                select op
                {
                    when "+" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a + val;
                    }
                    when "-" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a - val;
                    }
                    when "*" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a * val;
                    }
                    when "/" { // truediv
                        // add check for division by zero!
                        if (val == 0) {
			              return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a / val;
                    } 
                    when "//" { // floordiv
                        // add check for division by zero!
                        if (val == 0) {
			              return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, l.size, real);
                        e.a = floor(l.a / val);
                    }
                    when "<" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a < val;
                    }
                    when ">" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a > val;
                    }
                    when "<=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a <= val;
                    }
                    when ">=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a >= val;
                    }
                    when "==" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a == val;
                    }
                    when "!=" {
                        var e = st.addEntry(rname, l.size, bool);
                        e.a = l.a != val;
                    }
                    when "**" { 
                        var e = st.addEntry(rname, l.size, real);
                        e.a= l.a**val;
                    }
                    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
                }
            }
	    when (DType.Bool, DType.Bool) {
	        var l = toSymEntry(left, bool);
		var val = try! value.toLower():bool;
		select op {
		    when "|" {
                        var e = st.addEntry(rname, l.size, bool);
		        e.a = l.a | val;
		    }
		    when "&" {
                        var e = st.addEntry(rname, l.size, bool);
		        e.a = l.a & val;
		    }
		    when "^" {
                        var e = st.addEntry(rname, l.size, bool);
		        e.a = l.a ^ val;
		    }
                    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
		}
	    }
	    when (DType.Bool, DType.Int64) {
	        var l = toSymEntry(left, bool);
		var val = try! value:int;
		select op {
		    when "+" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a:int + val;
		    }
		    when "-" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a:int - val;
		    }
		    when "*" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a:int * val;
		    }
                    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
		}
	    }
	    when (DType.Int64, DType.Bool) {
	        var l = toSymEntry(left, int);
		var val = try! value.toLower():bool;
		select op {
		    when "+" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a + val:int;
		    }
		    when "-" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a - val:int;
		    }
		    when "*" {
                        var e = st.addEntry(rname, l.size, int);
                        e.a = l.a * val:int;
		    }
                    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
		}
	    }
	    when (DType.Bool, DType.Float64) {
	        var l = toSymEntry(left, bool);
		var val = try! value:real;
		select op {
		    when "+" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a:real + val;
		    }
		    when "-" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a:real - val;
		    }
		    when "*" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a:real * val;
		    }
                    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
		}
	    }
	    when (DType.Float64, DType.Bool) {
	        var l = toSymEntry(left, real);
		var val = try! value.toLower():bool;
		select op {
		    when "+" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a + val:real;
		    }
		    when "-" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a - val:real;
		    }
		    when "*" {
                        var e = st.addEntry(rname, l.size, real);
                        e.a = l.a * val:real;
		    }
                    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
		}
	    }
            otherwise {return unrecognizedTypeError(pn,
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
        param pn = Reflection.getRoutineName();
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
        if (right == nil) {return unknownSymbolError(pn, aname);}
        select (dtype, right.dtype) {
            when (DType.Int64, DType.Int64) {
                var val = try! value:int;
                var r = toSymEntry(right,int);
                select op
                {
                    when "+" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val + r.a;
                    }
                    when "-" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val - r.a;
                    }
                    when "*" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val * r.a;
                    }
                    when "/" { // truediv
		        // Chapel will halt if any element of r.a is zero
		        if || reduce (r.a == 0) {
			  return "Error: Attempt to divide by zero";
			}
                        var e = st.addEntry(rname, r.size, real);
                        e.a =  val:real / r.a:real;
                    } 
                    when "//" { // floordiv
		        // Chapel will halt if any element of r.a is zero
		        if || reduce (r.a == 0) {
			  return "Error: Attempt to divide by zero";
			}
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val / r.a;
                    }
		    when "%" { // modulo
		        // Chapel will halt if any element of r.a is zero
		        if || reduce (r.a == 0) {
			  return "Error: Attempt to compute a modulus by zero";
			}
                        var e = st.addEntry(rname, r.size, int);
		        e.a = val % r.a;
		    }
                    when "<" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val < r.a;
                    }
                    when ">" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val > r.a;
                    }
                    when "<=" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val <= r.a;
                    }
                    when ">=" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val >= r.a;
                    }
                    when "==" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val == r.a;
                    }
                    when "!=" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val != r.a;
                    }
                    when "<<" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val << r.a;
                    }
                    when ">>" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val >> r.a;
                    }
                    when "&" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val & r.a;
                    }
                    when "|" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val | r.a;
                    }
                    when "^" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val ^ r.a;
                    }
                    when "**" { 
                        if || reduce (r.a<0){
                            return "Error: Attempt to exponentiate base of type Int64 to negative exponent";
                        }
                        var e = st.addEntry(rname, r.size, int);
                        e.a= val**r.a;
                    }
                    otherwise {return notImplementedError(pn,dtype,op,right.dtype);}
                }
            }
            when (DType.Int64, DType.Float64) {
                var val = try! value:int;
                var r = toSymEntry(right,real);
                select op
                {
                    when "+" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val + r.a;
                    }
                    when "-" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val - r.a;
                    }
                    when "*" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val * r.a;
                    }
                    when "/" { // truediv
                        if || reduce (r.a == 0) {
			                 return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val:real / r.a;
                    }
                    when "//" { // floordiv
                        if || reduce (r.a == 0) {
			                 return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, r.size, real);
                        e.a = floor(val:real / r.a);
                    }
                    when "<" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val < r.a;
                    }
                    when ">" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val > r.a;
                    }
                    when "<=" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val <= r.a;
                    }
                    when ">=" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val >= r.a;
                    }
                    when "==" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val == r.a;
                    }
                    when "!=" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val != r.a;
                    }
                    when "**" { 
                        var e = st.addEntry(rname, r.size, real);
                        e.a= val**r.a;
                    }
                    otherwise {return notImplementedError(pn,dtype,op,right.dtype);}
                }
            }
            when (DType.Float64, DType.Int64) {
                var val = try! value:real;
                var r = toSymEntry(right,int);
                select op
                {
                    when "+" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val + r.a;
                    }
                    when "-" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val - r.a;
                    }
                    when "*" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val * r.a;
                    }
                    when "/" { // truediv
                        // add check for division by zero!
                        if || reduce (r.a == 0) {
			                 return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val / r.a:real;
                    } 
                    when "//" { // floordiv
                        // add check for division by zero!
                        if || reduce (r.a == 0) {
			                 return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, r.size, real);
                        e.a = floor(val / r.a:real);
                    }
                    when "<" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val < r.a;
                    }
                    when ">" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val > r.a;
                    }
                    when "<=" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val <= r.a;
                    }
                    when ">=" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val >= r.a;
                    }
                    when "==" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val == r.a;
                    }
                    when "!=" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val != r.a;
                    }
                    when "**" { 
                        var e = st.addEntry(rname, r.size, real);
                        e.a= val**r.a;
                    }
                    otherwise {return notImplementedError(pn,dtype,op,right.dtype);}
                }
            }
            when (DType.Float64, DType.Float64) {
                var val = try! value:real;
                var r = toSymEntry(right,real);
                select op
                {
                    when "+" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val + r.a;
                    }
                    when "-" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val - r.a;
                    }
                    when "*" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val * r.a;
                    }
                    when "/" { // truediv
                        if || reduce (r.a == 0) {
			                 return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val / r.a;
                    } 
                    when "//" { // floordiv
                        if || reduce (r.a == 0) {
			                 return "Error: Attempt to divide by zero";
			            }
                        var e = st.addEntry(rname, r.size, real);
                        e.a = floor(val / r.a);
                    }
                    when "<" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val < r.a;
                    }
                    when ">" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val > r.a;
                    }
                    when "<=" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val <= r.a;
                    }
                    when ">=" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val >= r.a;
                    }
                    when "==" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val == r.a;
                    }
                    when "!=" {
                        var e = st.addEntry(rname, r.size, bool);
                        e.a = val != r.a;
                    }
                    when "**" { 
                        var e = st.addEntry(rname, r.size, real);
                        e.a= val**r.a;
                    }
                    otherwise {return notImplementedError(pn,dtype,op,right.dtype);}
                }
            }
	    when (DType.Bool, DType.Bool) {
	        var val = try! value.toLower():bool;
		var r = toSymEntry(right, bool);
		select op {
		    when "|" {
                        var e = st.addEntry(rname, r.size, bool);
 		        e.a = val | r.a;
		    }
		    when "&" {
                        var e = st.addEntry(rname, r.size, bool);
 		        e.a = val & r.a;
		    }
		    when "^" {
                        var e = st.addEntry(rname, r.size, bool);
 		        e.a = val ^ r.a;
		    }
                    otherwise {return notImplementedError(pn,dtype,op,right.dtype);}
		}
	    }
	    when (DType.Bool, DType.Int64) {
	        var val = try! value.toLower():bool;
		var r = toSymEntry(right, int);
		select op {
		    when "+" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val:int + r.a;
		    }
		    when "-" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val:int - r.a;
		    }
		    when "*" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val:int * r.a;
		    }
                    otherwise {return notImplementedError(pn,dtype,op,right.dtype);}
		}
	    }
	    when (DType.Int64, DType.Bool) {
		var val = try! value:int;
		var r = toSymEntry(right, bool);
		select op {
		    when "+" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val + r.a:int;
		    }
		    when "-" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val - r.a:int;
		    }
		    when "*" {
                        var e = st.addEntry(rname, r.size, int);
                        e.a = val * r.a:int;
		    }
                    otherwise {return notImplementedError(pn,dtype,op,right.dtype);}
		}
	    }
	    when (DType.Bool, DType.Float64) {
	        var val = try! value.toLower():bool;
		var r = toSymEntry(right, real);
		select op {
		    when "+" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val:real + r.a;
		    }
		    when "-" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val:real - r.a;
		    }
		    when "*" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val:real * r.a;
		    }
                    otherwise {return notImplementedError(pn,dtype,op,right.dtype);}
		}
	    }
	    when (DType.Float64, DType.Bool) {
		var val = try! value:real;
		var r = toSymEntry(right, bool);
		select op {
		    when "+" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val + r.a:real;
		    }
		    when "-" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val - r.a:real;
		    }
		    when "*" {
                        var e = st.addEntry(rname, r.size, real);
                        e.a = val * r.a:real;
		    }
                    otherwise {return notImplementedError(pn,dtype,op,right.dtype);}
		}
	    }
            otherwise {return unrecognizedTypeError(pn,
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
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var op = fields[2];
        var aname = fields[3];
        var bname = fields[4];
        if v {try! writeln("%s %s %s %s".format(cmd,op,aname,bname));try! stdout.flush();}
        
        var left: borrowed GenSymEntry = st.lookup(aname);
        if (left == nil) {return unknownSymbolError(pn,aname);}
        var right: borrowed GenSymEntry = st.lookup(bname);
        if (right == nil) {return unknownSymbolError(pn,bname);}
        select (left.dtype, right.dtype) {
            when (DType.Int64, DType.Int64) {
                var l = toSymEntry(left,int);
                var r = toSymEntry(right,int);
                select op
                {
                    when "+=" { l.a += r.a; }
                    when "-=" { l.a -= r.a; }
                    when "*=" { l.a *= r.a; }
                    when "//=" { 
                        if || reduce(r.a==0){
                            return "Error: Attempt to divide by zero";
                        }
                        l.a /= r.a; 
                    }//floordiv
                    when "**=" { 
                        if || reduce (r.a<0){
                            return "Error: Attempt to exponentiate base of type Int64 to negative exponent";
                        }
                        else{ l.a **= r.a; }
                    }
                    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
                }
            }
            when (DType.Int64, DType.Float64) {
                var l = toSymEntry(left,int);
                var r = toSymEntry(right,real);
                return notImplementedError(pn,left.dtype,op,right.dtype);
            }
            when (DType.Float64, DType.Int64) {
                var l = toSymEntry(left,real);
                var r = toSymEntry(right,int);
                select op
                {
                    when "+=" {l.a += r.a;}
                    when "-=" {l.a -= r.a;}
                    when "*=" {l.a *= r.a;}
                    when "/=" {
                        if || reduce(r.a==0){
                            return "Error: Attempt to divide by zero";
                        }
                        l.a /= r.a:real;
                    } //truediv
                    when "//=" {
                        if || reduce(r.a==0){
                            return "Error: Attempt to divide by zero";
                        }
                        l.a = floor(l.a / r.a);
                    } //floordiv
                    when "**=" { l.a **= r.a; }
                    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
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
                    when "//=" {
                        if || reduce(r.a==0){
                            return "Error: Attempt to divide by zero";
                        }
                        l.a = floor(l.a / r.a);
                    }//floordiv
                    when "**=" { l.a **= r.a; }
                    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
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
		    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
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
  		    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
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
  		    otherwise {return notImplementedError(pn,left.dtype,op,right.dtype);}
		}
	    }
            otherwise {return unrecognizedTypeError(pn,
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
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var op = fields[2];
        var aname = fields[3];
        var dtype = str2dtype(fields[4]);
        var value = fields[5];
        if v {try! writeln("%s %s %s %s %s".format(cmd,op,aname,dtype2str(dtype),value));try! stdout.flush();}

        var left: borrowed GenSymEntry = st.lookup(aname);
        if (left == nil) {return unknownSymbolError(pn,aname);}

        select (left.dtype, dtype) {
            when (DType.Int64, DType.Int64) {
                var l = toSymEntry(left,int);
                var val = try! value:int;
                select op
                {
                    when "+=" { l.a += val; }
                    when "-=" { l.a -= val; }
                    when "*=" { l.a *= val; }
                    when "//=" { 
                        if (val==0){
                            return "Error: Attempt to divide by zero";
                        }
                        l.a /= val; 
                    }//floordiv
                    when "**=" { 
                        if (val<0){
                            return "Error: Attempt to exponentiate base of type Int64 to negative exponent";
                        }
                        else{ l.a **= val; }

                    }
                    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
                }
            }
            when (DType.Int64, DType.Float64) {
                var l = toSymEntry(left,int);
                var val = try! value:real;
                return notImplementedError(pn,left.dtype,op,dtype);
            }
            when (DType.Float64, DType.Int64) {
                var l = toSymEntry(left,real);
                var val = try! value:int;
                select op
                {
                    when "+=" {l.a += val;}
                    when "-=" {l.a -= val;}
                    when "*=" {l.a *= val;}
                    when "/=" {
                        if (val==0){
                            return "Error: Attempt to divide by zero";
                        }
                        l.a /= val:real;
                    } //truediv
                    when "//=" {
                        if (val==0){
                            return "Error: Attempt to divide by zero";
                        }
                        l.a = floor(l.a / val);
                    } //floordiv
                    when "**=" { l.a **= val; }
                    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
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
                    when "/=" {
                        if (val==0){
                            return "Error: Attempt to divide by zero";
                        }
                        l.a /= val;
                    }//truediv
                    when "//=" {
                        if (val==0){
                            return "Error: Attempt to divide by zero";
                        }
                        l.a = floor(l.a / val);
                    }//floordiv
                    when "**=" { l.a **= val; }
                    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
                }
            }
	    when (DType.Int64, DType.Bool) {
	        var l = toSymEntry(left, int);
		var val = try! value: bool;
		select op {
		    when "+=" {l.a += val:int;}
		    when "-=" {l.a -= val:int;}
		    when "*=" {l.a *= val:int;}
		    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
		}
	    }
	    when (DType.Float64, DType.Bool) {
	        var l = toSymEntry(left, real);
		var val = try! value: bool;
		select op {
		    when "+=" {l.a += val:real;}
		    when "-=" {l.a -= val:real;}
		    when "*=" {l.a *= val:real;}
		    otherwise {return notImplementedError(pn,left.dtype,op,dtype);}
		}
	    }
            otherwise {return unrecognizedTypeError(pn,
                                                    "("+dtype2str(left.dtype)+","+dtype2str(dtype)+")");}
        }
        return "opeqvs success";
    }

}
