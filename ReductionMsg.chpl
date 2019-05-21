module ReductionMsg
{
    use ServerConfig;

    use Time only;
    use Math only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use AryUtil;
      
    // these functions take an array and produce a scalar
    // parse and respond to reduction message
    // scalar = reductionop(vector)
    proc reductionMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var reductionop = fields[2];
        var name = fields[3];
        if v {try! writeln("%s %s %s".format(cmd,reductionop,name));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError("reduction",name);}
       
        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                select reductionop
                {
                    when "any" {
                        var val:string;
                        var sum = + reduce (e.a != 0);
                        if sum != 0 {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "all" {
                        var val:string;
                        var sum = + reduce (e.a != 0);
                        if sum == e.aD.size {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "sum" {
                        var sum = + reduce e.a;
                        var val = sum:string;
                        return try! "int64 %i".format(val);
                    }
                    when "prod" {
                        var prod = * reduce e.a;
                        var val = prod:string;
                        return try! "int64 %i".format(val);
                    }
		    when "min" {
		        var minVal = min reduce e.a;
			var val = minVal:string;
			return try! "int64 %i".format(val);
		    }
		    when "max" {
		        var maxVal = max reduce e.a;
			var val = maxVal:string;
			return try! "int64 %i".format(val);
		    }
                    when "argmin" {
                        var (minVal, minLoc) = minloc reduce zip(e.a,e.aD);
                        var val = minLoc:string;
                        return try! "int64 %i".format(val);
                    }
                    when "argmax" {
                        var (maxVal, maxLoc) = maxloc reduce zip(e.a,e.aD);
                        var val = maxLoc:string;
                        return try! "int64 %i".format(val);
                    }
                    when "is_sorted" {
                        ref ea = e.a;
                        var sorted = isSorted(ea);
                        var val:string;
                        if sorted {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    otherwise {return notImplementedError("reduction",reductionop,gEnt.dtype);}
                }
            }
            when (DType.Float64) {
                var e = toSymEntry(gEnt,real);
                select reductionop
                {
                    when "any" {
                        var val:string;
                        var sum = + reduce (e.a != 0.0);
                        if sum != 0.0 {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "all" {
                        var val:string;
                        var sum = + reduce (e.a != 0.0);
                        if sum == e.aD.size {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "sum" {
                        var sum = + reduce e.a;
                        var val = sum:string;
                        return try! "float64 %.17r".format(val);
                    }
                    when "prod" {
                        var prod = * reduce e.a;
                        var val = prod:string;
                        return try! "float64 %.17r".format(val);
                    }
		    when "min" {
                        var minVal = min reduce e.a;
                        var val = minVal:string;
                        return try! "float64 %.17r".format(val);
                    }
		    when "max" {
                        var maxVal = max reduce e.a;
                        var val = maxVal:string;
                        return try! "float64 %.17r".format(val);
                    }
                    when "argmin" {
                        var (minVal, minLoc) = minloc reduce zip(e.a,e.aD);
                        var val = minLoc:string;
                        return try! "int64 %i".format(val);
                    }
                    when "argmax" {
                        var (maxVal, maxLoc) = maxloc reduce zip(e.a,e.aD);
                        var val = maxLoc:string;
                        return try! "int64 %i".format(val);
                    }
                    when "is_sorted" {
                        var sorted = isSorted(e.a);
                        var val:string;
                        if sorted {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    otherwise {return notImplementedError("reduction",reductionop,gEnt.dtype);}
                }
            }
            when (DType.Bool) {
                var e = toSymEntry(gEnt,bool);
                select reductionop
                {
                    when "any" {
                        var val:string;
                        var any = | reduce e.a;
                        if any {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "all" {
                        var val:string;
                        var all = & reduce e.a;
                        if all {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "sum" {
                        var sum = + reduce e.a:int;
                        var val = sum:string;
                        return try! "int64 %i".format(val);
                    }
                    when "prod" {
                        var prod = * reduce e.a:int;
                        var val = prod:string;
                        return try! "int64 %i".format(val);
                    }
		    when "min" {
		        var val:string;
			if (& reduce e.a) { val = "True"; } else { val = "False"; }
			return try! "bool %s".format(val);
		    }
		    when "max" {
		        var val:string;
			if (| reduce e.a) { val = "True"; } else { val = "False"; }
			return try! "bool %s".format(val);
		    }
                    otherwise {return notImplementedError("reduction",reductionop,gEnt.dtype);}
                }
            }
            otherwise {return unrecognizedTypeError("reduction", dtype2str(gEnt.dtype));}
        }
    }

}

