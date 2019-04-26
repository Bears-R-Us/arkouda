
module MsgProcessing
{
    use ServerConfig;

    use Time only;
    use Math only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use OperatorMsg;
    use RandMsg;
    use IndexingMsg;
    use UniqueMsg;
    use In1dMsg;
    use HistogramMsg;
    
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
