
module MsgProcessing
{
    use ServerConfig;

    use Time;
    use Math;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use OperatorMsg;
    use RandMsg;
    use IndexingMsg;
    
    // parse, execute, and respond to create message
    proc createMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var dtype = str2dtype(fields[2]);
        var size = try! fields[3]:int;

        // get next symbol name
        var rname = st.next_name();
        
        // if verbose print action
        if v {try! writeln("%s %s %i : %s".format(cmd,dtype2str(dtype),size,rname)); try! stdout.flush();}
        // create and add entry to symbol table
        st.addEntry(rname, size, dtype);
        // response message
        return try! "created " + st.attrib(rname);
    }

    // parse, execute, and respond to delete message
    proc deleteMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        if v {try! writeln("%s %s".format(cmd,name));try! stdout.flush();}
        // delete entry from symbol table
        st.deleteEntry(name);
        return try! "deleted %s".format(name);
    }

    // info header only
    proc infoMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        if v {try! writeln("%s %s".format(cmd,name));try! stdout.flush();}
        // if name == "__AllSymbols__" passes back info on all symbols
        return st.info(name);
    }
    
    // dump info and values
    proc dumpMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        if v {try! writeln("%s %s".format(cmd,name));try! stdout.flush();}
        // if name == "__AllSymbols__" passes back dump on all symbols
        return st.dump(name);
    }

    // response to __str__ method in python
    // str convert array data to string
    proc strMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var print_thresh = try! fields[3]:int;
        if v {try! writeln("%s %s %i".format(cmd,name,print_thresh));try! stdout.flush();}
        return st.datastr(name,print_thresh);
    }

    // response to __repr__ method in python
    // repr convert array data to string
    proc reprMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var print_thresh = try! fields[3]:int;
        if v {try! writeln("%s %s %i".format(cmd,name,print_thresh));try! stdout.flush();}
        return st.datarepr(name,print_thresh);
    }

    proc arangeMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var start = try! fields[2]:int;
        var stop = try! fields[3]:int;
        var stride = try! fields[4]:int;
        // compute length
        var len = (stop - start + stride - 1) / stride;
        // get next symbol name
        var rname = st.next_name();
        if v {try! writeln("%s %i %i %i : %i , %s".format(cmd, start, stop, stride, len, rname));try! stdout.flush();}
        
        var t1 = getCurrentTime();
        var aD = makeDistDom(len);
        var a = makeDistArray(len, int);
        writeln("alloc time = ",getCurrentTime() - t1,"sec"); try! stdout.flush();

        t1 = getCurrentTime();
        forall i in aD {
            a[i] = start + (i * stride);
        }
        writeln("compute time = ",getCurrentTime() - t1,"sec"); try! stdout.flush();

        st.addEntry(rname, new shared SymEntry(a));
        return try! "created " + st.attrib(rname);
    }            

    proc linspaceMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var start = try! fields[2]:real;
        var stop = try! fields[3]:real;
        var len = try! fields[4]:int;
        // compute stride
        var stride = (stop - start) / (len-1);
        // get next symbol name
        var rname = st.next_name();
        if v {try! writeln("%s %r %r %i : %r , %s".format(cmd, start, stop, len, stride, rname));try! stdout.flush();}

        var t1 = getCurrentTime();
        var aD = makeDistDom(len);
        var a = makeDistArray(len, real);
        writeln("alloc time = ",getCurrentTime() - t1,"sec"); try! stdout.flush();

        t1 = getCurrentTime();
        forall i in aD {
            a[i] = start + (i * stride);
        }
        a[0] = start;
        a[len-1] = stop;
        writeln("compute time = ",getCurrentTime() - t1,"sec"); try! stdout.flush();

        st.addEntry(rname, new shared SymEntry(a));
        return try! "created " + st.attrib(rname);
    }

    // histogram takes a pdarray and returns a pdarray with the histogram in it
    proc histogramMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var bins = try! fields[3]:int;
        
        // get next symbol name
        var rname = st.next_name();
        if v {try! writeln("%s %s %i : %s".format(cmd, name, bins, rname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError("histogram",name);}

        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var a_min = min reduce e.a;
                var a_max = max reduce e.a;
                var bin_width:real = (a_max - a_min):real / bins:real;
                if v {try! writeln("bin_width %r ".format(bin_width)); try! stdout.flush();}
                var hD = makeDistDom(bins);
                var atomic_hist: [hD] atomic int;
                // count into atomic histogram
                forall v in e.a {
                    var v_bin = ((v - a_min) / bin_width):int;
                    if v == a_max {v_bin = bins-1;}
                    //if (v_bin < 0) | (v_bin > (bins-1)) {try! writeln("OOB");try! stdout.flush();}
                    atomic_hist[v_bin].add(1);
                }
                var hist = makeDistArray(bins,int);
                // copy from atomic histogram to normal histogram
                [(e,ae) in zip(hist, atomic_hist)] e = ae.read();
                if v {try! writeln("hist =",hist); try! stdout.flush();}
                
                st.addEntry(rname, new shared SymEntry(hist));
            }
            when (DType.Float64) {
                var e = toSymEntry(gEnt,real);
                var a_min = min reduce e.a;
                var a_max = max reduce e.a;
                var bin_width:real = (a_max - a_min):real / bins:real;
                if v {try! writeln("bin_width %r ".format(bin_width)); try! stdout.flush();}
                var hD = makeDistDom(bins);
                var atomic_hist: [hD] atomic int;
                // count into atomic histogram
                forall v in e.a {
                    var v_bin = ((v - a_min) / bin_width):int;
                    if v == a_max {v_bin = bins-1;}
                    //if (v_bin < 0) | (v_bin > (bins-1)) {try! writeln("OOB");try! stdout.flush();}
                    atomic_hist[v_bin].add(1);
                }
                var hist = makeDistArray(bins,int);
                // copy from atomic histogram to normal histogram
                [(e,ae) in zip(hist, atomic_hist)] e = ae.read();
                if v {try! writeln("hist =",hist); try! stdout.flush();}
                
                st.addEntry(rname, new shared SymEntry(hist));
            }
            otherwise {return notImplementedError("histogram",gEnt.dtype);}
        }
        
        return try! "created " + st.attrib(rname);
    }

    // in1d takes two pdarray and returns a bool pdarray
    // with the "in"/contains for each element tested against the second pdarray
    proc in1dMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var sname = fields[3];

        // get next symbol name
        var rname = st.next_name();
        if v {try! writeln("%s %s %s : %s".format(cmd, name, sname, rname));try! stdout.flush();}

        var gAr1: borrowed GenSymEntry = st.lookup(name);
        if (gAr1 == nil) {return unknownSymbolError("in1d",name);}
        var gAr2: borrowed GenSymEntry = st.lookup(sname);
        if (gAr2 == nil) {return unknownSymbolError("in1d",sname);}

        select (gAr1.dtype, gAr2.dtype) {
            when (DType.Int64, DType.Int64) {
                var ar1 = toSymEntry(gAr1,int);
                var ar2 = toSymEntry(gAr2,int);

                var truth = makeDistArray(ar1.size, bool);
                [(a,t) in zip(ar1.a,truth)] t = | reduce (a == ar2.a);
                
                st.addEntry(rname, new shared SymEntry(truth));
            }
            otherwise {return notImplementedError("in1d",gAr1.dtype,"in",gAr2.dtype);}
        }
        
        return try! "created " + st.attrib(rname);
    }

    // unique take a pdarray and returns a pdarray with the unique values
    proc uniqueMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];

        // get next symbol name
        var rname = st.next_name();
        if v {try! writeln("%s %s : %s".format(cmd, name, rname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError("unique",name);}

        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var e_min:int = min reduce e.a;
                var e_max:int = max reduce e.a;

                // how many bins in histogram
                var bins = e_max-e_min+1;
                if v {try! writeln("bins = %t".format(bins));}

                var hD = makeDistDom(bins);
                // atomic histogram
                var atomic_hist: [hD] atomic int;
                // count into atomic histogram
                forall v in e.a {
                    var bin = v - e_min;
                    if v == e_max {bin = bins-1;}
                    atomic_hist[bin].add(1);
                }
                var itruth = makeDistArray(bins,int);
                // copy from atomic histogram to normal histogram
                [(t,ae) in zip(itruth, atomic_hist)] t = (ae.read() != 0):int;
                // calc indices of the non-zero count elements
                var iv: [hD] int = (+ scan itruth);
                var pop = iv[iv.size-1];
                var a = makeDistArray(pop, int);
                [i in hD] if (itruth[i] == 1) {a[iv[i]-1] = i+e_min;}// iv[i]-1 for zero base index
                st.addEntry(rname, new shared SymEntry(a));
            }
            otherwise {return notImplementedError("unique",gEnt.dtype);}
        }
        
        return try! "created " + st.attrib(rname);
    }

    // value_counts rtakes a pdarray and returns two pdarrays unique values and counts for each value
    proc value_countsMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];

        // get next symbol name
        var vname = st.next_name();
        var cname = st.next_name();
        if v {try! writeln("%s %s : %s %s".format(cmd, name, vname, cname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError("value_counts",name);}

        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var e_min:int = min reduce e.a;
                var e_max:int = max reduce e.a;

                // how many bins in histogram
                var bins = e_max-e_min+1;
                if v {try! writeln("bins = %t".format(bins));}

                var hD = makeDistDom(bins);
                // atomic histogram
                var atomic_hist: [hD] atomic int;
                // count into atomic histogram
                forall v in e.a {
                    var bin = v - e_min;
                    if v == e_max {bin = bins-1;}
                    atomic_hist[bin].add(1);
                }
                var itruth = makeDistArray(bins,int);
                // copy from atomic histogram to normal histogram
                [(t,ae) in zip(itruth, atomic_hist)] t = (ae.read() != 0):int;
                // calc indices of the non-zero count elements
                var iv: [hD] int = (+ scan itruth);
                var pop = iv[iv.size-1];
                var a_v = makeDistArray(pop, int);
                var a_c = makeDistArray(pop, int);
                [i in hD] if (itruth[i] == 1) {
                    a_v[iv[i]-1] = i+e_min;
                    a_c[iv[i]-1] = atomic_hist[i].read();
                        }// iv[i]-1 for zero base index
                
                st.addEntry(vname, new shared SymEntry(a_v));
                st.addEntry(cname, new shared SymEntry(a_c));
            }
            otherwise {return notImplementedError("value_counts",gEnt.dtype);}
        }
        
        return try! "created " + st.attrib(vname) + " +created " + st.attrib(cname);
    }

    // sets all elements in array to a value (broadcast)
    proc setMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
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
                rep_msg = try! "set %s to %t".format(name, val);
            }
            when (DType.Int64, DType.Float64) {
                var e = toSymEntry(gEnt,int);
                var val: real = try! value:real;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:int));try! stdout.flush();}
                e.a = val:int;
                rep_msg = try! "set %s to %t".format(name, val:int);
            }
            when (DType.Int64, DType.Bool) {
                var e = toSymEntry(gEnt,int);
                value = value.replace("True","true");
                value = value.replace("False","false");
                var val: bool = try! value:bool;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:int));try! stdout.flush();}
                e.a = val:int;
                rep_msg = try! "set %s to %t".format(name, val:int);
            }
            when (DType.Float64, DType.Int64) {
                var e = toSymEntry(gEnt,real);
                var val: int = try! value:int;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:real));try! stdout.flush();}
                e.a = val:real;
                rep_msg = try! "set %s to %t".format(name, val:real);
            }
            when (DType.Float64, DType.Float64) {
                var e = toSymEntry(gEnt,real);
                var val: real = try! value:real;
                if v {try! writeln("%s %s to %t".format(cmd,name,val));try! stdout.flush();}
                e.a = val;
                rep_msg = try! "set %s to %t".format(name, val);
            }
            when (DType.Float64, DType.Bool) {
                var e = toSymEntry(gEnt,real);
                value = value.replace("True","true");
                value = value.replace("False","false");                
                var val: bool = try! value:bool;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:real));try! stdout.flush();}
                e.a = val:real;
                rep_msg = try! "set %s to %t".format(name, val:real);
            }
            when (DType.Bool, DType.Int64) {
                var e = toSymEntry(gEnt,bool);
                var val: int = try! value:int;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:bool));try! stdout.flush();}
                e.a = val:bool;
                rep_msg = try! "set %s to %t".format(name, val:bool);
            }
            when (DType.Bool, DType.Float64) {
                var e = toSymEntry(gEnt,int);
                var val: real = try! value:real;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:bool));try! stdout.flush();}
                e.a = val:bool;
                rep_msg = try! "set %s to %t".format(name, val:bool);
            }
            when (DType.Bool, DType.Bool) {
                var e = toSymEntry(gEnt,int);
                value = value.replace("True","true");
                value = value.replace("False","false");
                var val: bool = try! value:bool;
                if v {try! writeln("%s %s to %t".format(cmd,name,val));try! stdout.flush();}
                e.a = val;
                rep_msg = try! "set %s to %t".format(name, val);
            }
            otherwise {return unrecognizedTypeError("set",fields[3]);}
        }
        return rep_msg;
    }
    
    // these ops are functions which take an array and produce and array
    // do scans fit here also? I think so... vector = scanop(vector)
    // parse and respond to efunc "elemental function" message
    // vector = efunc(vector)
    proc efuncMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var efunc = fields[2];
        var name = fields[3];
        var rname = st.next_name();
        if v {try! writeln("%s %s %s : %s".format(cmd,efunc,name,rname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError("efunc",name);}
       
        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                select efunc
                {
                    when "abs" {
                        var a = abs(e.a);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "log" {
                        var a = log(e.a);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "exp" {
                        var a = exp(e.a);
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
                        var a = abs(e.a);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "log" {
                        var a = log(e.a);
                        st.addEntry(rname, new shared SymEntry(a));
                    }
                    when "exp" {
                        var a = exp(e.a);
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
    proc reductionMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
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
                        return try! "int64 %i".format(dtype2str(e.dtype), val);
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
                    otherwise {return notImplementedError("reduction",reductionop,gEnt.dtype);}
                }
            }
            otherwise {return unrecognizedTypeError("reduction", dtype2str(gEnt.dtype));}
        }
    }
}
