module RandMsg
{
    use ServerConfig;
    
    use Time;
    use Math;
    use Random;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    // parse, execute, and respond to randint message
    // uniform int in half-open interval [min,max)
    proc randintMsg(req_msg: string, st: borrowed SymTab): string {
        var rep_msg: string; // response message
        var fields = req_msg.split(); // split request into fields
        var cmd = fields[1];
        var a_min = try! fields[2]:int;
        var a_max = try! fields[3]:int;
        var len = try! fields[4]:int;
        var dtype = str2dtype(fields[5]);

        var seed = 241;
        
        // get next symbol name
        var rname = st.next_name();
        
        // if verbose print action
        if v {try! writeln("%s %i %i %i %s : %s".format(cmd,a_min,a_max,len,dtype2str(dtype),rname)); try! stdout.flush();}
        select (dtype) {
            when (DType.Int64) {                
                var t1 = getCurrentTime();
                var aD = makeDistDom(len);
                var a = makeDistArray(len, int);
                writeln("alloc time = ",getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                t1 = getCurrentTime();
                // this is potentially clunky due to only one RandomStream
                // need to do at least one for each locale maybe...
                var R = new owned RandomStream(real, seed); R.getNext();
                [e in a] e = (R.getNext() * (a_max - a_min) + a_min):int;
                writeln("compute time = ",getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                st.addEntry(rname, new shared SymEntry(a));
            }
            when (DType.Float64) {
                var t1 = getCurrentTime();
                var aD = makeDistDom(len);
                var a = makeDistArray(len, real);
                writeln("alloc time = ",getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                t1 = getCurrentTime();
                // this is potentially clunky due to only one RandomStream
                // need to do at least one for each locale maybe...
                var R = new owned RandomStream(real, seed); R.getNext();
                [e in a] e = ((R.getNext() * (a_max - a_min) + a_min):int):real;
                writeln("compute time = ",getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                st.addEntry(rname, new shared SymEntry(a));                
            }
            when (DType.Bool) {
                var t1 = getCurrentTime();
                var aD = makeDistDom(len);
                var a = makeDistArray(len, bool);
                writeln("alloc time = ",getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                t1 = getCurrentTime();
                // this is potentially clunky due to only one RandomStream
                // need to do at least one for each locale maybe...
                var R = new owned RandomStream(real, seed); R.getNext();
                [e in a] e = (R.getNext() >= 0.5);
                writeln("compute time = ",getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                st.addEntry(rname, new shared SymEntry(a));
            }            
            otherwise {return notImplementedError("randint",dtype);}
        }
        // response message
        return try! "created " + st.attrib(rname);
    }

}