module RandMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    use Reflection only;
    use RandArray;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    /*
    parse, execute, and respond to randint message
    uniform int in half-open interval [min,max)

    :arg reqMsg: message to process (contains cmd,aMin,aMax,len,dtype)
    */
    proc randintMsg(reqMsg: string, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var aMin = try! fields[2]:int;
        var aMax = try! fields[3]:int;
        var len = try! fields[4]:int;
        var dtype = str2dtype(fields[5]);

        // get next symbol name
        var rname = st.nextName();
        
        // if verbose print action
        if v {try! writeln("%s %i %i %i %s : %s".format(cmd,aMin,aMax,len,dtype2str(dtype),rname)); try! stdout.flush();}
        select (dtype) {
            when (DType.Int64) {                
                var t1 = Time.getCurrentTime();
                var e = st.addEntry(rname, len, int);
                writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                t1 = Time.getCurrentTime();
                fillInt(e.a, aMin, aMax);
                writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
            }
            when (DType.UInt8) {                
                var t1 = Time.getCurrentTime();
                var e = st.addEntry(rname, len, uint(8));
                writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                t1 = Time.getCurrentTime();
                fillUInt(e.a, aMin, aMax);
                writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
            }
            when (DType.Float64) {
                var t1 = Time.getCurrentTime();
                var e = st.addEntry(rname, len, real);
                writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                t1 = Time.getCurrentTime();
                fillReal(e.a, aMin, aMax);
                writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
            }
            when (DType.Bool) {
                var t1 = Time.getCurrentTime();
                var e = st.addEntry(rname, len, bool);
                writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                t1 = Time.getCurrentTime();
                fillBool(e.a);
                writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
            }            
            otherwise {return notImplementedError(pn,dtype);}
        }
        // response message
        return try! "created " + st.attrib(rname);
    }

}
