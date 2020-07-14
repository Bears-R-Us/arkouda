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
    proc randintMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (lenStr,dtypeStr,aMinStr,aMaxStr) = payload.decode().splitMsgToTuple(4);
        var len = lenStr:int;
        var dtype = str2dtype(dtypeStr);

        // get next symbol name
        var rname = st.nextName();
        
        // if verbose print action
        if v {try! writeln("%s %i %s %s %s: %s".format(cmd,len,dtype2str(dtype),rname,aMinStr,aMaxStr)); try! stdout.flush();}
        select (dtype) {
            when (DType.Int64) {
                var aMin = aMinStr:int;
                var aMax = aMaxStr:int;
                var t1 = Time.getCurrentTime();
                var e = st.addEntry(rname, len, int);
                if v {writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();}
                
                t1 = Time.getCurrentTime();
                fillInt(e.a, aMin, aMax);
                if v {writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();}
            }
            when (DType.UInt8) {
                var aMin = aMinStr:int;
                var aMax = aMaxStr:int;
                var t1 = Time.getCurrentTime();
                var e = st.addEntry(rname, len, uint(8));
                if v {writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();}
                
                t1 = Time.getCurrentTime();
                fillUInt(e.a, aMin, aMax);
                if v {writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();}
            }
            when (DType.Float64) {
                var aMin = aMinStr:real;
                var aMax = aMaxStr:real;
                var t1 = Time.getCurrentTime();
                var e = st.addEntry(rname, len, real);
                if v {writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();}
                
                t1 = Time.getCurrentTime();
                fillReal(e.a, aMin, aMax);
                if v {writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();}
            }
            when (DType.Bool) {
                var t1 = Time.getCurrentTime();
                var e = st.addEntry(rname, len, bool);
                if v {writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();}
                
                t1 = Time.getCurrentTime();
                fillBool(e.a);
                if v {writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();}
            }            
            otherwise {return notImplementedError(pn,dtype);}
        }
        // response message
        return try! "created " + st.attrib(rname);
    }

    proc randomNormalMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();
      var (lenStr) = payload.decode().splitMsgToTuple(1);
      var len = lenStr:int;
      // Result + 2 scratch arrays
      overMemLimit(3*8*len);
      var rname = st.nextName();
      var entry = new shared SymEntry(len, real);
      fillNormal(entry.a);
      st.addEntry(rname, entry);
      return "created " + st.attrib(rname);
    }

}
