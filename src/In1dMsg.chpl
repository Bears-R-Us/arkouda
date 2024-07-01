
module In1dMsg
{
    use ServerConfig;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use In1d;
    use SegmentedString;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const iLogger = new Logger(logLevel, logChannel);
    
    /* in1d takes two pdarray and returns a bool pdarray
       with the "in"/contains for each element tested against the second pdarray.
       
       in1dMsg processes the request, considers the size of the arguements, and decides 
       which implementation
       of in1d to utilize.
    */
    proc in1dMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var invert: bool = msgArgs.get("invert").getBoolValue();

        // get next symbol name
        var rname = st.nextName();

        const name: string = msgArgs.getValueOf("pda1");
        const sname: string = msgArgs.getValueOf("pda2");

        var gAr1: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        var gAr2: borrowed GenSymEntry = getGenericTypedArrayEntry(sname, st);
        
        iLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "cmd: %s pdarray1: %s pdarray2: %s invert: %? new pdarray name: %?".format(
                                   cmd,st.attrib(name),st.attrib(sname),invert,rname));

        select (gAr1.dtype, gAr2.dtype) {
            when (DType.Int64, DType.Int64) {
                var ar1 = toSymEntry(gAr1,int);
                var ar2 = toSymEntry(gAr2,int);

                var truth = in1d(ar1.a, ar2.a, invert);
                st.addEntry(rname, createSymEntry(truth));
            }
            when (DType.UInt64, DType.UInt64) {
                var ar1 = toSymEntry(gAr1,uint);
                var ar2 = toSymEntry(gAr2,uint);

                var truth = in1d(ar1.a, ar2.a, invert);
                st.addEntry(rname, createSymEntry(truth));
            }
            when (DType.Float64, DType.Float64) {
                var ar1 = toSymEntry(gAr1,real);
                var ar2 = toSymEntry(gAr2,real);

                var transmuted1 = [ei in ar1.a] ei.transmute(uint(64));
                var transmuted2 = [ei in ar2.a] ei.transmute(uint(64));
                var truth = in1d(transmuted1, transmuted2, invert);
                st.addEntry(rname, createSymEntry(truth));
            }
            otherwise {
                var errorMsg = notImplementedError(pn,gAr1.dtype,"in",gAr2.dtype);
                iLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
        repMsg = "created " + st.attrib(rname);
        iLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    use CommandMap;
    registerFunction("in1d", in1dMsg, getModuleName());
}
