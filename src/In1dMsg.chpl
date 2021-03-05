
module In1dMsg
{
    use ServerConfig;

    use Reflection;
    use Errors;
    use Logging;
    use Message;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use In1d;

    var iLogger = new Logger();
    if v {
        iLogger.level = LogLevel.DEBUG;
    } else {
        iLogger.level = LogLevel.INFO;    
    }
    
    /*
    Small bound const. Brute force in1d implementation recommended.
    */
    private config const sBound = 2**4; 

    /*
    Medium bound const. Per locale associative domain in1d implementation recommended.
    */
    private config const mBound = 2**25; 

    /* in1d takes two pdarray and returns a bool pdarray
       with the "in"/contains for each element tested against the second pdarray.
       
       in1dMsg processes the request, considers the size of the arguements, and decides 
       which implementation
       of in1d to utilize.
    */
    proc in1dMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, sname, flag) = payload.splitMsgToTuple(3);
        var invert: bool;
        
        if flag == "True" {invert = true;}
        else if flag == "False" {invert = false;}
        else {
            var errorMsg = "Error: %s: %s".format(pn,flag);
            iLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);         
        }

        // get next symbol name
        var rname = st.nextName();

        var gAr1: borrowed GenSymEntry = st.lookup(name);
        var gAr2: borrowed GenSymEntry = st.lookup(sname);
        
        iLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "cmd: %s pdarray1: %s pdarray2: %s invert: %t new pdarray name: %t".format(
                                   cmd,st.attrib(name),st.attrib(sname),invert,rname));

        select (gAr1.dtype, gAr2.dtype) {
            when (DType.Int64, DType.Int64) {
                var ar1 = toSymEntry(gAr1,int);
                var ar2 = toSymEntry(gAr2,int);

                // things to do...
                // if ar2 is big for some value of big... call unique on ar2 first

                // brute force if below small bound
                if (ar2.size <= sBound) {
                    iLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "%t <= %t, using GlobalAr2Bcast".format(ar2.size,sBound));                    
                    var truth = in1dGlobalAr2Bcast(ar1.a, ar2.a);
                    if (invert) {truth = !truth;}
                    
                    st.addEntry(rname, new shared SymEntry(truth));
                }
                // per locale assoc domain if below medium bound
                else if (ar2.size <= mBound) {
                    iLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                               "%t <= %t, using Ar2PerLocAssoc".format(ar2.size,mBound));                  
                    var truth = in1dAr2PerLocAssoc(ar1.a, ar2.a);
                    if (invert) {truth = !truth;}
                    
                    st.addEntry(rname, new shared SymEntry(truth));
                }
                // sort-based strategy if above medium bound
                else {
                    iLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                          "%t > %t, using sort-based strategy".format(ar2.size, mBound));
                    var truth = in1dSort(ar1.a, ar2.a);
                    if (invert) {truth = !truth;}

                    st.addEntry(rname, new shared SymEntry(truth));
                }
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
}
