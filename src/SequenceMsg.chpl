module SequenceMsg {
    use ServerConfig;

    use Time;
    use Reflection;
    use Logging;
    use Message;
    use BigInteger;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    
    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const smLogger = new Logger(logLevel, logChannel);
    /*
    Creates a sym entry with distributed array adhering to the Msg parameters (start, stop, step)

    :arg reqMsg: request containing (cmd,start,stop,step)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
    */
    @arkouda.instantiateAndRegister
    proc arange(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws 
        where ((array_dtype==int) || (array_dtype==uint(64)) || (array_dtype==bigint)) && (array_nd==1) {
        
        const start =  msgArgs["start"].toScalar(array_dtype),
            stop =  msgArgs["stop"].toScalar(array_dtype),
            step =  msgArgs["step"].toScalar(array_dtype),
            len = ((stop - start + step - 1) / step):int;

        var ea = makeDistArray(len, array_dtype);
        const ref ead = ea.domain;
        forall (ei, i) in zip(ea, ead) {
            ei = start + (i * step);
        }
        return st.insert(new shared SymEntry(ea));
    }

    /* 
    Creates a sym entry with distributed array adhering to the Msg parameters (start, stop, len)

    :arg reqMsg: request containing (cmd,start,stop,len)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
    */
    proc linspaceMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        var start = msgArgs.get("start").getRealValue();
        var stop = msgArgs.get("stop").getRealValue();
        var len = msgArgs.get("len").getIntValue();
        // compute stride
        var stride = (stop - start) / (len-1);
        overMemLimit(8*len);
        // get next symbol name
        var rname = st.nextName();
        smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "cmd: %s start: %r stop: %r len: %i stride: %r rname: %s".format(
                         cmd, start, stop, len, stride, rname));

        var t1 = Time.timeSinceEpoch().totalSeconds();
        var e = st.addEntry(rname, len, real);
        smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                      "alloc time = %i".format(Time.timeSinceEpoch().totalSeconds() - t1));

        t1 = Time.timeSinceEpoch().totalSeconds();
        ref ea = e.a;
        const ref ead = e.a.domain;
        forall (ei, i) in zip(ea,ead) {
            ei = start + (i * stride);
        }
        ea[0] = start;
        ea[len-1] = stop;
        smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                   "compute time = %i".format(Time.timeSinceEpoch().totalSeconds() - t1));

        repMsg = "created " + st.attrib(rname);
        smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);       
        return new MsgTuple(repMsg,MsgType.NORMAL);
    }

    use CommandMap;
    registerFunction("linspace", linspaceMsg, getModuleName());
}
