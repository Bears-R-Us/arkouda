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
    Creates a sym entry with distributed array adhering to the Msg parameters (start, stop, stride)

    :arg reqMsg: request containing (cmd,start,stop,stride)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
    */
    proc arangeMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        proc arangeHelper(start: ?t, stop: t, stride: t, len, rname) throws {
            smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                "cmd: %s start: %? stop: %? stride: %? : len: %? rname: %s".format(
                cmd, start, stop, stride, len, rname));
            
            var t1 = Time.timeSinceEpoch().totalSeconds();
            var ea = makeDistArray(len, t);
            smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "alloc time = %i sec".format(Time.timeSinceEpoch().totalSeconds() - t1));

            t1 = Time.timeSinceEpoch().totalSeconds();
            const ref ead = ea.domain;
            forall (ei, i) in zip(ea, ead) {
                ei = start + (i * stride);
            }
            var e = st.addEntry(rname, createSymEntry(ea));
            smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "compute time = %i sec".format(Time.timeSinceEpoch().totalSeconds() - t1));
        }

        var repMsg: string; // response message
        var dtype = str2dtype(msgArgs.getValueOf("dtype"));
        // get next symbol name
        var rname = st.nextName();
        select dtype {
            when DType.Int64 {
                var start = msgArgs.get("start").getIntValue();
                var stop = msgArgs.get("stop").getIntValue();
                var stride = msgArgs.get("stride").getIntValue();
                // compute length
                var len = (stop - start + stride - 1) / stride;
                overMemLimit(8*len);
                arangeHelper(start, stop, stride, len, rname);
            }
            when DType.UInt64 {
                var start = msgArgs.get("start").getUIntValue();
                var stop = msgArgs.get("stop").getUIntValue();
                var stride = msgArgs.get("stride").getUIntValue();
                // compute length
                var len = ((stop - start + stride - 1) / stride):int;
                overMemLimit(8*len);
                arangeHelper(start, stop, stride, len, rname);
            }
            when DType.BigInt {
                var start = msgArgs.get("start").getBigIntValue();
                var stop = msgArgs.get("stop").getBigIntValue();
                var stride = msgArgs.get("stride").getBigIntValue();
                // compute length
                var len = ((stop - start + stride - 1) / stride):int;
                // TODO update when we have a better way to handle bigint mem estimation
                arangeHelper(start, stop, stride, len, rname);
            }
        }
        repMsg = "created " + st.attrib(rname);
        smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
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
    registerFunction("arange", arangeMsg, getModuleName());
    registerFunction("linspace", linspaceMsg, getModuleName());
}
