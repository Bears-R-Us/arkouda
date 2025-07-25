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

}
