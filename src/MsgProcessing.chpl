
module MsgProcessing
{
    use ServerConfig;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use BigInteger;
    use Math;
    use Time;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use AryUtil;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const mpLogger = new Logger(logLevel, logChannel);

    /* 
    Parse, execute, and respond to a create message 

    :arg : payload
    :type string: containing (dtype,size)

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (MsgTuple) response message
    */
    @arkouda.instantiateAndRegister
    proc create(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
        const shape = msgArgs["shape"].toScalarTuple(int, array_nd);

        var size = 1;
        for s in shape do size *= s;
        overMemLimit(typeSize(array_dtype) * size);

        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "creating new array (dtype=%s, shape=%?)".format(type2str(array_dtype),shape));

        return st.insert(createSymEntry((...shape), array_dtype));
    }

    @arkouda.instantiateAndRegister
    proc createScalarArray(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype): MsgTuple throws {
        const value = msgArgs["value"].toScalar(array_dtype);

        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "creating scalar array (dtype=%s)".format(type2str(array_dtype)));

        // on the client side, scalar (0D) arrays have a shape of "()" and a size of 1
        // here, we represent that using a 1D array with a shape of (1,) and a size of 1
        var e = createSymEntry(1, array_dtype);
        e.a[0] = value;

        var ge = e: GenSymEntry;
        ge.size = 1;
        ge.shape = "[]";
        ge.ndim = 1; // this is 1 rather than 0 s.t. calls to other message handlers treat it as a 1D
                    // array (e.g., we should call 'set<_,1>', not 'set<_,0>' on this array)

        return st.insert(e);
    }

    /* 
    Parse, execute, and respond to a delete message 

    :arg reqMsg: request containing (cmd,name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
    */
    proc deleteMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("name");
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                     "cmd: %s array: %s".format(cmd,st.attrib(name)));
        // delete entry from symbol table
        if st.deleteEntry(name) {
            repMsg = "deleted %s".format(name);
        }
        else {
            repMsg = "registered symbol, %s, not deleted".format(name);
        }
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);       
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /* 
    Clear all unregistered symbols and associated data from sym table
    
    :arg reqMsg: request containing (cmd)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
     */
    proc clearMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "cmd: %s".format(cmd));
        st.clear();

        repMsg = "success";
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /* 
    Takes the name of data referenced in a msg and searches for the name in the provided sym table.
    Returns a string of info for the sym entry that is mapped to the provided name.

    :arg reqMsg: request containing (cmd,name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
     */
    proc infoMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("names");
 
        // if name == "__AllSymbols__" passes back info on all symbols       
        repMsg = st.info(name);
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    
    /* 
    query server configuration...
    
    :arg reqMsg: request containing (cmd)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
     */
    proc getconfigMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s".format(cmd));
        return new MsgTuple(getConfig(), MsgType.NORMAL);
    }

    /* 
        query server total memory allocated or symbol table data memory

        :arg reqMsg: request containing (cmd)
        :type reqMsg: string

        :arg st: SymTab to act on
        :type st: borrowed SymTab

        :returns: MsgTuple
     */
    proc getmemusedMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        var factor = msgArgs.get("factor").getIntValue();
        var asPercent = msgArgs.get("as_percent").getBoolValue();
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s".format(cmd));
        var memUsed = if memTrack then getMemUsed():real * numLocales else st.memUsed():real;
        if asPercent {
            repMsg = Math.round((memUsed / (getMemLimit():real * numLocales)) * 100):uint:string;
        }
        else {
            repMsg = Math.round(memUsed / factor):uint:string;
        }
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /*
        query server total memory availble

        :arg reqMsg: request containing (cmd)
        :type reqMsg: string

        :arg st: SymTab to act on
        :type st: borrowed SymTab

        :returns: MsgTuple
     */
    proc getmemavailMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        var factor = msgArgs.get("factor").getIntValue();
        var asPercent = msgArgs.get("as_percent").getBoolValue();
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s".format(cmd));
        var memUsed = if memTrack then getMemUsed():real * numLocales else st.memUsed():real;
        var totMem = getMemLimit():real * numLocales;
        if asPercent {
            repMsg = (100 - Math.round((memUsed / totMem) * 100)):uint:string;
        }
        else {
            repMsg = Math.round((totMem - memUsed) / factor):uint:string;
        }
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    
    /**
     * Generate the mapping of server command to function as JSON
     * encoded string.
     *
     * The args are IGNORED. They are only here to match the CommandMap
     * standard function signature, similar to other procs.
     *
     * :arg cmd: Ignored
     * :type cmd: string 
     *
     * :arg payload: Ignored
     * :type payload: string
     *
     * :arg st: Ignored
     * :type st: borrowed SymTab 
     *
     * :returns: MsgTuple containing JSON formatted string of cmd -> function mapping
     */
    proc getCommandMapMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab) throws {
        // We can ignore the args, we just need it to match the CommandMap call signature
        import CommandMap;
        try {
            const json:string = CommandMap.dumpCommandMap();
            return new MsgTuple(CommandMap.dumpCommandMap():string, MsgType.NORMAL);
        } catch {
            var errorMsg = "Error converting CommandMap to JSON";
            mpLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
    }

    /* 
    Response to __str__ method in python str convert array data to string 

    :arg reqMsg: request containing (cmd,name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string,MsgType)
   */
    proc strMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("array");

        var printThresh = msgArgs.get("printThresh").getIntValue();
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                              "cmd: %s name: %s threshold: %i".format(
                                               cmd,name,printThresh));  
                                               
        repMsg  = st.datastr(name,printThresh);        
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);  
        return new MsgTuple(repMsg,MsgType.NORMAL);
    }

    /* Response to __repr__ method in python.
       Repr convert array data to string 
       
       :arg reqMsg: request containing (cmd,name)
       :type reqMsg: string 

       :arg st: SymTab to act on
       :type st: borrowed SymTab 

       :returns: MsgTuple
      */ 
    proc reprMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("array");
        var printThresh = msgArgs.get("printThresh").getIntValue();

        repMsg = st.datarepr(name,printThresh);
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg,MsgType.NORMAL);
    }

    /* 
    Sets all elements in array to a value (broadcast) 

    :arg reqMsg: request containing (cmd,name,dtype,value)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
    :throws: `UndefinedSymbolError(name)`
    */
    @arkouda.instantiateAndRegister(prefix="set")
    proc setMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
        var e = st[msgArgs["array"]]: SymEntry(array_dtype, array_nd);
        e.a = msgArgs["val"].toScalar(array_dtype);

        return MsgTuple.success();
    }

     /*
        Get a list of lists indicating how an array is "chunked" across locales.

        For example, a 100x40 2D array on 4 locales could return: [[0, 50], [0, 20]]
        indicating that the chunks start at indices 0 and 50 in the first dimension,
        and 0 and 20 in the second dimension.
    */
    @arkouda.registerCommand
    proc chunkInfoAsString(array: [?d] ?t): string throws
    where (t == bool) || (t == int(64)) || (t == uint(64)) || (t == uint(8)) ||(t == real) {

        var blockSizes: [0..<d.rank] [0..<numLocales] int;

        coforall loc in Locales with (ref blockSizes) do on loc {
            const locDom = d.localSubdomain();
            for i in 0..<d.rank do
                blockSizes[i][loc.id] = locDom.dim(i).low;
        }

        var msg = "[";
        var first = true;
        for dim in blockSizes {
            if first then first = false; else msg += ", ";
            msg += "[";
            var firstInner = true;
            for locSize in dim {
                if firstInner then firstInner = false; else msg += ", ";
                msg += locSize:string;
            }
            msg += "]";
        }
        msg += "]";

        return msg;
    }

    @arkouda.registerCommand
    proc chunkInfoAsArray(array: [?d] ?t):[] int throws
    where (t == bool) || (t == int(64)) || (t == uint(64)) || (t == uint(8)) ||(t == real) {
        var outShape = (d.rank, numLocales);
        var blockSizes= makeDistArray((...outShape), int);

        coforall loc in Locales with (ref blockSizes) do on loc {
            const locDom = d.localSubdomain();
            for i in 0..<d.rank do
                blockSizes[i,loc.id] = locDom.dim(i).low;
        }
        return blockSizes;
    }
}
