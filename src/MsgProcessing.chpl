
module MsgProcessing
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use AryUtil;
    
    private config const logLevel = ServerConfig.logLevel;
    const mpLogger = new Logger(logLevel);
    
    /* 
    Parse, execute, and respond to a create message 

    :arg : payload
    :type string: containing (dtype,size)

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (MsgTuple) response message
    */
    proc createMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        var dtype = str2dtype(msgArgs.getValueOf("dtype"));
        var size = msgArgs.get("size").getIntValue();
        if (dtype == DType.UInt8) || (dtype == DType.Bool) {
          overMemLimit(size);
        } else {
          overMemLimit(8*size);
        }
        // get next symbol name
        var rname = st.nextName();
        
        // if verbose print action
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
            "cmd: %s dtype: %s size: %i new pdarray name: %s".format(
                                                     cmd,dtype2str(dtype),size,rname));
        // create and add entry to symbol table
        st.addEntry(rname, size, dtype);
        // if verbose print result
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                    "created the pdarray %s".format(st.attrib(rname)));

        repMsg = "created " + st.attrib(rname);
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);                                 
        return new MsgTuple(repMsg, MsgType.NORMAL);
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
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s".format(cmd));
        if (memTrack) {
            return new MsgTuple((getMemUsed():uint * numLocales:uint):string, MsgType.NORMAL);
        }
        else {
            return new MsgTuple(st.memUsed():string, MsgType.NORMAL);
        }
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
    proc setMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("array");
        var dtype = str2dtype(msgArgs.getValueOf("dtype"));
        const value = msgArgs.get("val");

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "cmd: %s value: %s in pdarray %s".format(cmd,name,st.attrib(name)));

        select (gEnt.dtype, dtype) {
            when (DType.Int64, DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var val: int = value.getIntValue();
                e.a = val;
                repMsg = "set %s to %t".format(name, val);
            }
            when (DType.Int64, DType.Float64) {
                var e = toSymEntry(gEnt,int);
                var val: real = value.getRealValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "cmd: %s name: %s to val: %t".format(cmd,name,val:int));
                e.a = val:int;
                repMsg = "set %s to %t".format(name, val:int);
            }
            when (DType.Int64, DType.Bool) {
                var e = toSymEntry(gEnt,int);
                var val: bool = value.getBoolValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "cmd: %s name: %s to val: %t".format(cmd,name,val:int));
                e.a = val:int;
                repMsg = "set %s to %t".format(name, val:int);
            }
            when (DType.Float64, DType.Int64) {
                var e = toSymEntry(gEnt,real);
                var val: int = value.getIntValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                      "cmd: %s name: %s to value: %t".format(cmd,name,val:real));
                e.a = val:real;
                repMsg = "set %s to %t".format(name, val:real);
            }
            when (DType.Float64, DType.Float64) {
                var e = toSymEntry(gEnt,real);
                var val: real = value.getRealValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "cmd: %s name; %s to value: %t".format(cmd,name,val));
                e.a = val;
                repMsg = "set %s to %t".format(name, val);
            }
            when (DType.Float64, DType.Bool) {
                var e = toSymEntry(gEnt,real);           
                var val: bool = value.getBoolValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "cmd: %s name: %s to value: %t".format(cmd,name,val:real));
                e.a = val:real;
                repMsg = "set %s to %t".format(name, val:real);
            }
            when (DType.Bool, DType.Int64) {
                var e = toSymEntry(gEnt,bool);
                var val: int = value.getIntValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "cmd: %s name: %s to value: %t".format(cmd,name,val:bool));
                e.a = val:bool;
                repMsg = "set %s to %t".format(name, val:bool);
            }
            when (DType.Bool, DType.Float64) {
                var e = toSymEntry(gEnt,int);
                var val: real = value.getRealValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                      "cmd: %s name: %s to  value: %t".format(cmd,name,val:bool));
                e.a = val:bool;
                repMsg = "set %s to %t".format(name, val:bool);
            }
            when (DType.Bool, DType.Bool) {
                var e = toSymEntry(gEnt,bool);
                var val: bool = value.getBoolValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                            "cmd: %s name: %s to value: %t".format(cmd,name,val));
                e.a = val;
                repMsg = "set %s to %t".format(name, val);
            }
            when (DType.UInt64, DType.UInt64) {
                var e = toSymEntry(gEnt,uint);
                var val: uint = value.getUIntValue();
                e.a = val;
                repMsg = "set %s to %t".format(name, val);
            }
            otherwise {
                mpLogger.error(getModuleName(),getRoutineName(),
                                               getLineNumber(),"dtype: %s".format(msgArgs.getValueOf("dtype")));
                return new MsgTuple(unrecognizedTypeError(pn,msgArgs.getValueOf("dtype")), MsgType.ERROR);
            }
        }

        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
}
