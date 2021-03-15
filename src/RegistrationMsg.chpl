module RegistrationMsg
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Reflection;
    use Errors;
    use Logging;
    use Message;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    const regLogger = new Logger();
    if v {
        regLogger.level = LogLevel.DEBUG;
    } else {
        regLogger.level = LogLevel.INFO;    
    }

    /* 
    Parse, execute, and respond to a register message 

    :arg reqMsg: request containing (name,user_defined_name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple response message
    */
    proc registerMsg(cmd: string, payload: string,  
                                        st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        // split request into fields
        var (name, userDefinedName) = payload.splitMsgToTuple(2);

        // if verbose print action
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
               "cmd: %s name: %s userDefinedName: %s".format(cmd,name,userDefinedName));

        // register new user_defined_name for name
        st.regName(name, userDefinedName);
        
        // response message
        repMsg = "created %s".format(st.attrib(userDefinedName));
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /* 
    Parse, execute, and respond to a attach message 

    :arg reqMsg: request containing (name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple response message
    */
    proc attachMsg(cmd: string, payload: string, 
                                          st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        // split request into fields
        var (name) = payload.splitMsgToTuple(1);

        // if verbose print action
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                          "%s %s".format(cmd,name));

        // lookup name in symbol table to get attributes
        var attrib = st.attrib(name);
        
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                          "requested attrib: %s".format(attrib));
        // response message
        if (attrib.startsWith("Error:")) { 
            var errorMsg = attrib;
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        } else {
            repMsg = "created %s".format(attrib);
            regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL); 
        }
    }

    /* 
    Parse, execute, and respond to a unregister message 

    :arg reqMsg: request containing (name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple response message
    */
    proc unregisterMsg(cmd: string, payload: string, 
                                      st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        // split request into fields
        var (name) = payload.splitMsgToTuple(1);

        // if verbose print action
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                          "%s %s".format(cmd,name));

        // take name out of the registry
        st.unregName(name);
        
        repMsg = "success";
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
}
