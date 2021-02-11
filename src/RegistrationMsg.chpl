module RegistrationMsg
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Reflection;
    use Errors;
    use Logging;

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

    :returns: (string) response message
    */
    proc registerMsg(cmd: string, payload: string, st: borrowed SymTab): string throws {
        var repMsg: string; // response message
        // split request into fields
        var (name, userDefinedName) = payload.splitMsgToTuple(2);

        // if verbose print action
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                  "cmd: %s name: %s userDefinedName: %s".format(cmd,name,userDefinedName));

        // register new user_defined_name for name
        st.regName(name, userDefinedName);
        
        // response message
        return try! "created " + st.attrib(userDefinedName);
    }

    /* 
    Parse, execute, and respond to a attach message 

    :arg reqMsg: request containing (name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) response message
    */
    proc attachMsg(cmd: string, payload: string, st: borrowed SymTab): string throws {
        var repMsg: string; // response message
        // split request into fields
        var (name) = payload.splitMsgToTuple(1);

        // if verbose print action
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                          "%s %s".format(cmd,name));

        // lookup name in symbol table to get attributes
        var attrib = st.attrib(name);
        // response message
        if (attrib.startsWith("Error:")) { return (attrib); }
        else { return ("created " + attrib); }
    }

    /* 
    Parse, execute, and respond to a unregister message 

    :arg reqMsg: request containing (name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) response message
    */
    proc unregisterMsg(cmd: string, payload: string, st: borrowed SymTab): string throws {
        var repMsg: string; // response message
        // split request into fields
        var (name) = payload.splitMsgToTuple(1);

        // if verbose print action
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                          "%s %s".format(cmd,name));

        // take name out of the registry and delete entry in symbol table
        st.unregName(name);
        
        return "success";
    }
}
