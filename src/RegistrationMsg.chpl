module RegistrationMsg
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Reflection only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    /* 
    Parse, execute, and respond to a register message 

    :arg reqMsg: request containing (name,user_defined_name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) response message
    */
    proc registerMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        var repMsg: string; // response message
        // split request into fields
        var (name, userDefinedName) = payload.decode().splitMsgToTuple(2);

        // if verbose print action
        if v {try! writeln("%s %s %s".format(cmd,name,userDefinedName)); try! stdout.flush();}

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
    proc attachMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        var repMsg: string; // response message
        // split request into fields
        var (name) = payload.decode().splitMsgToTuple(1);

        // if verbose print action
        if v {try! writeln("%s %s".format(cmd,name)); try! stdout.flush();}

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
    proc unregisterMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        var repMsg: string; // response message
        // split request into fields
        var (name) = payload.decode().splitMsgToTuple(1);

        // if verbose print action
        if v {try! writeln("%s %s".format(cmd,name)); try! stdout.flush();}

        // take name out of the registry and delete entry in symbol table
        st.unregName(name);
        
        return "success";
    }

}
