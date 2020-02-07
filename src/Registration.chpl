module Registration
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
    proc registerMsg(reqMsg: string, st: borrowed SymTab): string throws {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var user_defined_name = fields[3];

        // if verbose print action
        if v {try! writeln("%s %s %s".format(cmd,name,user_defined_name)); try! stdout.flush();}

        // lookup name in symbol table
        var gEnt: borrowed GenSymEntry = st.lookup(name);

        // add entry to symbol table under user_defined_name
        st.addEntry(user_defined_name, gEnt);
        // put name in registry
        st.regName(user_defined_name);
        
        // response message
        return try! "created " + st.attrib(user_defined_name);
    }

    /* 
    Parse, execute, and respond to a attach message 

    :arg reqMsg: request containing (name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) response message
    */
    proc attachMsg(reqMsg: string, st: borrowed SymTab): string throws {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];

        // if verbose print action
        if v {try! writeln("%s %s".format(cmd,name)); try! stdout.flush();}

        // lookup name in symbol table to get attributes
        var attrib = st.attrib(name);
        // response message
        if (attrib.startsWith("Error:")) { return (s); }
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
    proc unregisterMsg(reqMsg: string, st: borrowed SymTab): string throws {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];

        // if verbose print action
        if v {try! writeln("%s %s".format(cmd,name)); try! stdout.flush();}

        // take name out of the registry in symbol table
        st.unregName(name);
        // delete the entry from the symbol table
        st.deleteEntry(name);
        
        return "success";
    }

}
