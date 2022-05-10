module RegistrationMsg
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use List;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use SegmentedArray;

    private config const logLevel = ServerConfig.logLevel;
    const regLogger = new Logger(logLevel);

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
        var msgTuple:MsgTuple;
        try {
            st.regName(name, userDefinedName);
            repMsg = "success";
            regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            msgTuple = new MsgTuple(repMsg, MsgType.NORMAL);
        } catch e: ArgumentError {
            repMsg = "Error: requested name '%s' was already in use.".format(userDefinedName);
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            msgTuple = new MsgTuple(repMsg, MsgType.ERROR);
        }

        return msgTuple;
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
            if (isStringAttrib(attrib)) {
                var s = getSegString(name, st);
                repMsg += "+created bytes.size %t".format(s.nBytes);
            }
            regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL); 
        }
    }

    /* 
    Compile the component parts of a Categorical attach message 

    :arg cmd: calling command 
    :type cmd: string 

    :arg name: name of SymTab element
    :type name: string

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple response message
    */
    proc attachCategoricalMsg(cmd: string, name: string, 
                                            st: borrowed SymTab): MsgTuple throws {
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                            "%s: Collecting Categorical components for '%s'".format(cmd, name));

        var repMsg: string;
                
        var cats = st.attrib("%s.categories".format(name));
        var codes = st.attrib("%s.codes".format(name));

        if (cats.startsWith("Error:")) { 
            var errorMsg = cats;
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        }
        if (codes.startsWith("Error:")) { 
            var errorMsg = codes;
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        }

        repMsg = "categorical+created %s".format(cats);
        // Check if the categories is numeric or string, if string add byte size
        if (isStringAttrib(cats)) {
            var s = getSegString("%s.categories".format(name), st);
            repMsg += "+created bytes.size %t".format(s.nBytes);
        }

        repMsg += "+created %s".format(codes);

        // Optional components of categorical
        if st.contains("%s.permutation".format(name)) {
            var perm = st.attrib("%s.permutation".format(name));
            if (perm.startsWith("Error:")) { 
                var errorMsg = perm;
                regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR); 
            }
            repMsg += "+created %s".format(perm);
        }
        if st.contains("%s.segments".format(name)) {
            var segs = st.attrib("%s.segments".format(name));
            if (segs.startsWith("Error:")) { 
                var errorMsg = segs;
                regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR); 
            }
            repMsg += "+created %s".format(segs);
        }

        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /* 
    Compile the component parts of a SegArray attach message 

    :arg cmd: calling command 
    :type cmd: string 

    :arg name: name of SymTab element
    :type name: string

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple response message
    */
    proc attachSegArrayMsg(cmd: string, name: string, st: borrowed SymTab): MsgTuple throws {
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                            "%s: Collecting SegArray components for '%s'".format(cmd, name));

        var repMsg: string;

        var segs = st.attrib("%s_segments".format(name));
        if (segs.startsWith("Error:")) { 
            var errorMsg = segs;
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        }

        var vals = st.attrib("%s_values".format(name));
        if (vals.startsWith("Error:")) { 
            var errorMsg = vals;
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        }

        var lens = st.attrib("%s_lengths".format(name));
        if (lens.startsWith("Error:")) { 
            var errorMsg = lens;
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        }

        repMsg = "segarray+created %s+created %s+created %s".format(segs, vals, lens);

        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /* 
    Parse, execute, and respond to a generic attach message

    :arg cmd: calling command 
    :type cmd: string 

    :arg payload: request containing (dtype+name)
    :type payload: string

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple response message
    */
    proc genAttachMsg(cmd: string, payload: string, 
                                            st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message

        var ele_parts = payload.split("+");
        var dtype = ele_parts[0];
        var name = ele_parts[1];

        if dtype == "infer" {
            // Try to determine the type from the entries in the symbol table
            regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                            "Attempting to find type of registered element '%s'".format(name));

            if st.contains(name) {
                // Easy case where full name given matches an entry, pdarray or Strings
                dtype = "simple";
            } else if st.contains("%s.categories".format(name)) && st.contains("%s.codes".format(name)) {
                dtype = "categorical";
            } else if st.contains("%s_segments".format(name)) && st.contains("%s_values".format(name)) {
                // Important to note that categorical has a .segments while segarray uses _segments
                dtype = "segarray";
            } else {
                throw getErrorWithContext(
                                    msg="Unable to determine type for given name: %s".format(name),
                                    lineNumber=getLineNumber(),
                                    routineName=getRoutineName(),
                                    moduleName=getModuleName(),
                                    errorClass="ValueError"
                                    );
            }
        }

        // type possibilities for pdarray and strings
        var simpleTypes: list(string) = ["pdarray","int64", "uint8", "uint64", "float64", "bool", "strings", "string", "str"];
        if simpleTypes.contains(dtype.toLower()) {
            dtype = "simple";
        }

        select (dtype.toLower()) {
            when ("simple") {
                // pdarray and strings can use the attachMsg method
                return attachMsg(cmd, name, st);
            }
            when ("categorical") {
                return attachCategoricalMsg(cmd, name, st);
            }
            when ("segarray") {
                return attachSegArrayMsg(cmd, name, st);
            }
            otherwise {
                regLogger.warn(getModuleName(),getRoutineName(),getLineNumber(), 
                            "Unsupported type provided: '%s'. Supported types are: pdarray, strings, categorical, segarray".format(dtype));
                
                throw getErrorWithContext(
                                    msg="Unknown type (%s) supplied for given name: %s".format(dtype, name),
                                    lineNumber=getLineNumber(),
                                    routineName=getRoutineName(),
                                    moduleName=getModuleName(),
                                    errorClass="ValueError"
                                    );
            }
        }
    }

    /*
     * Determine if the attributes belong to a SegString
     * :arg attrs: attributes from SymTab
     * :type attrs: string
     * :returns: bool
     */
    proc isStringAttrib(attrs:string):bool throws {
        var parts = attrs.split();
        return parts.size >=6 && "str" == parts[1];
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

    proc registerMe() {
      use CommandMap;
      registerFunction("register", registerMsg, getModuleName());
      registerFunction("attach", attachMsg, getModuleName());
      registerFunction("genericAttach", genAttachMsg, getModuleName());
      registerFunction("unregister", unregisterMsg, getModuleName());
    }
}
