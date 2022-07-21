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
    use SegmentedString;

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
        var naCode = st.attrib("%s._akNAcode".format(name));

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
        if (naCode.startsWith("Error:")) { 
            var errorMsg = naCode;
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
        repMsg += "+created %s".format(naCode);

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
    Compile the component parts of a SegString attach message 

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
                            "%s: Collecting SegString components for '%s'".format(cmd, name));

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
    Compile the component parts of a Series attach message 

    :arg cmd: calling command 
    :type cmd: string 

    :arg name: name of SymTab element
    :type name: string

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple response message
    */
    proc attachSeriesMsg(cmd: string, name: string, st: borrowed SymTab): MsgTuple throws {
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                            "%s: Collecting Series components for '%s'".format(cmd, name));

        var repMsg: string;

        var ind = "";

        // if Series matches MultiIndex format
        if st.contains("%s_key_0".format(name)) {
            var nameList = st.findAll("%s_key_\\d".format(name));
            nameList = nameList.sorted();  // Sort the list to return the indexes in order from 0 to N
            for regName in nameList {
                var entry = st.attrib(regName);
                if (regName.startsWith("Error:")) { 
                    var errorMsg = regName;
                    regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR); 
                }
                ind += "+created %s".format(entry);
            }
        }
        else {  // Series only contains one key for index
            ind = st.attrib("%s_key".format(name));
            if (ind.startsWith("Error:")) { 
                var errorMsg = ind;
                regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR); 
            }
            ind = "+created %s".format(ind);
        }

        var vals = st.attrib("%s_value".format(name));
        if (vals.startsWith("Error:")) { 
            var errorMsg = vals;
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        }

        repMsg = "series+created %s%s".format(vals, ind);

        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /*
    Attempt to determine the type of object base on a given name

    :arg cmd: calling command 
    :type cmd: string 

    :arg name: entry name to find type of
    :type name: string

    :arg st: SymTab to act on
    :type st: borrowed SymTab 
    */
    proc findType(cmd: string, name: string, st: borrowed SymTab): string throws {
        // Try to determine the type from the entries in the symbol table
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                        "Attempting to find type of registered element '%s'".format(name));

        var dtype: string;

        if st.contains(name) {
            // Easy case where full name given matches an entry, pdarray or Strings
            dtype = "simple";
        } else if st.contains("%s.categories".format(name)) && st.contains("%s.codes".format(name)) {
            dtype = "categorical";
        } else if st.contains("%s_segments".format(name)) && st.contains("%s_values".format(name)) {
            // Important to note that categorical has a .segments while segarray uses _segments
            dtype = "segarray";
        } else if st.contains("%s_value".format(name)) && (st.contains("%s_key".format(name)) || st.contains("%s_key_0".format(name))) {
            dtype = "series";
        } else {
            throw getErrorWithContext(
                                msg="Unable to determine type for given name: %s".format(name),
                                lineNumber=getLineNumber(),
                                routineName=getRoutineName(),
                                moduleName=getModuleName(),
                                errorClass="ValueError"
                                );
        }

        return dtype;
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
            dtype = findType(cmd, name, st);
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
            when ("series") {
                return attachSeriesMsg(cmd, name, st);
            }
            otherwise {
                regLogger.warn(getModuleName(),getRoutineName(),getLineNumber(), 
                            "Unsupported type provided: '%s'. Supported types are: pdarray, strings, categorical, segarray, and series".format(dtype));
                
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

    proc unregisterByName(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        var ele_parts = payload.split("+");
        var dtype = ele_parts[0];
        var name = ele_parts[1];
        var status = "";

        if dtype == "infer" {
            dtype = findType(cmd, name, st);
        }

        // type possibilities for pdarray and strings
        var simpleTypes: list(string) = ["pdarray","int64", "uint8", "uint64", "float64", "bool", "strings", "string", "str"];
        if simpleTypes.contains(dtype.toLower()) {
            dtype = "simple";
        }

        select (dtype.toLower()) {
            when ("simple") {
                // pdarray and strings can use the unregisterMsg method without any other processing
                return unregisterMsg(cmd, name, st);
            }
            when ("categorical") {
                // Create an array with 5 strings, one for each component of categorical, and assign the names
                var nameList: [0..4] string;
                nameList[0] = "%s.categories".format(name);
                nameList[1] = "%s.codes".format(name);
                nameList[2] = "%s._akNAcode".format(name);
                
                if st.contains("%s.permutation".format(name)) {
                    nameList[3] = "%s.permutation".format(name);
                }
                if st.contains("%s.segments".format(name)) {
                    nameList[4] = "%s.segments".format(name);
                }

                for n in nameList {
                    // Check for "" in case optional components aren't found
                    if n != "" {
                        var resp = unregisterMsg(cmd, n, st);
                        status += " %s: %s ".format(n, resp.msg);
                    }
                }
            }
            when ("segarray") {
                // Create an array with 3 strings, one for each component of segarray, and assign the names
                var nameList: [0..2] string;
                nameList[0] = "%s_segments".format(name);
                nameList[1] = "%s_values".format(name);
                nameList[2] = "%s_lengths".format(name);

                for n in nameList {
                    var resp = unregisterMsg(cmd, n, st);
                    status += " %s: %s ".format(n, resp.msg);
                }
            }
            when ("series") {
                // Identify if the series contains MultiIndex or Single Index components
                var nameStr = "";

                // MultiIndex
                if st.contains("%s_key_0".format(name)) {
                    // Get an array of all the multi-index parts
                    var indexList = st.findAll("%s_key_\\d".format(name));
                    // Convert the array into a + delimited string
                    nameStr = "+".join(indexList);
                } 
                else {  // Single index
                    // Add the name of the single key to the name String
                    nameStr = "%s_key".format(name);
                }
                // Add the name of the values to the name String
                nameStr += "+%s_value".format(name);

                // Convert the string back into an array for looping
                var nameList = nameStr.split("+");
                forall n in nameList with (+ reduce status) {
                    var resp = unregisterMsg(cmd, n, st);
                    status += " %s: %s ".format(n, resp.msg);
                }
            }
            otherwise {
                regLogger.warn(getModuleName(),getRoutineName(),getLineNumber(), 
                            "Unsupported type provided: '%s'. Supported types are: pdarray, strings, categorical, segarray, and series".format(dtype));
                
                throw getErrorWithContext(
                                    msg="Unknown type (%s) supplied for given name: %s".format(dtype, name),
                                    lineNumber=getLineNumber(),
                                    routineName=getRoutineName(),
                                    moduleName=getModuleName(),
                                    errorClass="ValueError"
                                    );
            }
        }

        var repMsg = status;
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    use CommandMap;
    registerFunction("register", registerMsg, getModuleName());
    registerFunction("attach", attachMsg, getModuleName());
    registerFunction("genericAttach", genAttachMsg, getModuleName());
    registerFunction("unregister", unregisterMsg, getModuleName());
    registerFunction("genericUnregisterByName", unregisterByName, getModuleName());
}
