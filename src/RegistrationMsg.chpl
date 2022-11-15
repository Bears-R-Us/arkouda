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
    use Map;
    use Set;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use SegmentedString;
    use SegmentedArray;
    use SegmentedMsg;

    private config const logLevel = ServerConfig.logLevel;
    const regLogger = new Logger(logLevel);
    private var simpleTypes: list(string) = ["pdarray","int64", "uint8", "uint64", "float64", "bool", "strings", "string", "str"];

    /* 
    Parse, execute, and respond to a register message 

    :arg reqMsg: request containing (name,user_defined_name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple response message
    */
    proc registerMsg(cmd: string, msgArgs: borrowed MessageArgs,
                                        st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("array");
        const userDefinedName = msgArgs.getValueOf("user_name");

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
    proc attachMsg(cmd: string, msgArgs: borrowed MessageArgs,
                                          st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message

        const name = msgArgs.getValueOf("name");

        var objType: string = "";
        if msgArgs.contains("objtype") {
            objType = msgArgs.getValueOf("objtype");
        }

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
            if objType.toLower() == "segarray" {
                var a = attrib.split(" ");
                var dtype = str2dtype(a[1]: string);
                var rtnmap: map(string, string) = new map(string, string);

                select dtype {
                    when (DType.Int64) {
                        var seg = getSegArray(name, st, int);
                        seg.fillReturnMap(rtnmap, st);
                    }
                    when (DType.UInt64) {
                        var seg = getSegArray(name, st, uint);
                        seg.fillReturnMap(rtnmap, st);
                    }
                    when (DType.Float64) {
                        var seg = getSegArray(name, st, real);
                        seg.fillReturnMap(rtnmap, st);
                    }
                    when (DType.Bool) {
                        var seg = getSegArray(name, st, bool);
                        seg.fillReturnMap(rtnmap, st);
                    }
                }

                repMsg = "%jt".format(rtnmap);
            }
            else if objType == "" || objType == "str" || objType == "pdarray" {
                repMsg = "created %s".format(attrib);
                if (isStringAttrib(attrib)) {
                    var s = getSegString(name, st);
                    repMsg += "+created bytes.size %t".format(s.nBytes);
                }
            }
            else {
                var errorMsg = "Error: Unkown object type passed to attachMsg - %s".format(objType);
                regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR); 
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
    Compile the component parts of a DataFrame attach message 

    :arg cmd: calling command 
    :type cmd: string 

    :arg payload: name of SymTab element
    :type payload: string

    :arg argSize: number of arguments in payload
    :type argSize: int

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple response message
    */
    proc attachDataFrameMsg(cmd: string, msgArgs: borrowed MessageArgs,
                                 st: borrowed SymTab): MsgTuple throws {
        const name = msgArgs.getValueOf("name"); 
        var colName = "df_columns_%s".format(name);
        var repMsg = "dataframe+%s".format(name);

        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                            "%s: Collecting DataFrame components for '%s'".format(cmd, name));

        var jsonParam = new ParameterObj("name", colName, ObjectType.VALUE, "str");
        var subArgs1 = new MessageArgs(new list([jsonParam]));
        // Add columns as a json list
        var cols = stringsToJSONMsg(cmd, subArgs1, st).msg;
        repMsg += "+json %s".format(cols);

        // Get index 
        var indParam = new ParameterObj("name", "df_index_%s_key".format(name), ObjectType.VALUE, "");
        var subArgs2 = new MessageArgs(new list([indParam]));
        var ind = attachMsg(cmd, subArgs2, st).msg;
        if ind.startsWith("Error:") { 
            var errorMsg = ind;
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        }
        repMsg += "+%s".format(ind);

        // Get column data
        var nameList = st.findAll("df_data_(pdarray|str|SegArray|Categorical)_.*_%s".format(name));
        
        if nameList.size == 1 && nameList[0] == "" {
            var errorMsg = "No data values found for DataFrame %s".format(name);
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        // Convert nameList to a Set to get unique values
        var u : set(string) = new set(string, nameList);

        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                            "%s: Data components found for dataframe: '%jt'".format(cmd, u));

        // Use existing attach functionality to build the response message based on the objType of each data column
        forall regName in u with (+ reduce repMsg) {
            var parts = regName.split("_");
            var objtype: string = parts[2];
            var msg: string;
            select (objtype){
                when ("pdarray") {
                    var attParam = new ParameterObj("name", regName, ObjectType.VALUE, "");
                    var subArgs = new MessageArgs(new list([attParam]));
                    msg = attachMsg(cmd, subArgs, st).msg;
                }
                when ("str") {
                    var attParam = new ParameterObj("name", regName, ObjectType.VALUE, "");
                    var subArgs = new MessageArgs(new list([attParam]));
                    msg = attachMsg(cmd, subArgs, st).msg;
                }
                when ("SegArray") {
                    var attParam = new ParameterObj("name", regName, ObjectType.VALUE, "");
                    var objParam =  new ParameterObj("objtype", objtype, ObjectType.VALUE, "");
                    var subArgs = new MessageArgs(new list([attParam, objParam]));
                    msg = "segarray+%s+%s".format(regName, attachMsg(cmd, subArgs, st).msg);
                }
                when ("Categorical") {
                    msg = attachCategoricalMsg(cmd, regName, st).msg;
                }
                otherwise {
                    regLogger.warn(getModuleName(),getRoutineName(),getLineNumber(), 
                                "Unsupported column type found in DataFrame: '%s'. \
                                Supported types are: pdarray, str, Categorical, and SegArray".format(objtype));
                    
                    throw getErrorWithContext(
                                        msg="Unknown column type (%s) found in DataFrame: %s".format(objtype, name),
                                        lineNumber=getLineNumber(),
                                        routineName=getRoutineName(),
                                        moduleName=getModuleName(),
                                        errorClass="ValueError"
                                        );
                }
            }

            if (msg.startsWith("Error:")) {
                regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),msg);
                repMsg = msg;
            } else {
                repMsg += "+%s".format(msg);
            }
        }

        var msgType = if repMsg.startsWith("Error:") then MsgType.ERROR else MsgType.NORMAL;
        return new MsgTuple(repMsg, msgType);
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

        var objtype: string;

        if st.contains(name) {
            // SegArray is now also represented in the SymbolTable as a single entry with no extras attached to the name
            var entry = st.lookup(name);
            if entry.isAssignableTo(SymbolEntryType.SegArraySymEntry) {
                objtype = "segarray";
            } else {
                // pdarray or Strings
                objtype = "simple";
            }
        } else if st.contains("%s.categories".format(name)) && st.contains("%s.codes".format(name)) {
            objtype = "categorical";
        } else if st.contains("%s_value".format(name)) && (st.contains("%s_key".format(name)) || st.contains("%s_key_0".format(name))) {
            objtype = "series";
        } else if st.contains("df_columns_%s".format(name)) && (st.contains("df_index_%s_key".format(name))) {
            objtype = "dataframe";
        } else {
            throw getErrorWithContext(
                                msg="Unable to determine type for given name: %s".format(name),
                                lineNumber=getLineNumber(),
                                routineName=getRoutineName(),
                                moduleName=getModuleName(),
                                errorClass="ValueError"
                                );
        }

        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                        "Type determined to be: '%s'".format(objtype));

        return objtype;
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
    proc genAttachMsg(cmd: string, msgArgs: borrowed MessageArgs,
                                            st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        var dtype = msgArgs.getValueOf("dtype");
        const name = msgArgs.getValueOf("name");

        if dtype == "infer" {
            dtype = findType(cmd, name, st);
        }

        // type possibilities for pdarray and strings
        if simpleTypes.contains(dtype.toLower()) {
            dtype = "simple";
        }

        select (dtype.toLower()) {
            when ("simple") {
                // pdarray, strings, and segarray can use the attachMsg method
                var aRet = attachMsg(cmd, msgArgs, st);
                var msg = aRet.msg;
                var msgType = aRet.msgType;
                repMsg = "simple+%s".format(msg);
                return new MsgTuple(repMsg, msgType);
            }
            when ("segarray") {
                var attParam = new ParameterObj("name", name, ObjectType.VALUE, "");
                var objParam =  new ParameterObj("objtype", dtype, ObjectType.VALUE, "");
                var subArgs = new MessageArgs(new list([attParam, objParam]));
                var aRet = attachMsg(cmd, subArgs, st);
                var msg = aRet.msg;
                var msgType = aRet.msgType;
                repMsg = "segarray+%s".format(msg);
                return new MsgTuple(repMsg, msgType);
            }
            when ("categorical") {
                return attachCategoricalMsg(cmd, name, st);
            }
            when ("series") {
                return attachSeriesMsg(cmd, name, st);
            }
            when ("dataframe") {
                return attachDataFrameMsg(cmd, msgArgs, st);
            }
            otherwise {
                regLogger.warn(getModuleName(),getRoutineName(),getLineNumber(), 
                            "Unsupported type provided: '%s'. Supported types are: pdarray, strings, categorical, segarray, series, and dataframe".format(dtype));
                
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
    proc unregisterMsg(cmd: string, msgArgs: borrowed MessageArgs,
                                      st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("name");

        // if verbose print action
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                          "%s %s".format(cmd,name));

        // take name out of the registry
        st.unregName(name);
        
        repMsg = "success";
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc unregisterByNameMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var dtype = msgArgs.getValueOf("dtype");
        const name = msgArgs.getValueOf("name");
        var status = "";

        if dtype == "infer" {
            dtype = findType(cmd, name, st);
        }

        if simpleTypes.contains(dtype.toLower()) || dtype == "segarray" {
            dtype = "simple";
        }

        select (dtype.toLower()) {
            when ("simple") {
                // pdarray and strings can use the unregisterMsg method without any other processing
                var subArgs = new MessageArgs(new list([msgArgs.get("name")]));
                return unregisterMsg(cmd, subArgs, st);
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

                var base_json = msgArgs.get("name");

                for n in nameList {
                    // Check for "" in case optional components aren't found
                    if n != "" {
                        base_json.setVal(n);
                        var subArgs = new MessageArgs(new list([base_json]));
                        var resp = unregisterMsg(cmd, subArgs, st);
                        status += " %s: %s ".format(n, resp.msg);
                    }
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
                var base_json = msgArgs.get("name");
                forall n in nameList with (in base_json, + reduce status) {
                    base_json.setVal(n);
                    var subArgs = new MessageArgs(new list([base_json]));
                    var resp = unregisterMsg(cmd, subArgs, st);
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
    registerFunction("genericUnregisterByName", unregisterByNameMsg, getModuleName());
}
