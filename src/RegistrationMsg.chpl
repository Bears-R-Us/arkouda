module RegistrationMsg
{
    use ServerConfig;

    use ArkoudaTimeCompat as Time;
    use Math only;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use List;
    use Set;
    use Sort;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use SegmentedString;
    use SegmentedMsg;

    use Map;

    use ArkoudaIOCompat;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const regLogger = new Logger(logLevel, logChannel);

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
               "cmd: %s name: %s userDefinedName: %s".doFormat(cmd,name,userDefinedName));

        // register new user_defined_name for name
        var msgTuple:MsgTuple;
        try {
            st.regName(name, userDefinedName);
            repMsg = "success";
            regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            msgTuple = new MsgTuple(repMsg, MsgType.NORMAL);
        } catch e: ArgumentError {
            repMsg = "Error: requested name '%s' was already in use.".doFormat(userDefinedName);
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

        var objType: ObjType = ObjType.UNKNOWN;
        if msgArgs.contains("objtype") {
            objType = msgArgs.getValueOf("objtype").toUpper(): ObjType;
        }

        // if verbose print action
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "%s %s".doFormat(cmd,name));

        // lookup name in symbol table to get attributes
        var attrib = st.attrib(name);
        
        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "requested attrib: %s".doFormat(attrib));

        // response message
        if (attrib.startsWith("Error:")) { 
            var errorMsg = attrib;
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        } else {
            if objType == ObjType.UNKNOWN || objType == ObjType.STRINGS || objType == ObjType.PDARRAY {
                repMsg = "created %s".doFormat(attrib);
                if (isStringAttrib(attrib)) {
                    var s = getSegString(name, st);
                    repMsg += "+created bytes.size %?".doFormat(s.nBytes);
                }
            }
            else {
                var errorMsg = "Error: Unkown object type passed to attachMsg - %s".doFormat(objType);
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
                            "%s: Collecting Categorical components for '%s'".doFormat(cmd, name));

        var rtnMap: map(string, string);
                
        var cats = st.attrib("%s.categories".doFormat(name));
        var codes = st.attrib("%s.codes".doFormat(name));
        var naCode = st.attrib("%s._akNAcode".doFormat(name));

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

        // categories should always be string, add bytes for string return message
        if (isStringAttrib(cats)) {
            var s = getSegString("%s.categories".doFormat(name), st);
            rtnMap.add("categories", "created %s+created %?".doFormat(st.attrib(s.name), s.nBytes));
        }
        rtnMap.add("codes", "created " + codes);
        rtnMap.add("_akNAcode", "created " + naCode);


        // Optional components of categorical
        if st.contains("%s.permutation".doFormat(name)) {
            var perm = st.attrib("%s.permutation".doFormat(name));
            if (perm.startsWith("Error:")) { 
                var errorMsg = perm;
                regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR); 
            }
            rtnMap.add("permutation", "created " + perm);
        }
        if st.contains("%s.segments".doFormat(name)) {
            var segs = st.attrib("%s.segments".doFormat(name));
            if (segs.startsWith("Error:")) { 
                var errorMsg = segs;
                regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR); 
            }
            rtnMap.add("segments", "created " + segs);
        }

        var repMsg: string = "categorical+%s+".doFormat(name)+formatJson(rtnMap);
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
                            "%s: Collecting Series components for '%s'".doFormat(cmd, name));

        var repMsg: string;

        var ind = "";

        // if Series matches MultiIndex format
        if st.contains("%s_key_0".doFormat(name)) {
            var nameList = st.findAll("%s_key_\\d".doFormat(name));
            sort(nameList);
            for regName in nameList {
                var entry = st.attrib(regName);
                if (regName.startsWith("Error:")) { 
                    var errorMsg = regName;
                    regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR); 
                }
                ind += "+created %s".doFormat(entry);
            }
        }
        else {  // Series only contains one key for index
            ind = st.attrib("%s_key".doFormat(name));
            if (ind.startsWith("Error:")) { 
                var errorMsg = ind;
                regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR); 
            }
            ind = "+created %s".doFormat(ind);
        }

        var vals = st.attrib("%s_value".doFormat(name));
        if (vals.startsWith("Error:")) { 
            var errorMsg = vals;
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        }

        repMsg = "series+created %s%s".doFormat(vals, ind);

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
        var colName = "df_columns_%s".doFormat(name);
        var repMsg = "dataframe+%s".doFormat(name);

        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                            "%s: Collecting DataFrame components for '%s'".doFormat(cmd, name));

        var jsonParam = new ParameterObj("name", colName, ObjectType.VALUE, "str");
        var subArgs1 = new MessageArgs(new list([jsonParam, ]));
        // Add columns as a json list
        var cols = stringsToJSONMsg(cmd, subArgs1, st).msg;
        repMsg += "+json %s".doFormat(cols);

        // Get index 
        var indParam = new ParameterObj("name", "df_index_%s_key".doFormat(name), ObjectType.VALUE, "");
        var subArgs2 = new MessageArgs(new list([indParam, ]));
        var ind = attachMsg(cmd, subArgs2, st).msg;
        if ind.startsWith("Error:") { 
            var errorMsg = ind;
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        }
        repMsg += "+%s".doFormat(ind);

        // Get column data
        var nameList = st.findAll("df_data_(pdarray|str|SegArray|Categorical)_.*_%s".doFormat(name));
        
        if nameList.size == 1 && nameList[0] == "" {
            var errorMsg = "No data values found for DataFrame %s".doFormat(name);
            regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        // Convert nameList to a Set to get unique values
        var u : set(string) = new set(string, nameList);

        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                        "%s: Data components found for dataframe: '".doFormat(cmd)+formatJson(u)+"'");

        // Use existing attach functionality to build the response message based on the objType of each data column
        forall regName in u with (+ reduce repMsg) {
            var parts = regName.split("_");
            var objtype: ObjType = parts[2].toUpper(): ObjType;
            var msg: string;
            select (objtype){
                when (ObjType.PDARRAY) {
                    var attParam = new ParameterObj("name", regName, ObjectType.VALUE, "");
                    var subArgs = new MessageArgs(new list([attParam, ]));
                    msg = attachMsg(cmd, subArgs, st).msg;
                }
                when (ObjType.SEGARRAY) {
                    var sa_map: map(string, string);
                    var attParam = new ParameterObj("name", regName+"_segments", ObjectType.VALUE, "");
                    var subArgs = new MessageArgs(new list([attParam, ]));
                    sa_map.add("segments", attachMsg(cmd, subArgs, st).msg);

                    var attParam2 = new ParameterObj("name", regName+"_values", ObjectType.VALUE, "");
                    var subArgs2 = new MessageArgs(new list([attParam2, ]));
                    sa_map.add("values", attachMsg(cmd, subArgs2, st).msg);

                    // attach to lengths
                    var attParam3 = new ParameterObj("name", regName+"_lengths", ObjectType.VALUE, "");
                    var subArgs3 = new MessageArgs(new list([attParam3, ]));
                    sa_map.add("lengths", attachMsg(cmd, subArgs3, st).msg);
                    msg = "segarray+"+formatJson(sa_map);
                }
                when (ObjType.STRINGS) {
                    var attParam = new ParameterObj("name", regName, ObjectType.VALUE, "");
                    var subArgs = new MessageArgs(new list([attParam, ]));
                    msg = attachMsg(cmd, subArgs, st).msg;
                }
                when (ObjType.CATEGORICAL) {
                    msg = attachCategoricalMsg(cmd, regName, st).msg;
                }
                otherwise {
                    regLogger.warn(getModuleName(),getRoutineName(),getLineNumber(), 
                                "Unsupported column type found in DataFrame: '%s'. \
                                Supported types are: pdarray, str, Categorical, and SegArray".doFormat(objtype));
                    
                    throw getErrorWithContext(
                                        msg="Unknown column type (%s) found in DataFrame: %s".doFormat(objtype, name),
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
                repMsg += "+%s".doFormat(msg);
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
                        "Attempting to find type of registered element '%s'".doFormat(name));

        var objtype: string;

        if st.contains(name) {
            // SegArray is now also represented in the SymbolTable as a single entry with no extras attached to the name
            var entry = st.lookup(name);
            // pdarray or Strings
            objtype = "simple";
        } else if st.contains("%s.categories".doFormat(name)) && st.contains("%s.codes".doFormat(name)) {
            objtype = "categorical";
        } else if st.contains("%s_value".doFormat(name)) && (st.contains("%s_key".doFormat(name)) || st.contains("%s_key_0".doFormat(name))) {
            objtype = "series";
        } else if st.contains("df_columns_%s".doFormat(name)) && (st.contains("df_index_%s_key".doFormat(name))) {
            objtype = "dataframe";
        } else if st.contains("%s_segments".doFormat(name)) && st.contains("%s_values".doFormat(name)) {
            objtype = "segarray";
        } 
        else {
            throw getErrorWithContext(
                                msg="Unable to determine type for given name: %s".doFormat(name),
                                lineNumber=getLineNumber(),
                                routineName=getRoutineName(),
                                moduleName=getModuleName(),
                                errorClass="ValueError"
                                );
        }

        regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                        "Type determined to be: '%s'".doFormat(objtype));

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
                // pdarray, strings can use the attachMsg method
                var aRet = attachMsg(cmd, msgArgs, st);
                var msg = aRet.msg;
                var msgType = aRet.msgType;
                repMsg = "simple+%s".doFormat(msg);
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
            when ("segarray"){
                // attach to segments
                var sa_map: map(string, string);
                var attParam = new ParameterObj("name", name+"_segments", ObjectType.VALUE, "");
                var subArgs = new MessageArgs(new list([attParam, ]));
                sa_map.add("segments", attachMsg(cmd, subArgs, st).msg);

                // attach to values
                var attParam2 = new ParameterObj("name", name+"_values", ObjectType.VALUE, "");
                var subArgs2 = new MessageArgs(new list([attParam2, ]));
                sa_map.add("values", attachMsg(cmd, subArgs2, st).msg);

                // attach to lengths
                var attParam3 = new ParameterObj("name", name+"_lengths", ObjectType.VALUE, "");
                var subArgs3 = new MessageArgs(new list([attParam3, ]));
                sa_map.add("lengths", attachMsg(cmd, subArgs3, st).msg);

                return new MsgTuple("segarray+"+formatJson(sa_map), MsgType.NORMAL); 
            }
            otherwise {
                regLogger.warn(getModuleName(),getRoutineName(),getLineNumber(), 
                            "Unsupported type provided: '%s'. Supported types are: pdarray, strings, categorical, segarray, series, and dataframe".doFormat(dtype));
                
                throw getErrorWithContext(
                                    msg="Unknown type (%s) supplied for given name: %s".doFormat(dtype, name),
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
                                                          "%s %s".doFormat(cmd,name));

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

        if simpleTypes.contains(dtype.toLower()) {
            dtype = "simple";
        }

        select (dtype.toLower()) {
            when ("simple") {
                // pdarray and strings can use the unregisterMsg method without any other processing
                var subArgs = new MessageArgs(new list([msgArgs.get("name"), ]));
                return unregisterMsg(cmd, subArgs, st);
            }
            when ("categorical") {
                // Create an array with 5 strings, one for each component of categorical, and assign the names
                var nameList: [0..4] string;
                nameList[0] = "%s.categories".doFormat(name);
                nameList[1] = "%s.codes".doFormat(name);
                nameList[2] = "%s._akNAcode".doFormat(name);
                
                if st.contains("%s.permutation".doFormat(name)) {
                    nameList[3] = "%s.permutation".doFormat(name);
                }
                if st.contains("%s.segments".doFormat(name)) {
                    nameList[4] = "%s.segments".doFormat(name);
                }

                var base_json = msgArgs.get("name");

                for n in nameList {
                    // Check for "" in case optional components aren't found
                    if n != "" {
                        base_json.setVal(n);
                        var subArgs = new MessageArgs(new list([base_json, ]));
                        var resp = unregisterMsg(cmd, subArgs, st);
                        status += " %s: %s ".doFormat(n, resp.msg);
                    }
                }
            }
            when ("series") {
                // Identify if the series contains MultiIndex or Single Index components
                var nameStr = "";

                // MultiIndex
                if st.contains("%s_key_0".doFormat(name)) {
                    // Get an array of all the multi-index parts
                    var indexList = st.findAll("%s_key_\\d".doFormat(name));
                    // Convert the array into a + delimited string
                    nameStr = "+".join(indexList);
                } 
                else {  // Single index
                    // Add the name of the single key to the name String
                    nameStr = "%s_key".doFormat(name);
                }
                // Add the name of the values to the name String
                nameStr += "+%s_value".doFormat(name);

                // Convert the string back into an array for looping
                var nameList = nameStr.split("+");
                var base_json = msgArgs.get("name");
                forall n in nameList with (in base_json, + reduce status) {
                    base_json.setVal(n);
                    var subArgs = new MessageArgs(new list([base_json, ]));
                    var resp = unregisterMsg(cmd, subArgs, st);
                    status += " %s: %s ".doFormat(n, resp.msg);
                }
            }
            otherwise {
                regLogger.warn(getModuleName(),getRoutineName(),getLineNumber(), 
                            "Unsupported type provided: '%s'. Supported types are: pdarray, strings, categorical, segarray and series".doFormat(dtype));
                
                throw getErrorWithContext(
                                    msg="Unknown type (%s) supplied for given name: %s".doFormat(dtype, name),
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
