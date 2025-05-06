module RegistrationMsg {
    use ServerConfig;

    use Time;
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
    use Registry;

    use Map;
    use MultiTypeRegEntry;
    use GenSymIO;
    use IOUtils;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const regLogger = new Logger(logLevel, logChannel);

    private proc register_array(msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var reg_name = msgArgs.getValueOf("name");
        var objType = msgArgs.getValueOf("objType").toUpper(): ObjType;
        var array_name = msgArgs.getValueOf("array");

        var are = new shared ArrayRegEntry(array_name, objType);
        st.registry.register_array(reg_name, are);
        return new MsgTuple("Registered %s".format(objType: string), MsgType.NORMAL);
    }

    private proc register_segarray(msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var reg_name = msgArgs.getValueOf("name");
        var segments = msgArgs.getValueOf("segments");
        var values = msgArgs.getValueOf("values");
        var val_type = msgArgs.getValueOf("val_type").toUpper(): ObjType;
        var lengths = if msgArgs.contains("lengths") then msgArgs.getValueOf("lengths") else "";

        var vre = new shared ArrayRegEntry(values, val_type);
        var sre = new shared SegArrayRegEntry(segments, vre, lengths);
        st.registry.register_segarray(reg_name, sre);
        return new MsgTuple("Registered SegArray", MsgType.NORMAL);
    }

    private proc register_dataframe(msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var reg_name = msgArgs.getValueOf("name");
        var idx_name = msgArgs.getValueOf("idx");
        var num_cols = msgArgs.get("num_cols").getIntValue();
        var column_names: list(string) = new list(msgArgs.get("column_names").getList(num_cols));
        var columns: list(string) = new list(msgArgs.get("columns").getList(num_cols));
        var col_objTypes: list(string) = new list(msgArgs.get("col_objTypes").getList(num_cols));

        var col_list: list(shared AbstractRegEntry);
        for (c, ot) in zip(columns, col_objTypes) {
            var objType: ObjType = ot.toUpper(): ObjType;
            select objType {
                when ObjType.PDARRAY, ObjType.STRINGS, ObjType.DATETIME, ObjType.TIMEDELTA, ObjType.TIMEDELTA, ObjType.IPV4 {
                    var are = new shared ArrayRegEntry(c, objType);
                    col_list.pushBack(are);
                }
                when ObjType.CATEGORICAL {
                    var comps = jsonToMap(c);
                    var perm = if comps.contains["permutation"] then comps["permutation"] else "";
                    var seg = if comps.contains["segments"] then comps["segments"] else "";
                    var cre = new shared CategoricalRegEntry(comps["codes"], comps["categories"], comps["NA_codes"], perm, seg);
                    col_list.pushBack(cre);
                }
                when ObjType.SEGARRAY {
                    var comps = jsonToMap(c);

                    var gse = toGenSymEntry(st[comps["values"]]);
                    var val_type: ObjType = if gse.dtype == DType.Strings then ObjType.STRINGS else ObjType.PDARRAY;
                    var vre = new shared ArrayRegEntry(comps["values"], val_type);

                    var lengths = if comps.contains("lengths") then comps["lengths"] else "";
                    var sre = new shared SegArrayRegEntry(comps["segments"], vre, lengths);
                    col_list.pushBack(sre);
                }
                when ObjType.BITVECTOR {
                    var comps = jsonToMap(c);

                    var bre = new shared BitVectorRegEntry(comps["name"], comps["width"]: int, comps["reverse"]: bool);
                    col_list.pushBack(bre);
                }
                otherwise {
                    var errorMsg = "DataFrames only support columns of type pdarray, Strings, Datetime, Timedelta, IPv4, Categorical, BitVector and SegArray. Found %s".format(objType: string);
                    throw getErrorWithContext(
                        msg=errorMsg,
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="IllegalArgumentError");
                }
            }
        }
        
        var dfre = new shared DataFrameRegEntry(idx_name, column_names, col_list);
        st.registry.register_dataframe(reg_name, dfre);
        return new MsgTuple("Registered DataFrame", MsgType.NORMAL);
    }

    private proc register_groupby(msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var reg_name = msgArgs.getValueOf("name");
        var seg_name = msgArgs.getValueOf("segments");
        var perm_name = msgArgs.getValueOf("permutation");
        var uki = msgArgs.getValueOf("uki");
        var num_keys = msgArgs.get("num_keys").getIntValue();
        var keys: list(string) = new list(msgArgs.get("keys").getList(num_keys));
        var key_objTypes: list(string) = new list(msgArgs.get("key_objTypes").getList(num_keys));

        var key_list: list(shared AbstractRegEntry);
        for (k, ot) in zip(keys, key_objTypes) {
            var objType: ObjType = ot.toUpper(): ObjType;
            if objType == ObjType.PDARRAY || objType == ObjType.STRINGS {
                var are = new shared ArrayRegEntry(k, objType);
                key_list.pushBack(are);
            }
            else if objType == ObjType.CATEGORICAL {
                var comps = jsonToMap(k);
                var perm = if comps.contains["permutation"] then comps["permutation"] else "";
                var seg = if comps.contains["segments"] then comps["segments"] else "";
                var cre = new shared CategoricalRegEntry(comps["codes"], comps["categories"], comps["NA_codes"], perm, seg);
                key_list.pushBack(cre);
            }
            else {
                var errorMsg = "GroupBys only support pdarray, Strings, and Categorical keys. Found %s".format(objType: string);
                throw getErrorWithContext(
                    msg=errorMsg,
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="IllegalArgumentError");
            }
        }

        var gbre = new shared GroupByRegEntry(seg_name, perm_name, key_list, uki);
        st.registry.register_groupby(reg_name, gbre);
        return new MsgTuple("Registered GroupBy", MsgType.NORMAL);
    }

    private proc register_categorical(msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var reg_name = msgArgs.getValueOf("name");
        var codes = msgArgs.getValueOf("codes");
        var categories = msgArgs.getValueOf("categories");
        var naCode = msgArgs.getValueOf("_akNAcode");
        var permutation: string;
        var segments: string;
        var perm_seg_exist: bool = false;
        if msgArgs.contains("permutation") && msgArgs.contains("segments") {
            permutation = msgArgs.getValueOf("permutation");
            segments = msgArgs.getValueOf("segments");
        }
        var cre = new shared CategoricalRegEntry(codes, categories, naCode, permutation, segments);
        st.registry.register_categorical(reg_name, cre);
        return new MsgTuple("Registered Categorical", MsgType.NORMAL);
    }

    private proc register_index(msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var reg_name = msgArgs.getValueOf("name");
        var num_idxs = msgArgs.get("num_idxs").getIntValue();
        var idx_names = msgArgs.get("idx_names").getList(num_idxs);
        var idx_types = msgArgs.get("idx_types").getList(num_idxs);
        var idx_objType = msgArgs.getValueOf("objType").toUpper(): ObjType;

        var idx: list(shared AbstractRegEntry);
        for (i, ot) in zip(idx_names, idx_types) {
            var objType: ObjType = ot.toUpper(): ObjType;
            if objType == ObjType.PDARRAY || objType == ObjType.STRINGS {
                var are = new shared ArrayRegEntry(i, objType);
                idx.pushBack(are);
            }
            else if objType == ObjType.CATEGORICAL {
                var comps = jsonToMap(i);
                var perm = if comps.contains["permutation"] then comps["permutation"] else "";
                var seg = if comps.contains["segments"] then comps["segments"] else "";
                var cre = new shared CategoricalRegEntry(comps["codes"], comps["categories"], comps["NA_codes"], perm, seg);
                idx.pushBack(cre);
            }
            else {
                var errorMsg = "Index only support pdarray, Strings, and Categorical ObjTypes. Found %s".format(objType: string);
                throw getErrorWithContext(
                    msg=errorMsg,
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="IllegalArgumentError");
            }
        }
        var ire = new shared IndexRegEntry(idx, idx_objType);
        st.registry.register_index(reg_name, ire);
        return new MsgTuple("Registered Index", MsgType.NORMAL);
    }

    private proc register_series(msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var reg_name = msgArgs.getValueOf("name");
        var num_idxs = msgArgs.get("num_idxs").getIntValue();
        var idx_names = msgArgs.get("idx_names").getList(num_idxs);
        var idx_types = msgArgs.get("idx_types").getList(num_idxs);
        var idx_objType = msgArgs.getValueOf("objType").toUpper(): ObjType;

        var idx: list(shared AbstractRegEntry);
        for (i, ot) in zip(idx_names, idx_types) {
            var objType: ObjType = ot.toUpper(): ObjType;
            if objType == ObjType.PDARRAY || objType == ObjType.STRINGS {
                var are = new shared ArrayRegEntry(i, objType);
                idx.pushBack(are);
            }
            else if objType == ObjType.CATEGORICAL {
                var comps = jsonToMap(i);
                var perm = if comps.contains["permutation"] then comps["permutation"] else "";
                var seg = if comps.contains["segments"] then comps["segments"] else "";
                var cre = new shared CategoricalRegEntry(comps["codes"], comps["categories"], comps["NA_codes"], perm, seg);
                idx.pushBack(cre);
            }
            else {
                var errorMsg = "Index only support pdarray, Strings, and Categorical ObjTypes. Found %s".format(objType: string);
                throw getErrorWithContext(
                    msg=errorMsg,
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="IllegalArgumentError");
            }
        }
        var ire = new shared IndexRegEntry(idx, idx_objType);

        var sre: shared SeriesRegEntry;
        // values can be pdarray, strings, categorical
        var values = msgArgs.getValueOf("values");
        var val_type = msgArgs.getValueOf("val_type").toUpper(): ObjType;
        if val_type == ObjType.PDARRAY || val_type == ObjType.STRINGS {
            var are = new shared ArrayRegEntry(values, val_type);
            sre = new shared SeriesRegEntry(ire, are: shared GenRegEntry);
        }
        else if val_type == ObjType.CATEGORICAL {
            var comps = jsonToMap(values);
            var perm = if comps.contains["permutation"] then comps["permutation"] else "";
            var seg = if comps.contains["segments"] then comps["segments"] else "";
            var cre = new shared CategoricalRegEntry(comps["codes"], comps["categories"], comps["NA_codes"], perm, seg);
            sre = new shared SeriesRegEntry(ire, cre: shared GenRegEntry);
        }
        else {
            var errorMsg = "Series only support pdarray, Strings, and Categorical ObjTypes. Found %s".format(val_type: string);
            throw getErrorWithContext(
                msg=errorMsg,
                lineNumber=getLineNumber(),
                routineName=getRoutineName(),
                moduleName=getModuleName(),
                errorClass="IllegalArgumentError");
        }

        st.registry.register_series(reg_name, sre);
        return new MsgTuple("Registered Series", MsgType.NORMAL);
    }

    proc register_bitvector(msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var reg_name = msgArgs.getValueOf("name");
        var values = msgArgs.getValueOf("values");
        var width = msgArgs.get("width").getIntValue();
        var reverse = msgArgs.get("reverse").getBoolValue();
        var bre = new shared BitVectorRegEntry(values, width, reverse);
        st.registry.register_bitvector(reg_name, bre);
        return new MsgTuple("Registered BitVector", MsgType.NORMAL);
    }

    proc registerMsg(cmd: string, msgArgs: borrowed MessageArgs,
                        st: borrowed SymTab): MsgTuple throws {
        var objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
        select objtype {
            when ObjType.PDARRAY, ObjType.STRINGS, ObjType.DATETIME, ObjType.TIMEDELTA, ObjType.IPV4 {
                return register_array(msgArgs, st);
            }
            when ObjType.SEGARRAY {
                return register_segarray(msgArgs, st);
            }
            when ObjType.DATAFRAME {
                return register_dataframe(msgArgs, st);
            }
            when ObjType.GROUPBY {
                return register_groupby(msgArgs, st);
            }
            when ObjType.CATEGORICAL {
                return register_categorical(msgArgs, st);
            }
            when ObjType.INDEX {
                return register_index(msgArgs, st);
            }
            when ObjType.MULTIINDEX {
                return register_index(msgArgs, st);
            }
            when ObjType.SERIES {
                return register_series(msgArgs, st);
            }
            when ObjType.BITVECTOR {
                return register_bitvector(msgArgs, st);
            }
            otherwise {
                var errorMsg = "ObjType Not Supported by Registry: %s".format(objtype: string);
                throw getErrorWithContext(
                    msg=errorMsg,
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="TypeError");
            }
        }
    }

    proc unregisterMsg(cmd: string, msgArgs: borrowed MessageArgs,
                        st: borrowed SymTab): MsgTuple throws {
        var name = msgArgs.getValueOf("name");
        var gre = st.registry.lookup(name): shared GenRegEntry;
        select gre.objType {
            when ObjType.PDARRAY, ObjType.STRINGS, ObjType.DATETIME, ObjType.TIMEDELTA, ObjType.IPV4 {
                var are = gre: shared ArrayRegEntry;
                st.registry.unregister_array(are);
            }
            when ObjType.SEGARRAY {
                var sre = gre: shared SegArrayRegEntry;
                st.registry.unregister_segarray(sre);
            }
            when ObjType.DATAFRAME {
                var dfre = gre: shared DataFrameRegEntry;
                st.registry.unregister_dataframe(dfre);
            }
            when ObjType.GROUPBY {
                var gbre = gre: shared GroupByRegEntry;
                st.registry.unregister_groupby(gbre);
            }
            when ObjType.CATEGORICAL {
                var cre = gre: shared CategoricalRegEntry;
                st.registry.unregister_categorical(cre);
            }
            when ObjType.INDEX {
                var ire = gre: shared IndexRegEntry;
                st.registry.unregister_index(ire);
            }
            when ObjType.MULTIINDEX {
                var ire = gre: shared IndexRegEntry;
                st.registry.unregister_index(ire);
            }
            when ObjType.SERIES {
                var sre = gre: shared SeriesRegEntry;
                st.registry.unregister_series(sre);
            }
            when ObjType.BITVECTOR {
                var bre = gre: shared BitVectorRegEntry;
                st.registry.unregister_bitvector(bre);
            }
            otherwise {
                var errorMsg = "ObjType Not Supported by Registry: %s".format(gre.objType: string);
                throw getErrorWithContext(
                    msg=errorMsg,
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="TypeError");
            }
        }
        return new MsgTuple("Unregistered %s %s".format(gre.objType: string, name), MsgType.NORMAL);
    }

    proc attachMsg(cmd: string, msgArgs: borrowed MessageArgs,
                        st: borrowed SymTab): MsgTuple throws {
        var name = msgArgs.getValueOf("name");
        var gre = st.registry.lookup(name): shared GenRegEntry;
        var rtnMap: map(string, string);
        select gre.objType {
            when ObjType.PDARRAY {
                var are = gre: shared ArrayRegEntry;
                rtnMap = are.asMap(st);
            }
            when ObjType.STRINGS {
                var are = gre: shared ArrayRegEntry;
                rtnMap = are.asMap(st);
            }
            when ObjType.DATETIME {
                var are = gre: shared ArrayRegEntry;
                rtnMap = are.asMap(st);
            }
            when ObjType.TIMEDELTA {
                var are = gre: shared ArrayRegEntry;
                rtnMap = are.asMap(st);
            }
            when ObjType.IPV4 {
                var are = gre: shared ArrayRegEntry;
                rtnMap = are.asMap(st);
            }
            when ObjType.SEGARRAY {
                var sre = gre: shared SegArrayRegEntry;
                rtnMap = sre.asMap(st);
            }
            when ObjType.DATAFRAME {
                var dfre = gre: shared DataFrameRegEntry;
                rtnMap = dfre.asMap(st);
            }
            when ObjType.GROUPBY {
                var gbre = gre: shared GroupByRegEntry;
                rtnMap = gbre.asMap(st);
            }
            when ObjType.CATEGORICAL {
                var cre = gre: shared CategoricalRegEntry;
                rtnMap = cre.asMap(st);
            }
            when ObjType.INDEX {
                var ire = gre: shared IndexRegEntry;
                rtnMap = ire.asMap(st);
            }
            when ObjType.MULTIINDEX {
                var ire = gre: shared IndexRegEntry;
                rtnMap = ire.asMap(st);
            }
            when ObjType.SERIES {
                var sre = gre: shared SeriesRegEntry;
                rtnMap = sre.asMap(st);
            }
            when ObjType.BITVECTOR {
                var bre = gre: shared BitVectorRegEntry;
                rtnMap = bre.asMap(st);
            }
            otherwise {
                var errorMsg = "Unexpected ObjType, %s, found in registry.".format(gre.objType: string);
                throw getErrorWithContext(
                    msg=errorMsg,
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="TypeError");
            }
        }
        return new MsgTuple(formatJson(rtnMap), MsgType.NORMAL);
    }

    proc listRegistryMsg(cmd: string, msgArgs: borrowed MessageArgs,
                            st: borrowed SymTab): MsgTuple throws {
        return new MsgTuple(st.registry.list_registry(), MsgType.NORMAL);
    }
    
    use CommandMap;
    registerFunction("register", registerMsg, getModuleName());
    registerFunction("list_registry", listRegistryMsg, getModuleName());
    registerFunction("unregister", unregisterMsg, getModuleName());
    registerFunction("attach", attachMsg, getModuleName());
}
