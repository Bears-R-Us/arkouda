module MultiTypeRegEntry {
    use Reflection;
    use Set;
    use List;
    use Map;
    use ServerErrors;

    use ServerConfig;
    use Logging;
    use AryUtil;
    use SegmentedString;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use GenSymIO;
    use NumPyDType;
    use IOUtils;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const regLogger = new Logger(logLevel, logChannel);

    enum RegistryEntryType {
        AbstractRegEntry,
            GenRegEntry,
                ArrayRegEntry,
                DataFrameRegEntry,
                GroupByRegEntry,
                CategoricalRegEntry,
                SegArrayRegEntry,
                IndexRegEntry, 
                SeriesRegEntry,
                BitVectorRegEntry,           
    }

    class AbstractRegEntry {
        var entryType:RegistryEntryType;
        var assignableTypes:set(RegistryEntryType); // All subclasses should add their type to this set
        var name = ""; // used to track user defined name assigned to the entry

        proc init() {
            this.entryType = RegistryEntryType.AbstractRegEntry;
            this.assignableTypes = new set(RegistryEntryType);
            this.assignableTypes.add(this.entryType);
        }

        proc setName(name: string) {
            this.name = name;
        }
    }

    class GenRegEntry: AbstractRegEntry {
        var objType: ObjType;

        proc init(objType: ObjType) {
            this.entryType = RegistryEntryType.GenRegEntry;
            assignableTypes.add(this.entryType);
            this.objType = objType;
        }

        proc toDataFrameRegEntry() {
            return try! this: borrowed DataFrameRegEntry;
        }
    }

    class ArrayRegEntry: GenRegEntry {
        var array: string; // name of symbol table entry
        proc init(array_name: string, objType: ObjType) {
            super.init(objType);
            this.array = array_name;
            this.name = array_name;
        }

        proc asMap(st: borrowed SymTab): map(string, string) throws {
            var rtnMap: map(string, string);
            rtnMap.add("objType", this.objType: string);
            if this.objType == ObjType.STRINGS {
                var segStr = getSegString(this.array, st);
                rtnMap.add("create", "created " + st.attrib(this.array) + "+created bytes.size " + segStr.nBytes: string);
            }
            else {
                rtnMap.add("create", "created " + st.attrib(this.array));
            }
            return rtnMap;
        }
    }

    class BitVectorRegEntry: GenRegEntry {
        var array: string;
        var width: int;
        var reverse: bool;

        proc init(array_name: string, width: int, reverse: bool) {
            super.init(ObjType.BITVECTOR);
            this.array = array_name;
            this.width = width;
            this.reverse = reverse;
            this.name = array_name;
        }

        proc asMap(st:borrowed SymTab): map(string, string) throws {
            var rtnMap: map(string, string);
            rtnMap.add("objType", this.objType: string);
            var comps: map(string, string);
            comps.add("values", "created " + st.attrib(this.array));
            comps.add("width", this.width: string);
            comps.add("reverse", this.reverse: string);
            rtnMap.add("create", formatJson(comps));
            return rtnMap;
        }
    }

    class SegArrayRegEntry: GenRegEntry {
        var segments: string;
        var values: shared ArrayRegEntry;
        var lengths: string;

        proc init(segments: string, values: shared ArrayRegEntry, lengths: string) {
            super.init(ObjType.SEGARRAY);
            this.segments = segments;
            this.values = values;
            this.lengths = lengths;
        }

        proc asMap(st: borrowed SymTab): map(string, string) throws {
            var rtnMap: map(string, string);
            rtnMap.add("objType", this.objType: string);

            var comp_create: map(string, string);
            comp_create.add("segments", "created " + st.attrib(this.segments));
            var val_map = this.values.asMap(st);
            comp_create.add("values", val_map["create"]);
            if this.lengths != "" {
                comp_create.add("lengths", "created " + st.attrib(this.lengths));
            }
            rtnMap.add("create", formatJson(comp_create));
            return rtnMap;
        } 
    }

    class DataFrameRegEntry: GenRegEntry {
        var idx: string; // sym_tab name of the dataframe index
        var column_names: list(string); // list of column names
        var columns: list(shared AbstractRegEntry); // list of sym tab names for each column+components of the column
        proc init(idx: string, column_names: list(string), columns: list(shared AbstractRegEntry)) {
            super.init(ObjType.DATAFRAME);
            this.idx = idx;
            this.column_names = column_names;
            this.columns = columns;
        }

        proc asMap(st: borrowed SymTab): map(string, string) throws {
            var rtnMap: map(string, string);
            rtnMap.add("objType", this.objType: string);
            var col_creates: map(string, string); // map column name to create statement
            for (cname, c) in zip(this.column_names, this.columns) {
                var gre = c: borrowed GenRegEntry;
                var col_map: map(string, string);
                select gre.objType {
                    when ObjType.PDARRAY, ObjType.STRINGS, ObjType.DATETIME, ObjType.TIMEDELTA, ObjType.IPV4 {
                        var are = gre: borrowed ArrayRegEntry;
                        col_map = are.asMap(st);
                    }
                    when ObjType.CATEGORICAL {
                        var cre = gre: borrowed CategoricalRegEntry;
                        col_map = cre.asMap(st);
                    }
                    when ObjType.SEGARRAY {
                        var sre = gre: borrowed SegArrayRegEntry;
                        col_map = sre.asMap(st);
                    }
                    when ObjType.BITVECTOR {
                        var bre = gre: borrowed BitVectorRegEntry;
                        col_map = bre.asMap(st);
                    }
                    otherwise {
                        throw getErrorWithContext(
                            msg="Invalid DataFrame column ObjType, %s".format(gre.objType: string),
                            lineNumber=getLineNumber(),
                            routineName=getRoutineName(),
                            moduleName=getModuleName(),
                            errorClass="IllegalArgumentError");
                    }
                }
                
                var create_str = "%s+|+%s".format(col_map["objType"], col_map["create"]);
                col_creates.add(cname, create_str);
            }
            rtnMap.add("create", formatJson(col_creates));
            return rtnMap;
        }
    }

    class GroupByRegEntry: GenRegEntry {
        var segments: string;
        var permutation: string;
        var keys: list(shared AbstractRegEntry);
        var uki: string;

        proc init(segments: string, permutation: string, keys: list(shared AbstractRegEntry), uki: string) {
            super.init(ObjType.GROUPBY);
            this.segments = segments;
            this.permutation = permutation;
            this.keys = keys;
            this.uki = uki;
        }

        proc asMap(st: borrowed SymTab): map(string, string) throws {
            var rtnMap: map(string, string);
            rtnMap.add("objType", this.objType: string);
            var comp_create: map(string, string); // map column name to create statement

            comp_create.add("segments", "created " + st.attrib(this.segments));
            comp_create.add("permutation", "created " + st.attrib(this.permutation));
            comp_create.add("uki", "created " + st.attrib(this.uki));

            for (k, i) in zip(this.keys, 0..) {
                var k_map: map(string, string);
                var gre = k: borrowed GenRegEntry;
                if gre.objType == ObjType.PDARRAY || gre.objType == ObjType.STRINGS {
                    var are = gre: borrowed ArrayRegEntry;
                    k_map = are.asMap(st);
                }
                else if gre.objType == ObjType.CATEGORICAL {
                    var cre = gre: borrowed CategoricalRegEntry;
                    k_map = cre.asMap(st);
                }
                var create_str = "%s+|+%s".format(k_map["objType"], k_map["create"]);
                comp_create.add("KEY_%i".format(i), create_str);
            }
            rtnMap.add("create", formatJson(comp_create));
            return rtnMap;
        }
    }

    class CategoricalRegEntry: GenRegEntry {
        var codes: string;
        var categories: string;
        var permutation: string;
        var segments: string;
        var naCode: string;

        proc init(codes: string, categories: string, naCode: string, permutation: string, segments: string) {
            super.init(ObjType.CATEGORICAL);
            this.codes = codes;
            this.categories = categories;
            this.permutation = permutation;
            this.segments = segments;
            this.naCode = naCode;
        }

        proc asMap(st: borrowed SymTab): map(string, string) throws {
            var rtnMap: map(string, string);
            rtnMap.add("objType", this.objType: string);

            var comp_create: map(string, string);
            comp_create.add("codes", "created " + st.attrib(this.codes));
            comp_create.add("_akNAcode", "created " + st.attrib(this.naCode));
            var segStr = getSegString(this.categories, st);
            comp_create.add("categories", "created " + st.attrib(this.categories) + "+created bytes.size " + segStr.nBytes: string);

            if this.permutation != "" && this.segments != "" {
                comp_create.add("permutation", "created " + st.attrib(this.permutation));
                comp_create.add("segments", "created " + st.attrib(this.segments));
            }
            rtnMap.add("create", formatJson(comp_create));
            return rtnMap;
        }
    }

    class IndexRegEntry: GenRegEntry {
        var idx: list(shared AbstractRegEntry);

        proc init(idx: list(shared AbstractRegEntry), objType: ObjType) {
            super.init(objType);
            this.idx = idx;

        }

        proc asMap(st: borrowed SymTab): map(string, string) throws {
            var rtnMap: map(string, string);
            rtnMap.add("objType", this.objType: string);
            var idxList: list(string);
            for i in idx {
                var idx_map: map(string, string);
                var gre = i: borrowed GenRegEntry;
                select gre.objType {
                    when ObjType.PDARRAY, ObjType.STRINGS {
                        var are = gre: borrowed ArrayRegEntry;
                        idx_map = are.asMap(st);
                    }
                    when ObjType.CATEGORICAL {
                        var cre = gre: borrowed CategoricalRegEntry;
                        idx_map = cre.asMap(st);
                    }
                    otherwise {
                        throw getErrorWithContext(
                            msg="Invalid Index ObjType, %s".format(gre.objType: string),
                            lineNumber=getLineNumber(),
                            routineName=getRoutineName(),
                            moduleName=getModuleName(),
                            errorClass="IllegalArgumentError");
                    }
                }
                var create_str = "%s+|+%s".format(idx_map["objType"], idx_map["create"]);
                idxList.pushBack(create_str);
            }
            rtnMap.add("create", formatJson(idxList));
            return rtnMap;
        }
    }

    class SeriesRegEntry: GenRegEntry {
        var idx: shared IndexRegEntry;
        var values: shared GenRegEntry;

        proc init(idx: shared IndexRegEntry, values: shared GenRegEntry) {
            super.init(ObjType.SERIES);
            this.idx = idx;
            this.values = values;
        }

        proc asMap(st: borrowed SymTab): map(string, string) throws {
            var rtnMap: map(string, string);
            rtnMap.add("objType", this.objType: string);

            var comp_create: map(string, string);
            var i_map = this.idx.asMap(st);
            comp_create.add("index", i_map["create"]);
            var val_map: map(string, string);
            if this.values.objType == ObjType.PDARRAY || this.values.objType == ObjType.STRINGS {
                var are = this.values: shared ArrayRegEntry;
                val_map = are.asMap(st);
            }
            else if this.values.objType == ObjType.CATEGORICAL {
                var cre = this.values: shared CategoricalRegEntry;
                val_map = cre.asMap(st);
            }
            comp_create.add("value", "%s+|+%s".format(val_map["objType"], val_map["create"]));

            rtnMap.add("create", formatJson(comp_create));
            return rtnMap;
        }
    }
}
