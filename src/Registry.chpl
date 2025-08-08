module Registry {
    use MultiTypeRegEntry;
    use ServerConfig;
    use ServerErrorStrings;;
    use Map;
    use List;
    use GenSymIO;
    use Reflection;
    use ServerErrors;
    use Logging;
    use IOUtils;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const regLogger = new Logger(logLevel, logChannel);

    class RegTab {
        var registered_entries: list(string);
        var tab: map(string, shared AbstractRegEntry);

        proc register_array(name: string, are: shared ArrayRegEntry) throws {
            checkAvailability(name);
            tab.addOrReplace(name, are);
            registered_entries.pushBack(are.array);
            are.setName(name);
        }

        proc register_segarray_components(sre: SegArrayRegEntry) throws {
            registered_entries.pushBack(sre.segments);
            registered_entries.pushBack(sre.values.array);
            if sre.lengths != "" {
                registered_entries.pushBack(sre.lengths);
            }
        }

        proc register_segarray(name: string, sre: shared SegArrayRegEntry) throws {
            checkAvailability(name);
            tab.addOrReplace(name, sre);
            register_segarray_components(sre);
            sre.setName(name);
        }

        proc register_dataframe(name: string, dfre: shared DataFrameRegEntry) throws {
            checkAvailability(name);
            tab.addOrReplace(name, dfre);
            registered_entries.pushBack(dfre.idx);
            for c in dfre.columns {
                var gre = c: borrowed GenRegEntry;
                select gre.objType {
                    when ObjType.PDARRAY, ObjType.STRINGS, ObjType.DATETIME, ObjType.TIMEDELTA, ObjType.IPV4 {
                        var are = gre: borrowed ArrayRegEntry;
                        registered_entries.pushBack(are.array);
                    }
                    when ObjType.CATEGORICAL {
                        var cre = gre: borrowed CategoricalRegEntry;
                        register_categorical_components(cre);
                    }
                    when ObjType.SEGARRAY {
                        var sre = gre: borrowed SegArrayRegEntry;
                        register_segarray_components(sre);
                    }
                    when ObjType.BITVECTOR {
                        var bre = gre: borrowed BitVectorRegEntry;
                        registered_entries.pushBack(bre.array);
                    }
                    otherwise {
                        var errorMsg = "Dataframes only support pdarray, Strings, SegArray, and Categorical columns. Found %s".format(gre.objType: string);
                        throw getErrorWithContext(
                            msg=errorMsg,
                            lineNumber=getLineNumber(),
                            routineName=getRoutineName(),
                            moduleName=getModuleName(),
                            errorClass="IllegalArgumentError");
                    }
                }
            }
            dfre.setName(name);
        }

        proc register_groupby(name: string, gbre: shared GroupByRegEntry) throws {
            checkAvailability(name);
            tab.addOrReplace(name, gbre);
            registered_entries.pushBack(gbre.segments);
            registered_entries.pushBack(gbre.permutation);
            registered_entries.pushBack(gbre.uki);

            for k in gbre.keys {
                var gre = k: borrowed GenRegEntry;
                if gre.objType == ObjType.PDARRAY || gre.objType == ObjType.STRINGS {
                    var are = gre: borrowed ArrayRegEntry;
                    registered_entries.pushBack(are.array);
                }
                else if gre.objType == ObjType.CATEGORICAL {
                    var cre = gre: borrowed CategoricalRegEntry;
                    register_categorical_components(cre);
                }
            }
            gbre.setName(name);
        }

        proc register_categorical_components(cre: CategoricalRegEntry) throws {
            registered_entries.pushBack(cre.codes);
            registered_entries.pushBack(cre.categories);
            registered_entries.pushBack(cre.naCode);
            if cre.permutation != "" && cre.segments != "" {
                registered_entries.pushBack(cre.permutation);
                registered_entries.pushBack(cre.segments);
            }
        }

        proc register_categorical(name: string, cre: shared CategoricalRegEntry) throws {
            checkAvailability(name);
            tab.addOrReplace(name, cre);
            register_categorical_components(cre);
            cre.setName(name);            
        }

        proc register_index_components(ire: IndexRegEntry) throws {
            for i in ire.idx {
                var gre = i: borrowed GenRegEntry;
                if gre.objType == ObjType.PDARRAY || gre.objType == ObjType.STRINGS {
                    var are = gre: borrowed ArrayRegEntry;
                    registered_entries.pushBack(are.array);
                }
                else if gre.objType == ObjType.CATEGORICAL {
                    var cre = gre: borrowed CategoricalRegEntry;
                    register_categorical_components(cre);
                }
            }
        }

        proc register_index(name: string, ire: shared IndexRegEntry) throws {
            checkAvailability(name);
            tab.addOrReplace(name, ire);
            register_index_components(ire);
            ire.setName(name);
        }

        proc register_series(name: string, sre: shared SeriesRegEntry) throws {
            checkAvailability(name);
            tab.addOrReplace(name, sre);
            register_index_components(sre.idx);

            if sre.values.objType == ObjType.PDARRAY || sre.values.objType == ObjType.STRINGS {
                var are = sre.values: borrowed ArrayRegEntry;
                registered_entries.pushBack(are.array);
            }
            else if sre.values.objType == ObjType.CATEGORICAL {
                var cre = sre.values: borrowed CategoricalRegEntry;
                register_categorical_components(cre);
            }
            sre.setName(name);
        }

        proc register_bitvector(name: string, bre: shared BitVectorRegEntry) throws {
            checkAvailability(name);
            tab.addOrReplace(name, bre);
            registered_entries.pushBack(bre.array);
            bre.setName(name);
        }

        proc unregister_array(are: shared ArrayRegEntry) throws {
            registered_entries.remove(are.array);
            tab.remove(are.name);
        }

        proc unregister_segarray_components(sre: SegArrayRegEntry) throws {
            registered_entries.remove(sre.segments);
            registered_entries.remove(sre.values.array);
            if sre.lengths != "" {
                registered_entries.remove(sre.lengths);
            }
        }

        proc unregister_segarray(sre: shared SegArrayRegEntry) throws {
            unregister_segarray_components(sre);
            tab.remove(sre.name);
        }

        proc unregister_dataframe(dfre: shared DataFrameRegEntry) throws {
            registered_entries.remove(dfre.idx);
            for c in dfre.columns {
                var gre = c: borrowed GenRegEntry;
                select gre.objType {
                    when ObjType.PDARRAY, ObjType.STRINGS, ObjType.DATETIME, ObjType.TIMEDELTA, ObjType.IPV4 {
                        var are = gre: borrowed ArrayRegEntry;
                        registered_entries.remove(are.array);
                    }
                    when ObjType.CATEGORICAL {
                        var cre = gre: borrowed CategoricalRegEntry;
                        unregister_categorical_components(cre);
                    }
                    when ObjType.SEGARRAY {
                        var sre = gre: borrowed SegArrayRegEntry;
                        unregister_segarray_components(sre);
                    }
                    when ObjType.BITVECTOR {
                        var bre = gre: borrowed BitVectorRegEntry;
                        registered_entries.remove(bre.array);
                    }
                    otherwise {
                        var errorMsg = "Dataframes only support pdarray, Strings, Datetime, Timedelta, IPv4, BitVector, SegArray, and Categorical columns. Found %s".format(gre.objType: string);
                        throw getErrorWithContext(
                            msg=errorMsg,
                            lineNumber=getLineNumber(),
                            routineName=getRoutineName(),
                            moduleName=getModuleName(),
                            errorClass="TypeError");
                    }
                }
            }
            tab.remove(dfre.name);
        }

        proc unregister_groupby(gbre: shared GroupByRegEntry) throws {
            registered_entries.remove(gbre.segments);
            registered_entries.remove(gbre.permutation);
            registered_entries.remove(gbre.uki);
            for k in gbre.keys { 
                var gre = k: borrowed GenRegEntry;
                if gre.objType == ObjType.PDARRAY || gre.objType == ObjType.STRINGS {
                    var are = gre: borrowed ArrayRegEntry;
                    registered_entries.remove(are.array);
                }
                else if gre.objType == ObjType.CATEGORICAL {
                    var cre = gre: borrowed CategoricalRegEntry;
                    unregister_categorical_components(cre);
                }
            }
            tab.remove(gbre.name);
        }

        proc unregister_categorical_components(cre: CategoricalRegEntry) throws {
            registered_entries.remove(cre.codes);
            registered_entries.remove(cre.categories);
            registered_entries.remove(cre.naCode);
            if cre.permutation != "" && cre.segments != "" {
                registered_entries.remove(cre.permutation);
                registered_entries.remove(cre.segments);
            }
        }

        proc unregister_categorical(cre: shared CategoricalRegEntry) throws {
            unregister_categorical_components(cre);
            tab.remove(cre.name);            
        }

        proc unregister_index_components(ire: IndexRegEntry) throws {
            for i in ire.idx {
                var gre = i: borrowed GenRegEntry;
                if gre.objType == ObjType.PDARRAY || gre.objType == ObjType.STRINGS {
                    var are = gre: borrowed ArrayRegEntry;
                    registered_entries.remove(are.array);
                }
                else if gre.objType == ObjType.CATEGORICAL {
                    var cre = gre: borrowed CategoricalRegEntry;
                    unregister_categorical_components(cre);
                }
            }
        }

        proc unregister_index(ire: shared IndexRegEntry) throws {
            unregister_index_components(ire);
            tab.remove(ire.name);
        }

        proc unregister_series(sre: shared SeriesRegEntry) throws {
            unregister_index_components(sre.idx);

            if sre.values.objType == ObjType.PDARRAY || sre.values.objType == ObjType.STRINGS {
                var are = sre.values: borrowed ArrayRegEntry;
                registered_entries.remove(are.array);
            }
            else if sre.values.objType == ObjType.CATEGORICAL {
                var cre = sre.values: borrowed CategoricalRegEntry;
                unregister_categorical_components(cre);
            }
            tab.remove(sre.name);
        }

        proc unregister_bitvector(bre: shared BitVectorRegEntry) throws {
            registered_entries.remove(bre.array);
            tab.remove(bre.name);
        }

        proc lookup(name: string): shared AbstractRegEntry throws {
            checkTable(name, "lookup");
            return tab[name];
        }

        proc checkAvailability(name: string) throws {
            if tab.contains(name) {
                regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                                "Name, %s, not available for registration. Already in use.".format(name));
                var errorMsg = "Name, %s, not available for registration. Already in use.".format(name); 
                    throw getErrorWithContext(
                        msg=errorMsg,
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="RegistrationError");
            }
        }

        /**
         * checks to see if a symbol is defined if it is not it throws an exception 
        */
        proc checkTable(name: string, calling_func="check") throws { 
            if (!tab.contains(name)) { 
                regLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                                "undefined registry entry: %s".format(name));
                throw getErrorWithContext(
                    msg=unknownSymbolError(pname=calling_func, sname=name),
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="UnknownSymbolError");
            } else {
                regLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                "found registered object: %s".format(name));
            }
        }

        proc contains(name: string): bool throws {
            // registered_entries contains names in symbol table we do not want
            // deleted.
            return registered_entries.contains(name);
        }

        proc list_registry(): string throws {
            var rtnMap: map(string, string);
            // create array of object names
            var objs: [0..#tab.size] string;
            for (name, o) in zip(tab.keys(), objs) {
                o = name;
            }
            rtnMap.addOrReplace("Objects", formatJson(objs));

            // if detailed, provide list of object types
            var obj_types: [0..#tab.size] string;
            for (o, ots) in zip(tab.values(), obj_types) {
                var gre = o: borrowed GenRegEntry;
                ots = gre.objType: string;
            }
            rtnMap.addOrReplace("Object_Types", formatJson(obj_types));
            
            rtnMap.addOrReplace("Components", formatJson(registered_entries));
            return formatJson(rtnMap);
        }
    }
}
