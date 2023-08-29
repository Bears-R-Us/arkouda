module MultiTypeSymbolTable
{
    use ServerConfig;
    use Security;
    use ServerErrorStrings;
    use Reflection;
    use ServerErrors;
    use Logging;
    use BigInteger;
    
    use MultiTypeSymEntry;
    use IO;

    use Map;
    use ArkoudaRegexCompat;
    use ArkoudaFileCompat;
    use ArkoudaMapCompat;
    use ArkoudaIOCompat;
    use Registry;
    
    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const mtLogger = new Logger(logLevel, logChannel);

    /* symbol table */
    class SymTab
    {
        /*
          Similar to the Symbol Table but for register object tracking
        */
        var registry = new owned RegTab();

        /*
          Map indexed by strings
        */
        var tab: map(string, shared AbstractSymEntry);

        var serverid = "id_" + generateToken(8) + "_";
        var nid = 0;

        /*
        Gives out symbol names.
        */
        proc nextName():string {
            nid += 1;
            return serverid + nid:string;
        }

        /*
            Takes args and creates a new SymEntry.

            :arg name: name of the array
            :type name: string

            :arg len: length of array
            :type len: int

            :arg t: type of array

            :returns: borrow of newly created `SymEntry(t)`
        */
        proc addEntry(name: string, len: int, type t): borrowed SymEntry(t) throws {
            var A = makeDistArray(len, t);

            var entry = createSymEntry(A);
            if (tab.contains(name)) {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "redefined symbol: %s ".doFormat(name));
            } else {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "adding symbol: %s ".doFormat(name));            
            }

            tab.addOrReplace(name, entry);
            entry.setName(name);
            // When we retrieve from table, it comes back as AbstractSymEntry so we need to cast it
            // back to the original type. Since we know it already we can skip isAssignableTo check
            return (tab[name]:borrowed GenSymEntry).toSymEntry(t);
        }

        /*
        Takes an already created AbstractSymEntry and creates a new AbstractSymEntry.

        :arg name: name of the array
        :type name: string

        :arg entry: AbstractSymEntry to convert
        :type entry: AbstractSymEntry

        :returns: borrow of newly created AbstractSymEntry
        */
        proc addEntry(name: string, in entry: shared AbstractSymEntry): borrowed AbstractSymEntry throws {
            // check and throw if memory limit would be exceeded
            if entry.isAssignableTo(SymbolEntryType.TypedArraySymEntry) {
                overMemLimit( (entry:GenSymEntry).size * (entry:GenSymEntry).itemsize);

            } // } else if entry.isAssignableTo(SymbolEntryType.CompositeSymEntry) {
            //     // TODO invoke memory check ... maybe the mem check should be part of the SymbolType API?
            // }
            

            if (tab.contains(name)) {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "redefined symbol: %s ".doFormat(name));
            } else {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "adding symbol: %s ".doFormat(name));            
            }

            tab.addOrReplace(name, entry);
            entry.setName(name);
            return tab[name];
        }

        /*
        Creates a symEntry from array name, length, and DType

        :arg name: name of the array
        :type name: string

        :arg len: length of array
        :type len: int

        :arg dtype: type of array

        :returns: borrow of newly created GenSymEntry
        */
        proc addEntry(name: string, len: int, dtype: DType): borrowed AbstractSymEntry throws {
            select dtype {
                when DType.Int64 { return addEntry(name, len, int); }
                when DType.UInt64 { return addEntry(name, len, uint); }
                when DType.Float64 { return addEntry(name, len, real); }
                when DType.Bool { return addEntry(name, len, bool); }
                when DType.BigInt { return addEntry(name, len, bigint); }
                otherwise { 
                    var errorMsg = "addEntry not implemented for %?".doFormat(dtype); 
                    throw getErrorWithContext(
                        msg=errorMsg,
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="IllegalArgumentError");
                }
            }
        }

        /*
        Removes an unregistered entry from the symTable

        :arg name: name of the array
        :type name: string

        :returns: bool indicating whether the deletion occurred
        */
        proc deleteEntry(name: string): bool throws {
            checkTable(name, "deleteEntry");
            if !registry.contains(name) {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "Deleting unregistered entry: %s".doFormat(name)); 
                tab.remove(name);
                return true;
            } else {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "Skipping registered entry: %s".doFormat(name)); 
                return false;
            }
        }

        /*
        Clears all unregistered entries from the symTable
        */
        proc clear() throws {
            mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "Clearing all unregistered entries"); 
            for n in tab.keysToArray() { deleteEntry(n); }
        }

        
        /**
         * Returns the AbstractSymEntry associated with the provided name, if the AbstractSymEntry exists
         * :arg name: string to index/query in the sym table
         * :type name: string

         * :returns: AbstractSymEntry or throws on error
         * :throws: `unkownSymbolError(name)`
         */
        proc lookup(name: string): borrowed AbstractSymEntry throws {
            checkTable(name, "lookup");
            return tab[name];
        }

        /**
         * checks to see if a symbol is defined if it is not it throws an exception 
        */
        proc checkTable(name: string, calling_func="check") throws { 
            if (!tab.contains(name)) { 
                mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                                "undefined symbol: %s".doFormat(name));
                throw getErrorWithContext(
                    msg=unknownSymbolError(pname=calling_func, sname=name),
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="UnknownSymbolError");
            } else {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                "found symbol: %s".doFormat(name));
            }
        }
        
        /**
         * Prints the SymTable in a pretty format (name,SymTable[name])
         */
        proc pretty() throws {
            for n in tab {
                writeln("%10s = ".doFormat(n), tab.getValue(n)); stdout.flush();
            }
        }

        /**
         * :returns: total bytes in arrays in the symbol table
         */
        proc memUsed(): int {
            var total: int = + reduce [e in tab.values()] e.getSizeEstimate();
            return total;
        }
        
        /*
        Attempts to format and return sym entries mapped to the provided string into JSON format.
        Pass __AllSymbols__ to process the entire sym table.

        :arg name: name of entry to be processed
        :type name: string
        */
        proc dump(name:string): string throws {
            if name == "__AllSymbols__" {
                return formatJson(this);
            } 
            checkTable(name, "dump");
            return formatJson(name) + " " + formatJson(tab.getReference(name));
        }
        
        /*
        Returns verbose attributes of the sym entries at the given string, if the string is a JSON formmated list of entry names.
        Pass __AllSymbols__ to process all sym entries in the sym table.
        Pass __RegisteredSymbols__ to process all registered sym entries.

        Returns: name, dtype, size, ndim, shape, item size, and registration status for each entry in names

        :arg names: list containing names of entries to be processed
        :type names: string

        :returns: JSON formatted list containing info for each entry in names
        */
        proc info(names:string): string throws {
            var entries;
            select names
            {
                when "__AllSymbols__"         {entries = getEntries(tab);}
                otherwise                     {entries = getEntries(parseJson(names));}
            }
            return "[%s]".doFormat(','.join(entries));
        }

        /*
        Convert JSON formmated list of entry names into a [] string object

        :arg names: JSON formatted list containing entry names
        :type names: string

        :returns: [] string of entry names
        */
        proc parseJson(names:string): [] string throws {
            var mem = openMemFile();
            mem.writer().write(names);
            var reader = mem.reader();

            var num_elements = 0;
            for i in names.split(',') {
                num_elements += 1;
            }

            var parsed_names: [1..num_elements] string;
            reader.readf("%jt", parsed_names);
            return parsed_names;
        }

        /*
        Returns an array of JSON formatted strings for each entry in infoList (tab, registry, or [names])

        :arg infoList: Iterable containing sym entries to be returned by info
        :type infoList: domain(string) or [] string for registry and [names]

        :returns: array of JSON formatted strings
        */
        proc getEntries(infoList): [] string throws {
            var entries: [1..infoList.size] string;
            var i = 0;
            for name in infoList {
                i+=1;
                checkTable(name);
                entries[i] = formatEntry(name, tab[name]);
            }
            return entries;
        }

        /*
        Returns an array of JSON formatted strings for each entry in infoList (tab, registry, or [names])

        :arg infoList: Iterable containing sym entries to be returned by info
        :type infoList: map(string, shared GenSymEntry) for tab

        :returns: array of JSON formatted strings
        */
        proc getEntries(infoList:map): [] string throws {
            var entries: [1..infoList.size] string;
            var i = 0;
            for name in infoList.keys() {
                i+=1;
                checkTable(name);
                entries[i] = formatEntry(name, tab[name]);
            }
            return entries;
        }

        /*
        Returns formatted string for an info entry.

        :arg name: name of entry to be formatted
        :type name: string

        :arg item: AbstractSymEntry to be formatted (tab[name])
        :type item: AbstractSymEntry

        :returns: JSON formatted dictionary
        */
        proc formatEntry(name:string, abstractEntry:borrowed AbstractSymEntry): string throws {
            if abstractEntry.isAssignableTo(SymbolEntryType.TypedArraySymEntry) {
                var item:borrowed GenSymEntry = toGenSymEntry(abstractEntry);
                return formatJson('{"name":%?, "dtype":%?, "size":%?, "ndim":%?, "shape":%?, "itemsize":%?, "registered":%?}',
                                  name, dtype2str(item.dtype), item.size, item.ndim, item.shape, item.itemsize, registry.contains(name));
            } else if abstractEntry.isAssignableTo(SymbolEntryType.SegStringSymEntry) {
                var item:borrowed SegStringSymEntry = toSegStringSymEntry(abstractEntry);
                return formatJson('{"name":%?, "dtype":%?, "size":%?, "ndim":%?, "shape":%?, "itemsize":%?, "registered":%?}',
                                  name, dtype2str(item.dtype), item.size, item.ndim, item.shape, item.itemsize, registry.contains(name));
                              
            } else {
              return formatJson('{"name":%?, "dtype":%?, "size":%?, "ndim":%?, "shape":%?, "itemsize":%?, "registered":%?}',
                                name, dtype2str(DType.UNDEF), 0, 0, (0,), 0, registry.contains(name));
            }
        }

        /*
        Returns raw attributes of the sym entry at the given string, if the string maps to an entry.
        Returns: name, dtype, size, ndim, shape, and item size

        :arg name: name of entry to be processed
        :type name: string

        :returns: s (string) containing info
        */
        proc attrib(name:string):string throws {
            checkTable(name, "attrib");

            var entry = tab[name];
            if entry.isAssignableTo(SymbolEntryType.TypedArraySymEntry){ //Anything considered a GenSymEntry
                var g:GenSymEntry = toGenSymEntry(entry);
                return "%s %s %? %? %? %?".doFormat(name, dtype2str(g.dtype), g.size, g.ndim, g.shape, g.itemsize);
            }
            else if entry.isAssignableTo(SymbolEntryType.CompositeSymEntry) { //CompositeSymEntry
                var c: CompositeSymEntry = toCompositeSymEntry(entry);
                return "%s %? %?".doFormat(name, c.size, c.ndim);
            }
            
            throw new Error("attrib - Unsupported Entry Type %s".doFormat(entry.entryType));
        }

        /*
        Attempts to find a sym entry mapped to the provided string, then 
        returns the data in the entry up to the specified threshold.
        Arrays of size less than threshold will be printed in their entirety. 
        Arrays of size greater than or equal to threshold will print the first 3 and last 3 elements

        :arg name: name of entry to be processed
        :type name: string

        :arg thresh: threshold for data to return
        :type thresh: int

        :returns: s (string) containing the array data
        */
        proc datastr(name: string, thresh:int): string throws {
            checkTable(name, "datastr");
            var u: borrowed AbstractSymEntry = tab[name];

            // I don't think we need to do this check, but I'm keeping the code around for now.
            // if (u.dtype == DType.UNDEF || u.dtype == DType.UInt8) {
            //     var s = unrecognizedTypeError("datastr",dtype2str(u.dtype));
            //     mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),s);
            //     return s;
            // }
            return u.__str__(thresh=thresh, prefix="[", suffix="]", baseFormat="%?");
        }

        /*
        Attempts to find a sym entry mapped to the provided string, then 
        returns the data in the entry up to the specified threshold. 
        This method returns the data in form "array([<DATA>])".
        Arrays of size less than threshold will be printed in their entirety. 
        Arrays of size greater than or equal to threshold will print the first 3 and last 3 elements

        :arg name: name of entry to be processed
        :type name: string

        :arg thresh: threshold for data to return
        :type thresh: int

        :returns: s (string) containing the array data
        */
        proc datarepr(name: string, thresh:int): string throws {
            checkTable(name, "datarepr");
            var entry = tab[name];
            if entry.isAssignableTo(SymbolEntryType.TypedArraySymEntry) {
                var u: borrowed GenSymEntry = toGenSymEntry(entry);
                if u.dtype == DType.UNDEF {
                    var s = unrecognizedTypeError("datarepr",dtype2str(u.dtype));
                    mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),s);
                    return s;
                }
                var frmt:string = if (u.dtype == DType.Float64) then "%.17r" else "%?";
                return u.__str__(thresh=thresh, prefix="array([", suffix="])", baseFormat=frmt);
            } else {
                return "Unhandled type %s".doFormat(entry.entryType);
            }

        }

        /*
        Attempts to find a sym entry mapped to the provided string, then 
        returns a boolean value signfying the provided string's sym entry 
        existance

        :arg name: name of entry to be checked
        :type name: string

        :returns: bool signifying existance of the sym entry
        */
        proc contains(name: string): bool {
            if tab.contains(name) {
                return true;
            } else {
                return false;
            }
        }

        /*
        Attempts to find all sym entries that match the provided regex string, then 
        returns a string array of matching names

        :arg pattern: regex string to search for
        :type pattern: string

        :returns: string array containing matching entry names
        */
        proc findAll(pattern: string): [] string throws {
            var rg = compile(pattern);
            var infoStr = "";
            forall name in tab.keysToArray() with (+ reduce infoStr) {
                var match = rg.match(name);
                if match.matched {
                    var end : int = (match.byteOffset: int) + match.numBytes;
                    infoStr += name[match.byteOffset..#end] + "+";
                }
            }
            return infoStr.strip("+").split("+");
        }
    }

    /**
     * Convenience proc to retrieve GenSymEntry from SymTab
     * Performs conversion from AbstractSymEntry to GenSymEntry
     * You can pass a logger from the calling function for better error reporting.
     */
    proc getGenericTypedArrayEntry(name:string, st: borrowed SymTab): borrowed GenSymEntry throws {
        var abstractEntry = st.lookup(name);
        if ! abstractEntry.isAssignableTo(SymbolEntryType.TypedArraySymEntry) {
            var errorMsg = "Error: SymbolEntryType %s is not assignable to GenSymEntry".doFormat(abstractEntry.entryType);
            mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw new Error(errorMsg);
        }
        return (abstractEntry: borrowed GenSymEntry);
    }

    /**
     * Convenience proc to retrieve SegStringSymEntry from SymTab
     * Performs conversion from AbstractySymEntry to SegStringSymEntry
     * You can pass a logger from the calling function for better error reporting.
     */
    proc getSegStringEntry(name:string, st: borrowed SymTab): borrowed SegStringSymEntry throws {
        var abstractEntry = st.lookup(name);
        if ! abstractEntry.isAssignableTo(SymbolEntryType.SegStringSymEntry) {
            var errorMsg = "Error: SymbolEntryType %s is not assignable to SegStringSymEntry".doFormat(abstractEntry.entryType);
            mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw new Error(errorMsg);
        }
        return (abstractEntry: borrowed SegStringSymEntry);
    }
}
