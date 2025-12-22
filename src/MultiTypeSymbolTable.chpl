module MultiTypeSymbolTable
{
    use ServerConfig;
    use Security;
    use ServerErrorStrings;
    use Reflection;
    use ServerErrors;
    use Logging;
    use BigInteger;
    use Regex;
    use MultiTypeSymEntry;
    use IO;
    use IOUtils;
    use Message;
    use UInt128;

    use Map;
    use Registry;

    import Message.ParameterObj;

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
            Insert a symbol into the table.

            Returns a symbol-creation message with the symbol's attributes
        */
        proc insert(in symbol: shared AbstractSymEntry): MsgTuple throws {
            const name = nextName(),
                  response = MsgTuple.newSymbol(name, symbol.borrow());
            tab.addOrReplace(name, symbol);
            symbol.setName(name);
            mtLogger.info(getModuleName(),getRoutineName(),getLineNumber(),response.msg);
            return response;
        }

        /*
            Takes args and creates a new SymEntry.

            :arg name: name of the array
            :type name: string

            :arg shape: length of array in each dimension
            :type shape: int

            :arg t: type of array

            :returns: borrow of newly created `SymEntry(t)`
        */
        proc addEntry(name: string, shape: int ...?N, type t): borrowed SymEntry(t, N) throws {
            if ! arrayDimIsSupported(N) then compilerWarning("arrays with rank ", N:string,
              " are not included in the server's configured ranks: ", arrayDimensionsStr);

            // check and throw if memory limit would be exceeded
            // TODO figure out a way to do memory checking for bigint
            if t != bigint {
                const len = * reduce shape;
                if t == bool {overMemLimit(len);} else {overMemLimit(len*numBytes(t));}
            }
            var entry = new shared SymEntry((...shape), t);
            if (tab.contains(name)) {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "redefined symbol: %s ".format(name));
            } else {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "adding symbol: %s ".format(name));
            }

            tab.addOrReplace(name, entry);
            entry.setName(name);

            return entry :borrowed :unmanaged :borrowed; // suppress lifetime checking
        }

        proc addEntry(name: string, shape: ?ND*int, type t): borrowed SymEntry(t, ND) throws
            do return addEntry(name, (...shape), t);

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
                                                        "redefined symbol: %s ".format(name));
            } else {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "adding symbol: %s ".format(name));            
            }

            tab.addOrReplace(name, entry);
            entry.setName(name);
            return tab[name];
        }

        /*
        Creates a symEntry from array name, length, and DType

        :arg name: name of the array
        :type name: string

        :arg shape: length of array in each dimension
        :type shape: int

        :arg dtype: type of array

        :returns: borrow of newly created GenSymEntry
        */
        proc addEntry(name: string, shape: int ...?ND, dtype: DType): borrowed AbstractSymEntry throws {
            for param idx in 0..arrayElementsTy.size-1 {
                type supportedType = arrayElementsTy[idx];
                if dtype == whichDtype(supportedType) then
                    return addEntry(name, (...shape), supportedType);
            }
            var errorMsg = "addEntry not implemented for %?".format(dtype);
            throw getErrorWithContext(
                msg=errorMsg,
                lineNumber=getLineNumber(),
                routineName=getRoutineName(),
                moduleName=getModuleName(),
                errorClass="IllegalArgumentError");
        }

        proc addEntry(name: string, shape: ?ND*int, dtype: DType): borrowed AbstractSymEntry throws
            do return addEntry(name, (...shape), dtype);

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
                                       "Deleting unregistered entry: %s".format(name));
                const removed = tab.getAndRemove(name);
                removed.removeDependents(this);
                return true;
            } else {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "Skipping registered entry: %s".format(name)); 
                return false;
            }
        }

        /*
        Clears all unregistered entries from the symTable
        */
        proc clear() throws {
            mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "Clearing all unregistered entries");

            // use keysToArray(): can't iterate over tab.keys() while deleting
            for n in tab.keysToArray() {
                // Deleting a SegStringSymEntry may have resulted in deletion
                // of its offset and value arrays. Check for that.
                if tab.contains(n) then
                    deleteEntry(n);
            }
        }

        /*
          Get a symbol from the table. Throw an error if the symbol is not found.
        */
        proc this(name: string): borrowed AbstractSymEntry throws {
            checkTable(name);
            return tab[name];
        }

        proc this(name: ParameterObj): borrowed AbstractSymEntry throws {
            return this[name.toScalar(string)];
        }

        /**
         * checks to see if a symbol is defined if it is not it throws an exception 
        */
        proc checkTable(name: string, calling_func="check") throws {
            if !tab.contains(name) {
                mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                                "undefined symbol: %s".format(name));
                throw getErrorWithContext(
                    msg=unknownSymbolError(pname=calling_func, sname=name),
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="UnknownSymbolError");
            } else {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                "found symbol: %s".format(name));
            }
        }

        /**
         * Prints the SymTable in a pretty format (name,SymTable[name])
         */
        proc pretty() throws {
            for n in tab {
                writeln("%10s = ".format(n), tab.getValue(n)); stdout.flush();
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
            return "[%s]".format(','.join(entries));
        }

        /*
        Convert JSON formmated list of entry names into a [] string object

        :arg names: JSON formatted list containing entry names
        :type names: string

        :returns: [] string of entry names
        */
        proc parseJson(names:string): [] string throws {
            var mem = openMemFile();
            mem.writer(locking=false).write(names);
            var reader = mem.reader(locking=false);

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
        proc getEntries(infoList:map(string, shared AbstractSymEntry)): [] string throws {
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
                // TODO: doesn't make sense to pass a string to 'formatJson' that is already in JSON format
                // (This will just add extra quotes around the string)
                return formatJson('{"name":%?, "dtype":%?, "size":%?, "ndim":%?, "shape":%s, "itemsize":%?, "registered":%?}',
                                  name, dtype2str(item.dtype), item.size, item.ndim, item.shape, item.itemsize, registry.contains(name));
            } else if abstractEntry.isAssignableTo(SymbolEntryType.SegStringSymEntry) {
                var item:borrowed SegStringSymEntry = toSegStringSymEntry(abstractEntry);
                return formatJson('{"name":%?, "dtype":%?, "size":%?, "ndim":%?, "shape":%s, "itemsize":%?, "registered":%?}',
                                  name, dtype2str(item.dtype), item.size, item.ndim, item.shape, item.itemsize, registry.contains(name));
            } else {
              return formatJson('{"name":%?, "dtype":%?, "size":%?, "ndim":%?, "shape":%s, "itemsize":%?, "registered":%?}',
                                name, dtype2str(DType.UNDEF), 0, 0, "[0]", 0, registry.contains(name));
            }
        }

        /*
        Returns raw attributes of the sym entry at the given string, if the string maps to an entry.
        Returns: name, dtype, size, ndim, shape, and item size

        :arg name: name of entry to be processed
        :type name: string

        :returns: s (string) containing info
        */
        // deprecated in favor of using st.insert(sym)
        proc attrib(name:string):string throws {
            checkTable(name, "attrib");

            var entry = tab[name];
            if entry.isAssignableTo(SymbolEntryType.TypedArraySymEntry){ //Anything considered a GenSymEntry
                var g:GenSymEntry = toGenSymEntry(entry);
                return "%s %s %? %? %s %?".format(name, dtype2str(g.dtype), g.size, g.ndim, g.shape, g.itemsize);
            }
            else if entry.isAssignableTo(SymbolEntryType.CompositeSymEntry) { //CompositeSymEntry
                var c: CompositeSymEntry = toCompositeSymEntry(entry);
                return "%s %? %?".format(name, c.size, c.ndim);
            }
            else if entry.isAssignableTo(SymbolEntryType.GeneratorSymEntry) {
                return name;
            }
            else if entry.isAssignableTo(SymbolEntryType.SparseSymEntry){
                var g: GenSparseSymEntry = toGenSparseSymEntry(entry);
                return name + " " + g.attrib();
            }

            throw new Error("attrib - Unsupported Entry Type %s".format(entry.entryType));
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
            return u.entry__str__(thresh=thresh, prefix="[", suffix="]", baseFormat="%?");
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
                return u.entry__str__(thresh=thresh, prefix="array([", suffix="])", baseFormat=frmt);
            } else {
                return "Unhandled type %s".format(entry.entryType);
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
            var rg = new regex(pattern);
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
        var abstractEntry = st[name];
        if ! abstractEntry.isAssignableTo(SymbolEntryType.TypedArraySymEntry) {
            var errorMsg = "Error: SymbolEntryType %s is not assignable to GenSymEntry".format(abstractEntry.entryType);
            mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw new Error(errorMsg);
        }
        return (abstractEntry: borrowed GenSymEntry);
    }

    proc getGenericTypedArrayEntry(name: ParameterObj, st: borrowed SymTab): borrowed GenSymEntry throws {
        return getGenericTypedArrayEntry(name.toScalar(string), st);
    }

    /**
     * Convenience proc to retrieve SegStringSymEntry from SymTab
     * Performs conversion from AbstractySymEntry to SegStringSymEntry
     * You can pass a logger from the calling function for better error reporting.
     */
    proc getSegStringEntry(name:string, st: borrowed SymTab): borrowed SegStringSymEntry throws {
        var abstractEntry = st[name];
        if ! abstractEntry.isAssignableTo(SymbolEntryType.SegStringSymEntry) {
            var errorMsg = "Error: SymbolEntryType %s is not assignable to SegStringSymEntry".format(abstractEntry.entryType);
            mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw new Error(errorMsg);
        }
        return (abstractEntry: borrowed SegStringSymEntry);
    }

    /**
     * Convenience proc to retrieve GenSparseSymEntry from SymTab
     * Performs conversion from AbstractSymEntry to GenSparseSymEntry
     * You can pass a logger from the calling function for better error reporting.
     */
    proc getGenericSparseArrayEntry(name:string, st: borrowed SymTab): borrowed GenSparseSymEntry throws {
        var abstractEntry = st[name];
        if ! abstractEntry.isAssignableTo(SymbolEntryType.SparseSymEntry) {
            var errorMsg = "Error: SymbolEntryType %s is not assignable to GenSparseSymEntry".format(abstractEntry.entryType);
            mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw new Error(errorMsg);
        }
        return (abstractEntry: borrowed GenSparseSymEntry);
    }
}
