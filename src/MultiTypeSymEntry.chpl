
module MultiTypeSymEntry
{
    use Reflection;
    use Set;

    use ServerConfig;
    use Logging;
    use AryUtil;

    public use NumPyDType;
    public use SymArrayDmapCompat;
    use ArkoudaSymEntryCompat;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const genLogger = new Logger(logLevel, logChannel);

    /**
     * Internal Types we can use to build our Symbol type hierarchy.
     * We are making the types a little more concrete than using Strings
     */
    enum SymbolEntryType {
        AbstractSymEntry,  // Root Type from which all other types will inherit
        
            TypedArraySymEntry, // Parent type for Arrays with a dtype, legacy->GenSymEntry
                PrimitiveTypedArraySymEntry, // int, uint8, bool, etc.
                ComplexTypedArraySymEntry,   // DateTime, TimeDelta, IP Address, etc.
        
            GenSymEntry,
                SegStringSymEntry,    // SegString composed of offset-int[], bytes->uint(8)

            CompositeSymEntry,        // Entries that consist of multiple SymEntries of varying type

            AnythingSymEntry, // Placeholder to stick aritrary things in the map
            UnknownSymEntry,
            None
    }

    /**
     * This is the root of our SymbolTable Entry / Typing system.
     * All other SymEntry classes should inherit from this class
     * or one of its ancestors and ultimately everything should
     * be assignable/coercible to this class.
     * 
     * All subclasses should set & add their type to the `assignableTypes`
     * set so we can maintain & determine the type hierarchy.
     */
    class AbstractSymEntry {
        var entryType:SymbolEntryType;
        var assignableTypes:set(SymbolEntryType); // All subclasses should add their type to this set
        var name = ""; // used to track the symbol table name assigned to the entry
        proc init() {
            this.entryType = SymbolEntryType.AbstractSymEntry;
            this.assignableTypes = new set(SymbolEntryType);
            this.assignableTypes.add(this.entryType);
        }

        proc init(input) {
          this.entryType = input.entryType;
          this.assignableTypes = input.assignableTypes;
        }

        /*
            Sets the name of the entry when it is added to the Symbol Table
        */
        proc setName(name: string) {
            this.name = name;
        }

        /**
         * This can be used to help determine if a class can be
         * assigned / coerced / cast to another one in its hierarchy.
         */
        proc isAssignableTo(entryType:SymbolEntryType):bool {
            return assignableTypes.contains(entryType);
        }

        /**
         * This is a hook for the ServerConfig.overMemLimit procedure
         * All concrete classes should override this method
         */
        proc getSizeEstimate(): int {
            return 0;
        }

        /**
         * Formats and returns data in this entry up to the specified threshold. 
         * Arrays of size less than threshold will be printed in their entirety. 
         * Arrays of size greater than or equal to threshold will print the first 3 and last 3 elements
         *
         * :arg thresh: threshold for data to return
         * :type thresh: int
         *
         * :arg prefix: String to prepend to the front of the data string
         * :type prefix: string
         *
         * :arg suffix: String to append to the tail of the data string
         * :type suffix: string
         *
         * :arg baseFormat: String which represents the base format string for the data type
         * :type baseFormat: string
         *
         * :returns: s (string) containing the array data
         */
        proc __str__(thresh:int=1, prefix:string="", suffix:string="", baseFormat:string=""): string throws {
            genLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "__str__ invoked");
            var s = "AbstractSymEntry:subClasses should override this __str__ proc";
            return prefix + s + suffix;
        }
    }

    /* Casts a GenSymEntry to the specified type and returns it.
       
       :arg gse: generic sym entry
       :type gse: borrowed GenSymEntry

        :arg etype: type for gse to be cast to
        :type etype: type
       */
    inline proc toSymEntry(gse: borrowed GenSymEntry, type etype, param dimensions=1) {
        return gse.toSymEntry(etype, dimensions);
    }

    /* 
        This is a dummy class to avoid having to talk about specific
        instantiations of SymEntry. 
        GenSymEntries can contain multiple SymEntries, but they represent a singular object.
        For example, SegArray contains the offsets and values array, but only the values are 
        considered data.
    */
    class GenSymEntry:AbstractSymEntry
    {
        var dtype: DType; // answer to numpy dtype
        var itemsize: int; // answer to numpy itemsize = num bytes per elt
        var size: int = 0; // answer to numpy size == num elts
        var ndim: int = 1; // answer to numpy ndim == 1-axis for now
        var shape: string = "[1]"; // answer to numpy shape

        // not sure yet how to implement numpy data() function

        proc init(type etype, len: int = 0, ndim: int = 1) {
            this.entryType = SymbolEntryType.TypedArraySymEntry;
            assignableTypes.add(this.entryType);
            this.dtype = whichDtype(etype);
            this.itemsize = dtypeSize(this.dtype);
            this.size = len;
            this.ndim = ndim;
            this.complete();
            this.shape = tupShapeString(1, ndim);
        }

        override proc getSizeEstimate(): int {
            return this.size * this.itemsize;
        }

        /* Cast this `GenSymEntry` to `borrowed SymEntry(etype)`

           This function will halt if the cast fails.

           :arg etype: `SymEntry` type parameter
           :type etype: type
         */
        inline proc toSymEntry(type etype, param dimensions=1) {
            return try! this :borrowed SymEntry(etype, dimensions);
        }

        /* 
          Formats and returns data in this entry up to the specified threshold. 
          Arrays of size less than threshold will be printed in their entirety. 
          Arrays of size greater than or equal to threshold will print the first 3 and last 3 elements

            :arg thresh: threshold for data to return
            :type thresh: int

            :arg prefix: String to prepend to the front of the data string
            :type prefix: string

            :arg suffix: String to append to the tail of the data string
            :type suffix: string

            :arg baseFormat: String which represents the base format string for the data type
            :type baseFormat: string

            :returns: s (string) containing the array data
        */
        override proc __str__(thresh:int=1, prefix:string="", suffix:string="", baseFormat:string=""): string throws {
            genLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "__str__ invoked");
            var s = "DType: %s, itemsize: %?, size: %?".doFormat(this.dtype, this.itemsize, this.size);
            return prefix + s + suffix;
        }
    }

    /* Symbol table entry
       Only supports 1-d arrays for now */
    class SymEntry : GenSymEntry
    {
        /*
        generic element type array
        etype is different from dtype (chapel vs numpy)
        */
        type etype;

        /*
        number of dimensions, to be passed back to the `GenSymEntry` so that
        we are able to make it visible to the Python client
        */
        param dimensions: int;

        /*
        the actual shape of the array, this has to live here, since GenSymEntry
        has to stay generic
        */
        var tupShape: dimensions*int;

        /*
        'a' is the distributed array whose value and type are defined by
        makeDist{Dom,Array}() to support varying distributions
        */
        var a: [makeDistDom((...tupShape))] etype;
        /* Removed domain accessor, use `a.domain` instead */
        proc aD { compilerError("SymEntry.aD has been removed, use SymEntry.a.domain instead"); }
        /* only used with bigint pdarrays */
        var max_bits:int = -1;

        /*
        This init takes length and element type

        :arg len: length of array to be allocated
        :type len: int

        :arg etype: type to be instantiated
        :type etype: type
        */
        proc init(args: int ...?N, type etype) {
            var len = 1;
            for i in 0..#N {
              len *= args[i];
            }
            super.init(etype, len, N);
            this.entryType = SymbolEntryType.PrimitiveTypedArraySymEntry;
            assignableTypes.add(this.entryType);

            this.etype = etype;
            this.dimensions = N;
            this.tupShape = args;
            this.a = try! makeDistArray((...args), etype);
            this.complete();
            this.shape = tupShapeString(this.tupShape);
        }

        /*
        This init takes an array whose type matches `makeDistArray()`

        :arg a: array
        :type a: [] ?etype
        */
        proc init(in a: [?D] ?etype, max_bits=-1) {
            super.init(etype, D.size);
            this.entryType = SymbolEntryType.PrimitiveTypedArraySymEntry;
            assignableTypes.add(this.entryType);

            this.etype = etype;
            this.dimensions = D.rank;
            this.tupShape = D.shape;
            this.a = a;
            this.max_bits=max_bits;
            this.complete();
            this.shape = tupShapeString(this.tupShape);
        }

        /*
        This init takes an array whose type is defaultRectangular (convenience
        function for creating a distributed array from a non-distributed one)

        :arg a: array
        :type a: [] ?etype
        */
        proc init(a: [?D] ?etype) where MyDmap != Dmap.defaultRectangular && a.isDefaultRectangular() {
            this.init(D.size, etype, D.rank);
            this.tupShape = D.shape;
            this.a = a;
            this.shape = tupShapeString(this.tupShape);
        }

        /*
        Verbose flag utility method
        */
        proc deinit() {
            if logLevel == LogLevel.DEBUG {writeln("deinit SymEntry");try! stdout.flush();}
        }

        /*
        Formats and returns data in this entry up to the specified threshold. 
        Arrays of size less than threshold will be printed in their entirety. 
        Arrays of size greater than or equal to threshold will print the first 3 and last 3 elements

            :arg thresh: threshold for data to return
            :type thresh: int

            :arg prefix: String to pre-pend to the front of the data string
            :type prefix: string

            :arg suffix: String to append to the tail of the data string
            :type suffix: string

            :arg baseFormat: String which represents the base format string for the data type
            :type baseFormat: string

            :returns: s (string) containing the array data
        */
        override proc __str__(thresh:int=6, prefix:string = "[", suffix:string = "]", baseFormat:string = "%?"): string throws {
            proc dimSummary(dim: int): string throws {
                const dimSize = this.tupShape[dim];
                var s: string;
                if dimSize == 0 {
                    s = "";
                } else if dimSize < thresh || dimSize <= 6 {
                    var first = true,
                        idx: this.dimensions*int;

                    for i in 0..<dimSize {
                        if first then first = false; else s += " ";
                        idx[dim] = i;
                        s += baseFormat.doFormat(this.a[idx]);
                    }
                } else {
                    const fstring = baseFormat + " " + baseFormat + " " + baseFormat + " ... " +
                        baseFormat + " " + baseFormat + " " + baseFormat;

                    var indices: 6*(this.dimensions*int);
                    indices[0][dim] = 0;
                    indices[1][dim] = 1;
                    indices[2][dim] = 2;

                    for d in 0..<this.dimensions {
                        const dMax = this.tupShape[d]-1;
                        indices[3][d] = dMax;
                        indices[4][d] = dMax;
                        indices[5][d] = dMax;
                    }

                    indices[3][dim] = dimSize-3;
                    indices[4][dim] = dimSize-2;
                    indices[5][dim] = dimSize-1;

                    s = fstring.doFormat(this.a[indices[0]], this.a[indices[1]], this.a[indices[2]],
                                            this.a[indices[3]], this.a[indices[4]], this.a[indices[5]]);
                }

                if this.etype == bool {
                    s = s.replace("true","True");
                    s = s.replace("false","False");
                }
                return s;
            }

            var s = "",
                first = true;
            for d in 0..<this.dimensions {
                if first then first = false; else s += "\n";
                s += prefix + dimSummary(d) + suffix;
            }

            return s;
        }
    }
    
    inline proc createSymEntry(len: int, type etype) throws {
      var a = makeDistArray(len, etype);
      return new shared SymEntry(a);
    }

    inline proc createSymEntry(in a: [?D] ?etype, max_bits=-1) throws {
      var A = makeDistArray(a);
      return new shared SymEntry(A, max_bits=max_bits);
    }

    /*
        Base class for any entry that consists of multiple SymEntries that have varying types.
        These entries are related, but do not represent a single object.
        For Example, group by contains multiple SymEntries that are all considered part of the dataset.
    */
    class CompositeSymEntry:AbstractSymEntry {
        // This class is functionally equivalent to GenSymEntry, but used to denote
        // a Symbol Table Entry made up of multiple components.
        var ndim: int = 1; // answer to numpy ndim == 1-axis for now
        var size: int = 0; // answer to numpy size == num elts
        proc init(len: int = 0) {
            this.entryType = SymbolEntryType.CompositeSymEntry;
            assignableTypes.add(this.entryType);
            this.size = len;
        }

    }

    /**
     * Factory method for creating a typed SymEntry and checking mem limits
     * :arg len: the number of elements to allocate
     * :type len: int
     * 
     * :arg t: the element type
     * :type t: type
    */
    proc createTypedSymEntry(len: int, type t) throws {
        if t == bool {overMemLimit(len);} else {overMemLimit(len*numBytes(t));}
        return createSymEntry(len,t);
    }

    class SegStringSymEntry:GenSymEntry {
        type etype = string;

        var offsetsEntry: shared SymEntry(int, 1);
        var bytesEntry: shared SymEntry(uint(8), 1);

        proc init(offsetsSymEntry: shared SymEntry(int), bytesSymEntry: shared SymEntry(uint(8)), type etype) {
            super.init(etype, bytesSymEntry.size);
            this.entryType = SymbolEntryType.SegStringSymEntry;
            assignableTypes.add(this.entryType);

            this.offsetsEntry = offsetsSymEntry;
            this.bytesEntry = bytesSymEntry;

            this.dtype = whichDtype(etype);
            this.itemsize = this.bytesEntry.itemsize;
            this.size = this.offsetsEntry.size;
            this.ndim = this.offsetsEntry.ndim;
            this.shape = this.offsetsEntry.shape;
        }

        override proc getSizeEstimate(): int {
            return this.offsetsEntry.getSizeEstimate() + this.bytesEntry.getSizeEstimate();
        }

        /**
         * Formats and returns data in this entry up to the specified threshold. 
         * Arrays of size less than threshold will be printed in their entirety. 
         * Arrays of size greater than or equal to threshold will print the first 3 and last 3 elements
         *
         * :arg thresh: threshold for data to return
         * :type thresh: int
         *
         * :arg prefix: String to prepend to the front of the data string
         * :type prefix: string
         *
         * :arg suffix: String to append to the tail of the data string
         * :type suffix: string
         *
         * :arg baseFormat: String which represents the base format string for the data type
         * :type baseFormat: string
         *
         * :returns: s (string) containing the array data
         */
        override proc __str__(thresh:int=1, prefix:string="", suffix:string="", baseFormat:string=""): string throws {
            genLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "__str__ invoked");
            var s = "DType: %s, itemsize: %?, size: %?".doFormat(this.dtype, this.itemsize, this.size);
            return prefix + s + suffix;
        }
    }

    /**
     * Helper proc to cast AbstrcatSymEntry to GenSymEntry
     */
    proc toGenSymEntry(entry: borrowed AbstractSymEntry) throws {
        return (entry: borrowed GenSymEntry);
    }

    /**
     * Helper proc to cast AbstractSymEntry to CompositeSymEntry
     */
    proc toCompositeSymEntry(entry: borrowed AbstractSymEntry) throws {
        return (entry: borrowed CompositeSymEntry);
    }
    
    /**
     * Helper proc to cast AbstractSymEntry to SegStringSymEntry
     */
    proc toSegStringSymEntry(entry: borrowed AbstractSymEntry) throws {
        return (entry: borrowed SegStringSymEntry);
    }

    /**
     * Temporary shim to ease transition to Typed Symbol Table Entries.
     * This attempts to retrieve the Dtype, size/array-length, and itemsize from a SymbolTable
     * entry if the entry type supports it. Returns default tuple of valuse otherwise.
     * 
     * :arg entry: AbstractSymEntry or descendant
     * :type entry: borrowed AbstractSymEntry
     * :retruns: tuple of (dtype, entry.size, entry.itemsize)
     * 
     * Note: entry.size is generally the number of elements in the array
     *       and is more synonymous with length
     */
    proc getArraySpecFromEntry(entry: borrowed AbstractSymEntry) throws {
        if entry.isAssignableTo(SymbolEntryType.TypedArraySymEntry) {
            var g: borrowed GenSymEntry = entry:borrowed GenSymEntry;
            return (g.dtype, g.size, g.itemsize);
        } else if entry.isAssignableTo(SymbolEntryType.SegStringSymEntry) {
            var g: borrowed SegStringSymEntry = entry:borrowed SegStringSymEntry;
            return (g.dtype, g.size, g.itemsize);
        } else {
            return (DType.UNDEF, -1, -1);
        }
    }

    /*
      Create a string to represent a JSON tuple of an array's shape
    */
    proc tupShapeString(shape): string {
        var s = "[",
            first = true;
        for x in shape {
            if first then first = false; else s += ",";
            s += x:string;
        }
        s += "]";
        return s;
    }

    proc tupShapeString(val: int, ndim: int): string {
        return tupShapeString([i in 1..ndim] val);
    }

}
