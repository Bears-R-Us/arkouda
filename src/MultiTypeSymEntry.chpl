
module MultiTypeSymEntry
{
    use Reflection;
    use Set;

    use ServerConfig;
    use Logging;
    use AryUtil;

    public use NumPyDType;
    public use SymArrayDmap;

    private config const logLevel = ServerConfig.logLevel;
    const genLogger = new Logger(logLevel);

    /**
     * Internal Types we can use to build our Symbol type hierarchy.
     * We are making the types a little more concrete than using Strings
     */
    enum SymbolEntryType {
        AbstractSymEntry,  // Root Type from which all other types will inherit
        
            TypedArraySymEntry, // Parent type for Arrays with a dtype, legacy->GenSymEntry
                PrimitiveTypedArraySymEntry, // int, uint8, bool, etc.
                ComplexTypedArraySymEntry,   // DateTime, TimeDelta, IP Address, etc.
        
            CompositeSymEntry, // Parent type for things which are composites of arrays
                SegStringSymEntry,    // SegString composed of offset-int[], bytes->uint(8)
                CategoricalSymEntry,  // Categorical
                GroupBySymEntry,      // GroupBy

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
        proc init() {
            this.entryType = SymbolEntryType.AbstractSymEntry;
            this.assignableTypes = new set(SymbolEntryType);
            this.assignableTypes.add(this.entryType);
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
    inline proc toSymEntry(gse: borrowed GenSymEntry, type etype) {
        return gse.toSymEntry(etype);
    }

    /* This is a dummy class to avoid having to talk about specific
       instantiations of SymEntry. */
    class GenSymEntry:AbstractSymEntry
    {
        var dtype: DType; // answer to numpy dtype
        var itemsize: int; // answer to numpy itemsize = num bytes per elt
        var size: int = 0; // answer to numpy size == num elts
        var ndim: int = 1; // answer to numpy ndim == 1-axis for now
        var shape: 1*int = (0,); // answer to numpy shape == 1*int tuple
        
        // not sure yet how to implement numpy data() function

        proc init(type etype, len: int = 0) {
            this.entryType = SymbolEntryType.TypedArraySymEntry;
            assignableTypes.add(this.entryType);
            this.dtype = whichDtype(etype);
            this.itemsize = dtypeSize(this.dtype);
            this.size = len;
            this.shape = (len,);
        }

        override proc getSizeEstimate(): int {
            return this.size * this.itemsize;
        }

        /* Cast this `GenSymEntry` to `borrowed SymEntry(etype)`

           This function will halt if the cast fails.

           :arg etype: `SymEntry` type parameter
           :type etype: type
         */
        inline proc toSymEntry(type etype) {
            return try! this :borrowed SymEntry(etype);
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
            var s = "DType: %s, itemsize: %t, size: %t".format(this.dtype, this.itemsize, this.size);
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
        'aD' is the distributed domain for 'a' whose value and type
        are defined by makeDistDom() to support varying distributions
        */
        var aD: makeDistDom(size).type;
        var a: [aD] etype;
        
        /*
        This init takes length and element type

        :arg len: length of array to be allocated
        :type len: int

        :arg etype: type to be instantiated
        :type etype: type
        */
        proc init(len: int, type etype) {
            super.init(etype, len);
            this.entryType = SymbolEntryType.PrimitiveTypedArraySymEntry;
            assignableTypes.add(this.entryType);

            this.etype = etype;
            this.aD = makeDistDom(len);
            // this.a uses default initialization
        }

        /*This init takes an array of a type

        :arg a: array
        :type a: [] ?etype
        */
        proc init(a: [?D] ?etype) {
            super.init(etype, D.size);
            this.entryType = SymbolEntryType.PrimitiveTypedArraySymEntry;
            assignableTypes.add(this.entryType);

            this.etype = etype;
            this.aD = D;
            this.a = a;
        }

        /*
        Verbose flag utility method
        */
        proc postinit() {
            //if v {write("aD = "); printOwnership(this.a);}
        }
        /*
        Verbose flag utility method
        */
        proc deinit() {
            if logLevel == LogLevel.DEBUG {writeln("deinit SymEntry");try! stdout.flush();}
        }
        
        override proc writeThis(f) throws {
          use Reflection;
          proc writeField(f, param i) throws {
            if !isArray(getField(this, i)) {
              f <~> getFieldName(this.type, i) <~> " = " <~> getField(this, i):string;
            } else {
              f <~> getFieldName(this.type, i) <~> " = " <~> formatAry(getField(this, i));
            }
          }

          super.writeThis(f);
          f <~> " {";
          param nFields = numFields(this.type);
          for param i in 0..nFields-2 {
            writeField(f, i);
            f <~> ", ";
          }
          writeField(f, nFields-1);
          f <~> "}";
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
        override proc __str__(thresh:int=6, prefix:string = "[", suffix:string = "]", baseFormat:string = "%t"): string throws {
            var s:string = "";
            if (this.size == 0) {
                s =  ""; // Unnecessary, but left for clarity
            } else if (this.size < thresh || this.size <= 6) {
                for i in 0..(this.size-2) {s += try! baseFormat.format(this.a[i]) + " ";}
                s += try! baseFormat.format(this.a[this.size-1]);
            } else {
                var b = baseFormat + " " + baseFormat + " " + baseFormat + " ... " +
                            baseFormat + " " + baseFormat + " " + baseFormat;
                s = try! b.format(
                            this.a[0], this.a[1], this.a[2],
                            this.a[this.size-3], this.a[this.size-2], this.a[this.size-1]);
            }
            
            if (bool == this.etype) {
                s = s.replace("true","True");
                s = s.replace("false","False");
            }

            return prefix + s + suffix;
        }
    }

    /**
     * The base parent class for SymbolEntries that are composites
     * of other entry types.
     */
    class CompositeSymEntry:AbstractSymEntry {
        proc init() {
            super.init();
            this.entryType = SymbolEntryType.CompositeSymEntry;
            assignableTypes.add(this.entryType);
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
        return new shared SymEntry(len, t);
    }

    class SegStringSymEntry:CompositeSymEntry {
        type etype = string;

        var offsetsEntry: shared SymEntry(int);
        var bytesEntry: shared SymEntry(uint(8));

        var dtype: DType; // answer to numpy dtype
        var itemsize: int; // answer to numpy itemsize = num bytes per elt
        var size: int; // answer to numpy size == num elts
        var ndim: int = 1; // answer to numpy ndim == 1-axis for now
        var shape: 1*int = (0,); // answer to numpy shape == 1*int tuple

        proc init(offsetsSymEntry: shared SymEntry, bytesSymEntry: shared SymEntry, type etype) {
            super.init();
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
            var s = "DType: %s, itemsize: %t, size: %t".format(this.dtype, this.itemsize, this.size);
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
     * Helper proc to cast AbstrcatSymEntry to SegStringSymEntry
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
}
