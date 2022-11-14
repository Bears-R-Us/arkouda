
module MultiTypeSymEntry
{
    use Reflection;
    use Set;

    use ServerConfig;
    use Logging;
    use AryUtil;

    public use NumPyDType;
    public use SymArrayDmap;
    use MultiTypeSymbolTable;

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
        
            GenSymEntry,
                SegStringSymEntry,    // SegString composed of offset-int[], bytes->uint(8)
                CategoricalSymEntry,  // Categorical
                SegArraySymEntry,     // Segmented Array

            CompositeSymEntry,        // Entries that consist of multiple SymEntries of varying type
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
        var name = ""; // used to track the symbol table name assigned to the entry
        proc init() {
            this.entryType = SymbolEntryType.AbstractSymEntry;
            this.assignableTypes = new set(SymbolEntryType);
            this.assignableTypes.add(this.entryType);
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
    inline proc toSymEntry(gse: borrowed GenSymEntry, type etype) {
        return gse.toSymEntry(etype);
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
        'a' is the distributed array whose value and type are defined by
        makeDist{Dom,Array}() to support varying distributions
        */
        var a = makeDistArray(size, etype);
        /* Removed domain accessor, use `a.domain` instead */
        proc aD { compilerError("SymEntry.aD has been removed, use SymEntry.a.domain instead"); }
        
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
            this.a = makeDistArray(size, etype);
        }

        /*
        This init takes an array whose type matches `makeDistArray()`

        :arg a: array
        :type a: [] ?etype
        */
        proc init(in a: [?D] ?etype) {
            super.init(etype, D.size);
            this.entryType = SymbolEntryType.PrimitiveTypedArraySymEntry;
            assignableTypes.add(this.entryType);

            this.etype = etype;
            this.a = a;
        }

        /*
        This init takes an array whose type is defaultRectangular (convenience
        function for creating a distributed array from a non-distributed one)

        :arg a: array
        :type a: [] ?etype
        */
        proc init(a: [?D] ?etype) where MyDmap != Dmap.defaultRectangular && a.isDefaultRectangular() {
            this.init(D.size, etype);
            this.a = a;
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
              f.write(getFieldName(this.type, i), " = ", getField(this, i):string);
            } else {
              f.write(getFieldName(this.type, i), " = ", formatAry(getField(this, i)));
            }
          }

          super.writeThis(f);
          f.write(" {");
          param nFields = numFields(this.type);
          for param i in 0..nFields-2 {
            writeField(f, i);
            f.write(", ");
          }
          writeField(f, nFields-1);
          f.write("}");
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
        return new shared SymEntry(len, t);
    }

    class SegStringSymEntry:GenSymEntry {
        type etype = string;

        var offsetsEntry: shared SymEntry(int);
        var bytesEntry: shared SymEntry(uint(8));

        proc init(offsetsSymEntry: shared SymEntry, bytesSymEntry: shared SymEntry, type etype) {
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
            var s = "DType: %s, itemsize: %t, size: %t".format(this.dtype, this.itemsize, this.size);
            return prefix + s + suffix;
        }
    }

    class SegArraySymEntry:GenSymEntry {
        type etype;

        var segmentsEntry: shared SymEntry(int);
        var valuesEntry: shared SymEntry(etype);
        var lengthsEntry: shared SymEntry(int);

        proc init(segmentsSymEntry: shared SymEntry, valuesSymEntry: shared SymEntry, type etype) {
            super.init(etype, valuesSymEntry.size);
            this.entryType = SymbolEntryType.SegArraySymEntry;
            assignableTypes.add(this.entryType);
            this.etype = etype;
            this.segmentsEntry = segmentsSymEntry;
            this.valuesEntry = valuesSymEntry;

            ref sa = segmentsSymEntry.a;
            const high = segmentsSymEntry.a.domain.high;
            var lengths = [(i, s) in zip (segmentsSymEntry.a.domain, sa)] if i == high then valuesSymEntry.size - s else sa[i+1] - s;
            
            lengthsEntry = new shared SymEntry(lengths);

            this.dtype = whichDtype(etype);
            this.itemsize = this.valuesEntry.itemsize;
            this.size = this.segmentsEntry.size;
            this.ndim = this.segmentsEntry.ndim;
            this.shape = this.segmentsEntry.shape;
        }

        override proc getSizeEstimate(): int {
            return this.segmentsEntry.getSizeEstimate() + this.valuesEntry.getSizeEstimate();
        }
    }

    /*
        Symbol Table entry representing a GroupBy object.
    */
    class GroupBySymEntry:CompositeSymEntry {

        var keyNamesEntry: shared SymEntry(string);
        var keyTypesEntry: shared SymEntry(string);
        var segmentsEntry: shared SymEntry(int);
        var permEntry: shared SymEntry(int);
        var ukIndEntry: shared SymEntry(int);
        
        proc init(keyNamesEntry: shared SymEntry, keyTypesEntry: shared SymEntry, segmentsSymEntry: shared SymEntry, 
                    permSymEntry: shared SymEntry, ukIndSymEntry: shared SymEntry, itemsize: int) {
            super.init(permSymEntry.size); // sets this.size = permEntry.size
            this.entryType = SymbolEntryType.GroupBySymEntry;
            assignableTypes.add(this.entryType);
            this.keyNamesEntry = keyNamesEntry;
            this.keyTypesEntry = keyTypesEntry;
            this.segmentsEntry = segmentsSymEntry;
            this.permEntry = permSymEntry;
            this.ukIndEntry = ukIndSymEntry;

            this.ndim = this.segmentsEntry.size; // used as the number of groups
        }

        override proc getSizeEstimate(): int {
            return this.keyNamesEntry.getSizeEstimate() + this.keyTypesEntry.getSizeEstimate() + 
            this.segmentsEntry.getSizeEstimate() + this.permEntry.getSizeEstimate() + this.ukIndEntry.getSizeEstimate();
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
}
