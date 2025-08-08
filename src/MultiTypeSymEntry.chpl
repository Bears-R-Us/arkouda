
module MultiTypeSymEntry
{
    use Random;
    use Reflection;
    use Set;

    use ServerConfig;
    use Logging;
    use AryUtil;

    use MultiTypeSymbolTable;
    public use NumPyDType;
    public use SymArrayDmap;

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

            GenSparseSymEntry,    // Generic entry for sparse matrices
                SparseSymEntry,    // Entry for sparse matrices
            GeneratorSymEntry,  // Entry for random number generators

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
         * When a SegStringSymEntry is removed from the ST, need to remove
         * its "offsets" and "values". Otherwise nothing to do.
         */
        proc removeDependents(st: borrowed SymTab) throws {
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
        proc entry__str__(thresh:int=1, prefix:string="", suffix:string="", baseFormat:string=""): string throws {
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
    inline proc toSymEntry(gse: borrowed GenSymEntry, type etype, param dimensions=1) throws {
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
        var shape: string = "[0]"; // answer to numpy shape

        /*
          Create a 1D GenSymEntry from an array element type and length
        */
        proc init(type etype, len: int = 0, ndim: int = 1) {
            this.entryType = SymbolEntryType.TypedArraySymEntry;
            assignableTypes.add(this.entryType);
            this.dtype = whichDtype(etype);
            this.itemsize = dtypeSize(this.dtype);
            this.size = len;
            this.ndim = ndim;
            init this;
            if len == 0 then
              this.shape = "[0]";
            else
              this.shape = tupShapeString(1, ndim);
        }


        // not sure yet how to implement numpy data() function

        override proc getSizeEstimate(): int {
            return this.size * this.itemsize;
        }

        /* Cast this `GenSymEntry` to `borrowed SymEntry(etype)`

           This function will halt if the cast fails.

           :arg etype: `SymEntry` type parameter
           :type etype: type
         */
        inline proc toSymEntry(type etype, param dimensions=1) throws {
            try {
                return this :borrowed SymEntry(etype, dimensions);
            } catch e {
                    const errorMsg = "Could not cast this `GenSymEntry` to `borrowed SymEntry(%s)".format(type2str(etype));
                    throw new Error(errorMsg);
            }
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
        override proc entry__str__(thresh:int=1, prefix:string="", suffix:string="", baseFormat:string=""): string throws {
            genLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "__str__ invoked");
            var s = "DType: %s, itemsize: %?, size: %?".format(this.dtype, this.itemsize, this.size);
            return prefix + s + suffix;
        }

        proc attrib(): string throws {
            return "%s %? %? %s %?".format(dtype2str(this.dtype), this.size, this.ndim, this.shape, this.itemsize);
        }
    }

    /* Symbol table entry */
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
        var a = makeDistArray((...tupShape), etype);
        /* Removed domain accessor, use `a.domain` instead */
        proc aD { compilerError("SymEntry.aD has been removed, use SymEntry.a.domain instead"); }
        /* only used with bigint pdarrays */
        var max_bits:int = -1;

        /*
          Create a SymEntry from a defaultRectangular array (when the server is
          configured to create distributed arrays)

          :arg a: array
          :type a: [] ?etype
        */
        proc init(a: [?D] ?etype, max_bits=-1) where MyDmap != Dmap.defaultRectangular && a.isDefaultRectangular() {
            this.init((...D.shape), etype);
            this.tupShape = D.shape;
            this.a = a;
            this.shape = tupShapeString(this.tupShape);
            this.ndim = D.rank;
            this.max_bits = max_bits;
        }

        /*
          Create a SymEntry from an array
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
            init this;
            this.shape = tupShapeString(this.tupShape);
            this.ndim = this.tupShape.size;
        }

        /*
          Create a SymEntry from a shape and element type

          :args len: size of each dimension
          :type len: int

          :arg etype: type to be instantiated
          :type etype: type
        */
        proc init(args: int ...?N, type etype) {
            var len = 1;
            for param i in 0..#N {
                len *= args[i];
            }
            super.init(etype, len, N);
            this.entryType = SymbolEntryType.PrimitiveTypedArraySymEntry;
            assignableTypes.add(this.entryType);

            this.etype = etype;
            this.dimensions = N;
            this.tupShape = args;
            this.a = try! makeDistArray((...args), etype);
            init this;
            this.shape = tupShapeString(this.tupShape);
            this.ndim = this.tupShape.size;
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
        override proc entry__str__(thresh:int=6, prefix:string = "[", suffix:string = "]", baseFormat:string = "%?"): string throws {
            const threshND = thresh / this.dimensions;
            proc subArrayStr(param dim: int, in idx: this.dimensions*int): string {
                const dimSize = this.tupShape[dim];
                var s = prefix;
                var first = true;
                if dim < this.dimensions-1 {
                    if dimSize <= threshND || dimSize <= 6 {
                        for i in 0..<dimSize {
                            if first then first = false; else s += " ";
                            idx[dim] = i;
                            s += subArrayStr(dim+1, idx);
                        }
                    } else {
                        for i in 0..<3 {
                            idx[dim] = i;
                            s += subArrayStr(dim+1, idx) + " ";
                        }
                        s += "... ";
                        for i in (dimSize - 3)..<dimSize {
                            if first then first = false; else s += " ";
                            idx[dim] = i;
                            s += subArrayStr(dim+1, idx);
                        }
                    }
                } else {
                    if dimSize <= thresh || dimSize <= 6 {
                        for i in 0..<dimSize {
                            if first then first = false; else s += " ";
                            idx[dim] = i;
                            s += try! baseFormat.format(this.a[idx]);
                        }
                    } else {
                        for i in 0..<3 {
                            idx[dim] = i;
                            s += try! baseFormat.format(this.a[idx]) + " ";
                        }
                        s += "... ";
                        for i in (dimSize - 3)..<dimSize {
                            if first then first = false; else s += " ";
                            idx[dim] = i;
                            s += try! baseFormat.format(this.a[idx]);
                        }
                    }
                }
                s += suffix;
                return s;
            }

            var s = subArrayStr(0, this.tupShape);
            if this.etype == bool {
                s = s.replace("true","True");
                s = s.replace("false","False");
            }

            return s;
        }
    }

    inline proc createSymEntry(shape: int ..., type etype) throws {
      var a = makeDistArray((...shape), etype);
      return new shared SymEntry(a);
    }

    inline proc createSymEntry(in a: [?D] ?etype, max_bits=-1) throws {
      var A = makeDistArray(a);
      return new shared SymEntry(A, max_bits=max_bits);
    }


    // override proc SymEntry.serialize(writer, ref serializer) throws {
    //   use Reflection;
    //   var f = writer;
    //   proc writeField(f, param i) throws {
    //     if !isArray(getField(this, i)) {
    //       f.write(getFieldName(this.type, i), " = ", getField(this, i):string);
    //     } else {
    //       f.write(getFieldName(this.type, i), " = ", formatAry(getField(this, i)));
    //     }
    //   }

    //   super.serialize(f);
    //   f.write(" {");
    //   param nFields = numFields(this.type);
    //   for param i in 0..nFields-2 {
    //     writeField(f, i);
    //     f.write(", ");
    //   }
    //   writeField(f, nFields-1);
    //   f.write("}");
    // }

    // implements writeSerializable(SymEntry);

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

        proc attrib(): string throws {
            return "%? %?".format(this.size, this.ndim);
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

        const offsetsEntry: shared SymEntry(int, 1);
        const bytesEntry: shared SymEntry(uint(8), 1);

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

        override proc removeDependents(st: borrowed SymTab) throws {
            if st.contains(offsetsEntry.name) then
                st.deleteEntry(offsetsEntry.name);
            if st.contains(bytesEntry.name) then
                st.deleteEntry(bytesEntry.name);
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
        override proc entry__str__(thresh:int=1, prefix:string="", suffix:string="", baseFormat:string=""): string throws {
            genLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "__str__ invoked");
            var s = "DType: %s, itemsize: %?, size: %?".format(this.dtype, this.itemsize, this.size);
            return prefix + s + suffix;
        }
    }

    class GenSparseSymEntry:AbstractSymEntry {
        var dtype: DType; // answer to numpy dtype
        var itemsize: int; // answer to numpy itemsize = num bytes per elt
        var size: int = 0; // answer to numpy size == num elts
        var nnz: int = 0;
        var ndim: int = 2; // answer to numpy ndim == 2-axis for now
        var shape: string = "[0,0]"; // answer to numpy shape
        var layoutStr: string = "UNKNOWN"; // How to initialize

        /*
          Create a 1D GenSymEntry from an array element type and length
        */
        proc init(type etype, size: int = 0, nnz: int, ndim: int = 2, layoutStr: string) {
            this.entryType = SymbolEntryType.SparseSymEntry;
            assignableTypes.add(this.entryType);
            this.dtype = whichDtype(etype);
            this.itemsize = dtypeSize(this.dtype);
            this.size = size;
            this.nnz = nnz;
            this.ndim = ndim;
            init this;
            if size == 0 then
              this.shape = "[0,0]";
            else
              this.shape = tupShapeString(1, ndim);
            this.layoutStr = layoutStr;
        }

        /* Cast this `SparseGenSymEntry` to `borrowed SparseSymEntry(etype)`

           This function will halt if the cast fails.

           :arg etype: `SparseSymEntry` type parameter
           :type etype: type
         */
        inline proc toSparseSymEntry(type etype, param dimensions=2, param layout) {
            return try! this :borrowed SparseSymEntry(etype, dimensions, layout);
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
        override proc entry__str__(thresh:int=1, prefix:string="", suffix:string="", baseFormat:string=""): string throws {
            genLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "__str__ invoked");
            var s = "DType: %s, itemsize: %?, size: %?, nnz: %?, layout: %s".format(this.dtype, this.itemsize, this.size, this.nnz, this.layoutStr);
            return prefix + s + suffix;
        }


        proc attrib(): string throws {
            return "%s %? %? %? %s %s %?".format(dtype2str(this.dtype), this.size, this.nnz, this.ndim, this.shape, this.layoutStr, this.itemsize);
        }

    }

    import SparseMatrix.Layout;

    proc layoutToStr(param l) param {
        select l {
            when Layout.CSR {
                return "CSR";
            }
            when Layout.CSC {
                return "CSC";
            }
            otherwise {
                return "UNKNOWN";
            }
        }
    }

    /* Symbol table entry */
    class SparseSymEntry : GenSparseSymEntry
    {
        /*
        generic element type array
        etype is different from dtype (chapel vs numpy)
        */
        type etype;

        /*
        number of dimensions, to be passed back to the `GenSparseSymEntry` so that
        we are able to make it visible to the Python client
        */
        param dimensions: int; // TODO: should we only support 2D sparse arrays and remove this field?

        /*
        the actual shape of the array, this has to live here, since GenSparseSymEntry
        has to stay generic
        For now, each dimension is assumed to be equal.
        */
        var tupShape: dimensions*int;

        /*
        layout of the sparse array: CSC or CSR
        */
        param matLayout : Layout;

        /*
        'a' is the distributed sparse array
        */
        // Hardcode 2D matrix for now (makeSparseArray accepts a 2-tuple shape)
        var a = makeSparseArray((...tupShape), etype, matLayout);

        /*
          Create a SparseSymEntry from a sparse array
        */
        proc init(a: [?D] ?eltType, param matLayout)
            where a.domain.parentDom.rank == 2 // Hardcode a 2D matrix for now
        {
            const size = D.shape[0] * D.shape[1];
            super.init(eltType, size, a.domain.getNNZ(), /*ndim*/2, layoutToStr(matLayout)); // Hardcode a 2D matrix for now
            this.entryType = SymbolEntryType.SparseSymEntry;
            assignableTypes.add(this.entryType);
            this.etype = eltType;
            this.dimensions = 2; // Hardcode a 2D matrix for now
            this.tupShape = D.shape;
            this.matLayout = matLayout;
            this.a = a;
            init this;
            this.shape = tupShapeString(this.tupShape);
            this.ndim = 2;
        }

        /*
        Formats and returns data in this entry up to the specified threshold.
        Matrices with nnz less than threshold will be printed in their entirety.
        Matrices with nnz greater than or equal to threshold will print the first 3 and last 3 elements

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
        override proc entry__str__(thresh:int=6, prefix:string = "noprefix", suffix:string = "nosuffix", baseFormat:string = "%?"): string throws {
            var s:string;
            const ref sparseDom = this.a.domain,
                        denseDom = sparseDom.parentDom;
            if this.a.domain.getNNZ() >= thresh {
                var count = 0;
                // Normal iteration like this is more efficient than
                // dense iteration, so we prefer that for the first elements
                for (_, (i, j)) in zip(1..3, sparseDom) {
                    const idxStr = "  (%?, %?)".format(i, j); // Padding to match SciPy
                    s += "%<16s%?\n".format(idxStr, this.a[i,j]);
                }

                s += "  :     :\n"; // Dot dot seperator, but vertical

                // For the last elements, we iterate in dense order
                // Since sparseArrays cant be strided by -1
                // We also have to do some i,j swaps for CSC vs CSR differences
                count = 0;
                var backString = "";
                for (i, j) in denseDom by -1 {
                    var row = i, col = j;
                    if this.matLayout==Layout.CSC {
                        row = j; // Iterate in Col Major Order for CSC
                        col = i; // To match SciPy behavior
                    }
                    if !sparseDom.contains(row, col) then continue;
                    const idxStr = "  (%?, %?)".format(row, col); // Padding to match SciPy
                    backString = "%<16s%?\n".format(idxStr, this.a[row,col]) + backString;
                    count += 1;
                    if count == 3 then break;
                }
                s+=backString;
            } else {
                for (i,j) in sparseDom {
                    const idxStr = "  (%?, %?)".format(i, j); // Padding to match SciPy
                    s += "%<16s%?\n".format(idxStr, this.a[i,j]);
                }
            }

            if this.etype == bool {
                s = s.replace("true","True");
                s = s.replace("false","False");
            }

            return s;
        }

        /*
        Verbose flag utility method
        */
        proc deinit() {
            if logLevel == LogLevel.DEBUG {writeln("deinit SparseSymEntry");try! stdout.flush();}
        }
    }


    class GeneratorSymEntry:AbstractSymEntry {
        type etype;
        var generator: randomStream(etype);
        var state: int;

        proc init(generator: randomStream(?etype), state: int = 1) {
            this.entryType = SymbolEntryType.GeneratorSymEntry;
            assignableTypes.add(this.entryType);
            this.etype = etype;
            this.generator = generator;
            this.state = state;
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
     * Helper proc to cast AbstractSymEntry to GenSparseSymEntry
     */
    proc toGenSparseSymEntry(entry: borrowed AbstractSymEntry) throws {
        return (entry: borrowed GenSparseSymEntry);
    }

    /**
     * Helper proc to cast AbstractSymEntry to GeneratorSymEntry
     */
    proc toGeneratorSymEntry(entry: borrowed AbstractSymEntry, type t) throws {
        return (entry: borrowed GeneratorSymEntry(t));
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
