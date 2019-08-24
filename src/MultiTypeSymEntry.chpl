
module MultiTypeSymEntry
{
    use ServerConfig;

    use NumPyDType;

    use SymArrayDmap;

    use AryUtil;
    
    /* Casts a GenSymEntry to the specified type and returns it.
       
       :arg gse: generic sym entry
       :type gse: borrowed GenSymEntry

        :arg etype: type for gse to be cast to
        :type etype: type
       */
    inline proc toSymEntry(gse: borrowed GenSymEntry, type etype) {
        return try! gse: borrowed SymEntry(etype);
    }

    /* This is a dummy class to avoid having to talk about specific
       instantiations of SymEntry. */
    class GenSymEntry
    {
        var dtype: DType; // answer to numpy dtype
        var itemsize: int; // answer to numpy itemsize = num bytes per elt
        var size: int = 0; // answer to numpy size == num elts
        var ndim: int = 1; // answer to numpy ndim == 1-axis for now
        var shape: 1*int = (0,); // answer to numpy shape == 1*int tuple
        
        // not sure yet how to implement numpy data() function

        proc init(type etype, len: int = 0) {
            this.dtype = whichDtype(etype);
	    this.itemsize = dtypeSize(this.dtype);
            this.size = len;
            this.shape = (len,);
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
            this.etype = etype;
            this.aD = D;
            this.a = a;
        }

        /*
        Verbose flag utility method
        */
        proc postinit() {
            if v {write("aD = "); printOwnership(this.a);}
        }
        /*
        Verbose flag utility method
        */
        proc deinit() {
            if v {writeln("deinit SymEntry");try! stdout.flush();}
        }
        
    }

}
