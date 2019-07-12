
module MultiTypeSymEntry
{
    use ServerConfig;

    use CyclicDist;
    use BlockDist;
    
    /* in chapel the types int and real defalut to int(64) and real(64)
       also need other types like float32, int32, etc */
    enum DType {Int64, Float64, Bool, UNDEF}; 

    /* take a chapel type and returns the matching DType */
    proc whichDtype(type etype) param : DType {
      if (etype == int) {return DType.Int64;}
      if (etype == real) {return DType.Float64;}
      if (etype == bool) {return DType.Bool;}
      return DType.UNDEF; // undefined type
    }

    /* return the size in bytes of a DType */
    proc dtypeSize(dt: DType): int {
      if (dt == DType.Int64) { return 8; }
      if (dt == DType.Float64) { return 8; }
      if (dt == DType.Bool) { return 1; }
      return 0;
    }

    /* turns a dtype string in pythonland into a DType */
    proc str2dtype(dstr:string): DType {
        if dstr == "int64" {return DType.Int64;}
        if dstr == "float64" {return DType.Float64;}        
        if dstr == "bool" {return DType.Bool;}        
        return DType.UNDEF;
    }
    
    /* turns a DType into a dtype string in pythonland */
    proc dtype2str(dtype:DType): string {
        if dtype == DType.Int64 {return "int64";}
        if dtype == DType.Float64 {return "float64";}        
        if dtype == DType.Bool {return "bool";}        
        return "UNDEF";
    }

    /* uses the MyDmap config param in ServerConfig.chpl
    if MyDmap = 0 {return (type CyclicDom(1,int(64),false));}
    if MyDmap = 1 {return (type BlockDom(1,int(64),false,unmanaged DefaultDist));} */
    proc makeDistDom(size:int) {
        select MyDmap
        {
            when 0 { // Cyclic distribution
                return {0..#size} dmapped Cyclic(startIdx=0);
            }
            when 1 { // Block Distribution
                if size > 0 {
                    return {0..#size} dmapped Block(boundingBox={0..#size});
                }
                // fix the annoyance about boundingBox being enpty
                else {return {0..#0} dmapped Block(boundingBox={0..0});}
            }
            otherwise { // default to cyclic distribution
                return {0..#size} dmapped Cyclic(startIdx=0);
            }
        }
    }
    
    /* make an array of the dmapped dom and type I want */
    proc makeDistArray(size:int, type etype) {
        var a: [makeDistDom(size)] etype;
        return a;
    }

    /* retrun the type of the dist domain */
    proc makeDistDomType(size: int) type {
        return makeDistDom(size).type;
    }

    /* cast the symbol entry of the right type and return it
       blah too much type inference still for my taste */
    inline proc toSymEntry(gse: borrowed GenSymEntry, type etype) {
        return try! gse: borrowed SymEntry(etype);
    }

    /* 1.18 version print out localSubdomains */
    proc printOwnership(x) {
        for loc in Locales do
            on loc do
                write(x.localSubdomain(), " ");
        writeln();
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

    /* symbol table entry
       we will just do 1-d arrays for now */
    class SymEntry : GenSymEntry
    {
        // generic element type array
        // etype is different from dtype (chapel vs numpy)
        type etype;

        // 'aD' is the distributed domain for 'a' whose value and type
        // are defined by makeDistDom() to support varying distributions
        var aD: makeDistDom(size).type;
        var a: [aD] etype;
        
        // this one takes length and element type
        proc init(len: int, type etype) {
            super.init(etype, len);
            this.etype = etype;
            this.aD = makeDistDom(len);
            // this.a uses default initialization
        }

        // this one takes an array of a type
        proc init(a: [?D] ?etype) {
            super.init(etype, D.size);
            this.etype = etype;
            this.aD = D;
            this.a = a;
        }

        proc postinit() {
            if v {write("aD = "); printOwnership(this.a);}
        }
        proc deinit() {
            if v {writeln("deinit SymEntry");try! stdout.flush();}
        }
        
    }

}
