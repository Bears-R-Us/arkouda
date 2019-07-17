
module MultiTypeSymbolTable
{
    use ServerConfig;
    use ServerErrorStrings;
    
    use MultiTypeSymEntry;
    use Chapel118;

    // symbol table
    class SymTab
    {
        /*
        Assocaitive domain of strings
        */
        var tD: domain(string);

        /*
        Associative array indexed by strings
        */
        var tab: [tD] shared GenSymEntry; 

        var nid = 0;
        /*
        Gives out symbol names.
        */
        proc nextName():string {
            nid += 1;
            return "id_"+ nid:string;
        }

        // is it an error to redefine an entry? ... probably not
        // this addEntry takes stuff to create a new SymEntry

        /*
        Takes args and creates a new SymEntry.

        :arg name: name of the array
        :type name: string

        :arg len: length of array
        :type len: int

        :arg t: type of array
        */
        proc addEntry(name: string, len: int, type t) {
            var entry = new shared SymEntry(len, t);
            if (tD.contains(name))
            {
                if (v) {writeln("redefined symbol ",name);try! stdout.flush();}
            }
            else
                tD += name;
            
            tab[name] = entry;
        }

        /*
        Takes an already created GenSymEntry and creates a new SymEntry.

        :arg name: name of the array
        :type name: string

        :arg entry: Generic Sym Entry to convert
        :type entry: GenSymEntry
        */
        proc addEntry(name: string, entry: shared GenSymEntry) {
            if (tD.contains(name))
            {
                if (v) {writeln("redefined symbol ",name);try! stdout.flush();}
            }
            else
                tD += name;
            
            tab[name] = entry;
        }

        /*
        Creates a symEntry from array name, length, and DType

        :arg name: name of the array
        :type name: string

        :arg len: length of array
        :type len: int

        :arg dtype: type of array
        */
        proc addEntry(name: string, len: int, dtype: DType) {
            if dtype == DType.Int64 {addEntry(name,len,int);}
            else if dtype == DType.Float64 {addEntry(name,len,real);}
            else if dtype == DType.Bool {addEntry(name,len,bool);}
            else {writeln("should not get here!");try! stdout.flush();}
        }

        /*
        Removes an entry from the symTable

        :arg name: name of the array
        :type name: string
        */
        proc deleteEntry(name: string) {
            if (tD.contains(name))
            {
                tab[name] = nil;
                tD -= name;
            }
            else
                if (v) {writeln("unkown symbol ",name);try! stdout.flush();}
        }
        
        /*
        Returns the sym entry associated with the provided name, if the sym entry exists
        :arg name: string to index/query in the sym table
        :type name: string

        :returns: sym entry or nil
        */
        proc lookup(name: string): shared GenSymEntry {
            if (!tD.contains(name))
            {
                if (v) {writeln("undefined symbol ",name);try! stdout.flush();}
                // what does the def init return ??? nil?
                return nil; // undefined!
            }
            else
            {
                return tab[name];
            }
        }

        proc pretty(){
            for n in tD
            {
                try! writeln("%10s = ".format(n), tab[n]);try! stdout.flush();
            }
        }
        
        /*
        Attempts to format and return sym entries mapped to the provided string into JSON format.
        Pass __AllSymbols__ to process the entire sym table.

        :arg name: name of entry to be processed
        :type name: string
        */
        proc dump(name:string): string {
            if name == "__AllSymbols__" {return try! "%jt".format(this);}
            else if (tD.contains(name)) {return try! "%jt %jt".format(name, tab[name]);}
            else {return try! "Error: dump: undefined name: %s".format(name);}
        }
        
        /*
        Returns verbose attributes of the sym entry at the given string, if the string maps to an entry.
        Pass __AllSymbols__ to process all sym entries in the sym table.

        Returns: name, dtype, size, ndim, shape, and item size

        :arg name: name of entry to be processed
        :type name: string

        :returns: s (string) containing info
        */
        proc info(name:string): string {
            var s: string;
            if name == "__AllSymbols__" {
                for n in tD {
                    if (tab[n] != nil) {
                        try! s += "name:%t dtype:%t size:%t ndim:%t shape:%t itemsize:%t\n".format(n, dtype2str(tab[n].dtype), tab[n].size, tab[n].ndim, tab[n].shape, tab[n].itemsize);
                    }
                }
            }
            else
            {
                if (tD.contains(name)) {
                    try! s = "name:%t dtype:%t size:%t ndim:%t shape:%t itemsize:%t\n".format(name, dtype2str(tab[name].dtype), tab[name].size, tab[name].ndim, tab[name].shape, tab[name].itemsize);
                }
                else {s = unknownSymbolError("info",name);}
            }
            return s;
        }

        /*
        Returns raw attributes of the sym entry at the given string, if the string maps to an entry.
        Returns: name, dtype, size, ndim, shape, and item size

        :arg name: name of entry to be processed
        :type name: string

        :returns: s (string) containing info
        */
        proc attrib(name:string):string {
            var s:string;
            if (tD.contains(name)) {
                try! s = "%s %s %t %t %t %t".format(name, dtype2str(tab[name].dtype), tab[name].size, tab[name].ndim, tab[name].shape, tab[name].itemsize);
            }
            else {s = unknownSymbolError("attrib",name);}
            return s;
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
        proc datastr(name: string, thresh:int): string {
            var s:string;
            if (tD.contains(name)) {
                var u: borrowed GenSymEntry = tab[name];
                select u.dtype
                {
                    when DType.Int64
                    {
                        var e = toSymEntry(u,int);
                        if e.size == 0 {s =  "[]";}
                        else if e.size < thresh || e.size <= 6 {
                            s =  "[";
                            for i in 0..(e.size-2) {s += try! "%t ".format(e.a[i]);}
                            s += try! "%t]".format(e.a[e.size-1]);
                        }
                        else {
                            s = try! "[%t %t %t ... %t %t %t]".format(e.a[0],e.a[1],e.a[2],
                                                                      e.a[e.size-3],
                                                                      e.a[e.size-2],
                                                                      e.a[e.size-1]);
                        }
                    }
                    when DType.Float64
                    {
                        var e = toSymEntry(u,real);
                        if e.size == 0 {s =  "[]";}
                        else if e.size < thresh || e.size <= 6 {
                            s =  "[";
                            for i in 0..(e.size-2) {s += try! "%t ".format(e.a[i]);}
                            s += try! "%t]".format(e.a[e.size-1]);
                        }
                        else {
                            s = try! "[%t %t %t ... %t %t %t]".format(e.a[0],e.a[1],e.a[2],
                                                                      e.a[e.size-3],
                                                                      e.a[e.size-2],
                                                                      e.a[e.size-1]);
                        }
                    }
                    when DType.Bool
                    {
                        var e = toSymEntry(u,bool);
                        if e.size == 0 {s =  "[]";}
                        else if e.size < thresh || e.size <= 6 {
                            s =  "[";
                            for i in 0..(e.size-2) {s += try! "%t ".format(e.a[i]);}
                            s += try! "%t]".format(e.a[e.size-1]);
                        }
                        else {
                            s = try! "[%t %t %t ... %t %t %t]".format(e.a[0],e.a[1],e.a[2],
                                                                      e.a[e.size-3],
                                                                      e.a[e.size-2],
                                                                      e.a[e.size-1]);
                        }
                        s = s.replace("true","True");
                        s = s.replace("false","False");
                    }
                    otherwise {s = unrecognizedTypeError("datastr",dtype2str(u.dtype));}
                }
            }
            else {s = unknownSymbolError("datastr",name);}
            return s;
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
        proc datarepr(name: string, thresh:int): string {
            var s:string;
            if (tD.contains(name)) {
                var u: borrowed GenSymEntry = tab[name];
                select u.dtype
                {
                    when DType.Int64
                    {
                        var e = toSymEntry(u,int);
                        if e.size == 0 {s =  "array([])";}
                        else if e.size < thresh || e.size <= 6 {
                            s =  "array([";
                            for i in 0..(e.size-2) {s += try! "%t, ".format(e.a[i]);}
                            s += try! "%t])".format(e.a[e.size-1]);
                        }
                        else {
                            s = try! "array([%t, %t, %t, ..., %t, %t, %t])".format(e.a[0],e.a[1],e.a[2],
                                                                                    e.a[e.size-3],
                                                                                    e.a[e.size-2],
                                                                                    e.a[e.size-1]);
                        }
                    }
                    when DType.Float64
                    {
                        var e = toSymEntry(u,real);
                        if e.size == 0 {s =  "array([])";}
                        else if e.size < thresh || e.size <= 6 {
                            s =  "array([";
                            for i in 0..(e.size-2) {s += try! "%.17r, ".format(e.a[i]);}
                            s += try! "%.17r])".format(e.a[e.size-1]);
                        }
                        else {
                            s = try! "array([%.17r, %.17r, %.17r, ..., %.17r, %.17r, %.17r])".format(e.a[0],e.a[1],e.a[2],
                                                                                    e.a[e.size-3],
                                                                                    e.a[e.size-2],
                                                                                    e.a[e.size-1]);
                        }
                    }
                    when DType.Bool
                    {
                        var e = toSymEntry(u,bool);
                        if e.size == 0 {s =  "array([])";}
                        else if e.size < thresh || e.size <= 6 {
                            s =  "array([";
                            for i in 0..(e.size-2) {s += try! "%t, ".format(e.a[i]);}
                            s += try! "%t])".format(e.a[e.size-1]);
                        }
                        else {
                            s = try! "array([%t, %t, %t, ..., %t, %t, %t])".format(e.a[0],e.a[1],e.a[2],
                                                                                    e.a[e.size-3],
                                                                                    e.a[e.size-2],
                                                                                    e.a[e.size-1]);
                        }
                        s = s.replace("true","True");
                        s = s.replace("false","False");
                    }
                    otherwise {s = unrecognizedTypeError("datarepr",dtype2str(u.dtype));}
                }
            }
            else {s = unknownSymbolError("datarepr",name);}
            return s;
        }
    }      
}

