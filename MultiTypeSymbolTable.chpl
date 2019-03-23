
module MultiTypeSymbolTable
{
    use ServerConfig;
    use ServerErrorStrings;
    
    use MultiTypeSymEntry;
    use Chapel118;

    // symbol table
    class SymTab
    {
        var tD: domain(string); // assoc domain of string
        var tab: [tD] shared GenSymEntry; // assoc array indexed by string

        // give out symbol names
        var nid = 0;
        proc next_name():string {
            nid += 1;
            return "id_"+ nid:string;
        }

        // is it an error to redefine an entry? ... probably not
        // this addEntry takes stuff to create a new SymEntry
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

        // this addEntry takes an already created GenSymEntry
        proc addEntry(name: string, entry: shared GenSymEntry) {
            if (tD.contains(name))
            {
                if (v) {writeln("redefined symbol ",name);try! stdout.flush();}
            }
            else
                tD += name;
            
            tab[name] = entry;
        }

        proc addEntry(name: string, len: int, dtype: DType) {
            if dtype == DType.Int64 {addEntry(name,len,int);}
            else if dtype == DType.Float64 {addEntry(name,len,real);}
            else if dtype == DType.Bool {addEntry(name,len,bool);}
            else {writeln("should not get here!");try! stdout.flush();}
        }

        proc deleteEntry(name: string) {
            if (tD.contains(name))
            {
                tab[name] = nil;
                tD -= name;
            }
            else
                if (v) {writeln("unkown symbol ",name);try! stdout.flush();}
        }
        
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
        
        proc dump(name:string): string {
            if name == "__AllSymbols__" {return try! "%jt".format(this);}
            else if (tD.contains(name)) {return try! "%jt %jt".format(name, tab[name]);}
            else {return try! "Error: dump: undefined name: %s".format(name);}
        }
        
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

        proc attrib(name:string):string {
            var s:string;
            if (tD.contains(name)) {
                try! s = "%s %s %t %t %t %t".format(name, dtype2str(tab[name].dtype), tab[name].size, tab[name].ndim, tab[name].shape, tab[name].itemsize);
            }
            else {s = unknownSymbolError("attrib",name);}
            return s;
        }

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
                        else if e.size < thresh {
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
                        else if e.size < thresh {
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
                        else if e.size < thresh {
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
                        else if e.size < thresh {
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
                        else if e.size < thresh {
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
                        else if e.size < thresh {
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

