
module MultiTypeSymbolTable
{
    use ServerConfig;
    use ServerErrorStrings;
    use Reflection;
    use Errors;
    use Logging;
    
    use MultiTypeSymEntry;
    use Map;
    use RadixSortLSD only radixSortLSD_ranks;
    use RandArray only fillInt;


    
    var FilteringPattern=0:int;
    var mtLogger = new Logger();
    if v {
        mtLogger.level = LogLevel.DEBUG;
    } else {
        mtLogger.level = LogLevel.INFO;    
    }

    /* symbol table */
    class SymTab
    {
        /*
        Associative domain of strings
        */
        var registry: domain(string);

        /*
        Map indexed by strings
        */
        var tab: map(string, shared GenSymEntry);

        var nid = 0;
        /*
        Gives out symbol names.
        */
        proc nextName():string {
            nid += 1;
            return "id_"+ nid:string;
        }

        proc regName(name: string, userDefinedName: string) throws {

            // check to see if name is defined
            if (!tab.contains(name)) {
                mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                     "regName: undefined symbol: %s".format(name));
                throw getErrorWithContext(
                                   msg=unknownSymbolError("regName", name),
                                   lineNumber=getLineNumber(),
                                   routineName=getRoutineName(),
                                   moduleName=getModuleName(),
                                   errorClass="UnknownSymbolError");
            }

            // check to see if userDefinedName is defined
            if (registry.contains(userDefinedName)) {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                     "regName: redefined symbol: %s ".format(userDefinedName));
            } else {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                     "Registering symbol: %s ".format(userDefinedName));            
            }
            
            registry += userDefinedName; // add user defined name to registry

            // point at same shared table entry
            tab.addOrSet(userDefinedName, tab.getValue(name));
        }

        proc unregName(name: string) throws {
            
            // check to see if name is defined
            if !tab.contains(name)  {
                mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                             "unregName: undefined symbol: %s ".format(name));
                throw getErrorWithContext(
                                   msg=unknownSymbolError("regName", name),
                                   lineNumber=getLineNumber(),
                                   routineName=getRoutineName(),
                                   moduleName=getModuleName(),
                                   errorClass="UnknownSymbolError");
            } else {
                if registry.contains(name) {
                    mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                             "Unregistering symbol: %s ".format(name));  
                } else {
                    mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                             "The symbol %s is not registered".format(name));                  
                }          
            }

            registry -= name; // take name out of registry
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

        :returns: borrow of newly created `SymEntry(t)`
        */
        proc addEntry(name: string, len: int, type t): borrowed SymEntry(t) throws {
            // check and throw if memory limit would be exceeded
            if t == bool {overMemLimit(len);} else {overMemLimit(len*numBytes(t));}
            
            var entry = new shared SymEntry(len, t);
            if (tab.contains(name)) {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "redefined symbol: %s ".format(name));
            } else {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "adding symbol: %s ".format(name));            
            }

            tab.addOrSet(name, entry);
            return tab.getBorrowed(name).toSymEntry(t);
        }

        /*
        Takes an already created GenSymEntry and creates a new SymEntry.

        :arg name: name of the array
        :type name: string

        :arg entry: Generic Sym Entry to convert
        :type entry: GenSymEntry

        :returns: borrow of newly created GenSymEntry
        */
        proc addEntry(name: string, in entry: shared GenSymEntry): borrowed GenSymEntry throws {
            // check and throw if memory limit would be exceeded
            overMemLimit(entry.size*entry.itemsize);

            if (tab.contains(name)) {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "redefined symbol: %s ".format(name));
            } else {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "adding symbol: %s ".format(name));            
            }

            tab.addOrSet(name, entry);
            return tab.getBorrowed(name);
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
        proc addEntry(name: string, len: int, dtype: DType): borrowed GenSymEntry throws {
            select dtype {
                when DType.Int64 { return addEntry(name, len, int); }
                when DType.Float64 { return addEntry(name, len, real); }
                when DType.Bool { return addEntry(name, len, bool); }
                otherwise { 
                    var errorMsg = "addEntry not implemented for %t".format(dtype); 
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
        */
        proc deleteEntry(name: string) throws {
            if tab.contains(name) { 
               if !registry.contains(name) {
                   mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "Deleting unregistered entry: %s".format(name)); 
                   tab.remove(name);
               } else {
                    mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "Skipping registered entry: %s".format(name)); 
               }            
            } else {
                if registry.contains(name) {
                    mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                     "Registered entry is not in SymTab: %s".format(name));
                } else {
                    mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                     "Unregistered entry is not in SymTab: %s".format(name));    
                }           
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

        
        /*
        Returns the sym entry associated with the provided name, if the sym entry exists

        :arg name: string to index/query in the sym table
        :type name: string

        :returns: sym entry or throws on error
        :throws: `unkownSymbolError(name)`
        */
        proc lookup(name: string): borrowed GenSymEntry throws {
            if (!tab.contains(name)) {
                throw getErrorWithContext(
                    msg=unknownSymbolError(pname="lookup", sname=name),
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="UnknownSymbolError");
            } else {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                "found symbol: %s".format(name));
                return tab.getBorrowed(name);
            }
        }

        /*
        checks to see if a symbol is defined if it is not it throws an exception 
        */
        proc check(name: string) throws { 
            if (!tab.contains(name)) { 
                mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                                "undefined symbol: %s".format(name));
                throw getErrorWithContext(
                    msg=unknownSymbolError(pname="check", sname=name),
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="UnknownSymbolError");
            } else {
                mtLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                "found symbol: %s".format(name));
            }
        }
        
        /*
        Prints the SymTable in a pretty format (name,SymTable[name])
        */
        proc pretty() throws {
            for n in tab {
                writeln("%10s = ".format(n), tab.getValue(n)); stdout.flush();
            }
        }

        /*
        returns total bytes in arrays in the symbol table
        */
        proc memUsed(): int {
            var total: int = + reduce [e in tab.values()] e.size * e.itemsize;
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
                return "%jt".format(this);
            } else if (tab.contains(name)) {
                return "%jt %jt".format(name, tab.getReference(name));
            } else {
                throw getErrorWithContext(
                    msg=unknownSymbolError(pname="dump",sname=name),
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="UnknownSymbolError");
            }
        }
        
        /*
        Returns verbose attributes of the sym entry at the given string, if the string maps to an entry.
        Pass __AllSymbols__ to process all sym entries in the sym table.

        Returns: name, dtype, size, ndim, shape, and item size

        :arg name: name of entry to be processed
        :type name: string

        :returns: s (string) containing info
        */
        proc info(name:string): string throws {
            var s: string;
            if name == "__AllSymbols__" {
                for n in tab {
                    s += "name:%t dtype:%t size:%t ndim:%t shape:%t itemsize:%t\n".format(n, 
                              dtype2str(tab.getBorrowed(n).dtype), tab.getBorrowed(n).size, 
                              tab.getBorrowed(n).ndim, tab.getBorrowed(n).shape, 
                              tab.getBorrowed(n).itemsize);
                }
            } else {
                if (tab.contains(name)) {
                    s = "name:%t dtype:%t size:%t ndim:%t shape:%t itemsize:%t\n".format(name, 
                              dtype2str(tab.getBorrowed(name).dtype), tab.getBorrowed(name).size, 
                              tab.getBorrowed(name).ndim, tab.getBorrowed(name).shape, 
                              tab.getBorrowed(name).itemsize);
                }
                else {
                    throw getErrorWithContext(
                        msg=unknownSymbolError(pname="info",sname=name),
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="UnknownSymbolError");                
                }
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
        proc attrib(name:string):string throws {
            var s:string;
            if (tab.contains(name)) {
                s = "%s %s %t %t %t %t".format(name, dtype2str(tab.getBorrowed(name).dtype), 
                          tab.getBorrowed(name).size, tab.getBorrowed(name).ndim, 
                          tab.getBorrowed(name).shape, tab.getBorrowed(name).itemsize);
            }
            else {
                throw getErrorWithContext(
                    msg=unknownSymbolError("attrib",name),
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="UnknownSymbolError");                   
            }
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
        proc datastr(name: string, thresh:int): string throws {
            var s:string;
            if (tab.contains(name)) {
                var u: borrowed GenSymEntry = tab.getBorrowed(name);
                select u.dtype
                {
                    when DType.Int64
                    {
                        var e = toSymEntry(u,int);
                        if e.size == 0 {s =  "[]";}
                        else if e.size < thresh+4 || e.size <= 6 {
                            s =  "[";
                            for i in 0..(e.size-2) {s += try! "%t ".format(e.a[i]);}
                            s += try! "%t]".format(e.a[e.size-1]);
                        }
                        else {
                            select FilteringPattern  
                            {
                                 when 0 //HeadAndTail 
                                 {
                                      var half=thresh/2:int;
                                      s =  "[";
                                      for i in 0..(half-2) {s += try! "%t ".format(e.a[i]);}
                                      s += try! "%t ... ".format(e.a[half-1]);
                                      for i in e.size-2-half..(e.size-2) {s += try! "%t ".format(e.a[i]);}
                                      s += try! "%t]".format(e.a[e.size-1]);

                                      //s = try! "[%t %t %t ... %t %t %t]".format(e.a[0],e.a[1],e.a[2],
                                      //                                e.a[e.size-3],
                                      //                                e.a[e.size-2],
                                      //                                e.a[e.size-1]);
                                 }
                                 when 1 //Head
                                 {
                                      s =  "[";
                                      for i in 0..thresh-2 {s += try! "%t ".format(e.a[i]);}
                                      s += try! "%t ...] ".format(e.a[thresh-1]);
                                 }
                                 when 2 //Tail
                                 {
                                      s =  "[... ";
                                      for i in e.size-1-thresh..e.size-2  {s += try! "%t ".format(e.a[i]);}
                                      s += try! "%t]".format(e.a[e.size-1]);
                                 }
                                 when 3 //Middle 
                                 {
                                      var startM=e.size-1-thresh/2:int;
                                      s =  "[... ";
                                      for i in startM..startM+thresh-2  {s += try! "%t ".format(e.a[i]);}
                                      s += try! "%t ...]".format(e.a[startM+thresh-1]);
                                       
                                 }
                                 when 4 //Uniform
                                 {
                                      var stride =(e.size-1)/thresh:int;
                                      s =  "[... ";
                                      for i in 0..thresh-2  {s += try! "%t ".format(e.a[i*stride]);}
                                      s += try! "%t ...]".format(e.a[ stride*(thresh-1)]);
                                 }
                                 when 5 //Random
                                 {
                                      var samplearray:[0..thresh-1]int;
                                      var indexarray:[0..thresh-1]int;
                                      fillInt(samplearray,0,e.size-1);
                                      var iv = radixSortLSD_ranks(samplearray);
                                      indexarray=samplearray[iv]:int;
                                      s =  "[... ";
                                      for i in 0..thresh-2  {
                                          if (e.a[indexarray[i]]!=e.a[indexarray[i+1]]) {
                                               s += try! "%t ".format(e.a[indexarray[i]]);
                                          }
                                          s += try! "%t ...]".format(e.a[indexarray[thresh-1]]);
                                      }
                                 }

                            }//end select
                        }//end else
                    }//end DType.Int64
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
                    otherwise {
                        s = unrecognizedTypeError("datastr",dtype2str(u.dtype));
                        mtLogger.error(getModuleName(),getRoutineName(),getLineNumber(),s);                        
                    }
                }
            }
            else {
                throw getErrorWithContext(
                    msg=unknownSymbolError("datastr",name),
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="UnknownSymbolError");                          
            }
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
        proc datarepr(name: string, thresh:int): string throws {
            var s:string;
            if (tab.contains(name)) {
                var u: borrowed GenSymEntry = tab.getBorrowed(name);
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
                    otherwise {
                        throw getErrorWithContext(
                            msg=unrecognizedTypeError("datarepr",dtype2str(u.dtype)),
                            lineNumber=getLineNumber(),
                            routineName=getRoutineName(),
                            moduleName=getModuleName(),
                            errorClass="TypeError");                            
                    }
                }
            }
            else {
                throw getErrorWithContext(
                    msg=unknownSymbolError("datarepr",name),
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="UnknownSymbolError");                  
            }
            return s;
        }
    }      
}
