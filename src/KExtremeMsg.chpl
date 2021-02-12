/* min-k reduction 
 * Stores the sorted minimum k values onto the server
 */

module KExtremeMsg
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Reflection;
    use Errors;
    use Logging;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use SegmentedArray;
    use ServerErrorStrings;

    use KReduce;
    use Indexing;
    use RadixSortLSD;
    use ArraySetopsMsg;

    const keLogger = new Logger();
  
    if v {
        keLogger.level = LogLevel.DEBUG;
    } else {
        keLogger.level = LogLevel.INFO;
    }

    /*
    Parse, execute, and respond to a mink message
    :arg reqMsg: request containing (name,k,returnIndices)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (string) response message
    */
    proc minkMsg(cmd: string, payload: string, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, k, returnIndices) = payload.splitMsgToTuple(3);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = st.lookup(name);

        select(gEnt.dtype) {
          when (DType.Int64) {
             var e = toSymEntry(gEnt,int);

             var aV;

             if !stringtobool(returnIndices) {
               aV = computeExtremaValues(e.a, k:int);
             } else {
               aV = computeExtremaIndices(e.a, k:int);
             }

             st.addEntry(vname, new shared SymEntry(aV));

             var s = try! "created " + st.attrib(vname);
             return s;
          }
          when (DType.Float64) {
            if !stringtobool(returnIndices) {
             var e = toSymEntry(gEnt,real);

             var aV = computeExtremaValues(e.a, k:int);

             st.addEntry(vname, new shared SymEntry(aV));

             var s = try! "created " + st.attrib(vname);
             return s;
            } else {
             var e = toSymEntry(gEnt,real);

             var aV = computeExtremaIndices(e.a, k:int);

             st.addEntry(vname, new shared SymEntry(aV));

             var s = try! "created " + st.attrib(vname);
             return s;
            }
           }
           otherwise {
               var errorMsg = notImplementedError("mink",gEnt.dtype);
               keLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
               return errorMsg;               
           }
        }
    }
    
   /*
    Parse, execute, and respond to a maxk message
    :arg reqMsg: request containing (name,k,returnIndices)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (string) response message
    */
    proc maxkMsg(cmd: string, payload: string, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, k, returnIndices) = payload.splitMsgToTuple(3);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = st.lookup(name);

        select(gEnt.dtype) {
          when (DType.Int64) {
             var e = toSymEntry(gEnt,int);

             var aV;
             if !stringtobool(returnIndices) {
               aV = computeExtremaValues(e.a, k:int, false);
             } else {
               aV = computeExtremaIndices(e.a, k:int, false);
             }

             st.addEntry(vname, new shared SymEntry(aV));

             var s = try! "created " + st.attrib(vname);
             return s;
           }
           when (DType.Float64) {
             if !stringtobool(returnIndices) {
               var e = toSymEntry(gEnt,real);

               var aV = computeExtremaValues(e.a, k:int, false);

               st.addEntry(vname, new shared SymEntry(aV));

               var s = try! "created " + st.attrib(vname);
               return s;
             } else {
               var e = toSymEntry(gEnt,real);

               var aV = computeExtremaIndices(e.a, k:int, false);

               st.addEntry(vname, new shared SymEntry(aV));

               var s = try! "created " + st.attrib(vname);
               return s;
               
             }
           }

           otherwise {
               var errorMsg = notImplementedError("maxk",gEnt.dtype);
               keLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
               return errorMsg;
           }
        }
    }

}
