/* Array set operations
 includes intersection, union, xor, and diff

 currently, only performs operations with integer arrays 
 */

module LispMsg
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Reflection only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use SegmentedString;
    use ServerErrorStrings;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;

    private config const logLevel = ServerConfig.logLevel;
    const asLogger = new Logger(logLevel);

    /*
    Parse, execute, and respond to a setdiff1d message
    :arg reqMsg: request containing (cmd,name,name2,assume_unique)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (MsgTuple) response message
    */
    proc lispMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var (avalStr, xId, yId, code) = payload.splitMsgToTuple(4);
        writeln(avalStr, xId, yId, code);
        // Received: {'bindings': "{'a': {'type': 'float64', 'value': '5.0'}, 'x': {'type': 'pdarray', 'name': 'id_ej8Pi4s_1'}, 'y': {'type': 'pdarray', 'name': 'id_ej8Pi4s_2'}}", 'code': "'( begin ( return ( + ( * a x ) y ) ) )'"}
        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(xId, st);
        var gEnt2: borrowed GenSymEntry = getGenericTypedArrayEntry(yId, st);

        var x = toSymEntry(gEnt, real);
        var y = toSymEntry(gEnt2, real);
        writeln(x);
        writeln(y);

        repMsg = "applesauce and spaghetti";
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    
    use CommandMap;
    registerFunction("lispCode", lispMsg, getModuleName());
}
