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

    use LisExprData;
    use LisExprInterp;
    use TestLisExpr;

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

        evalLisp(code, 5, x.a, y.a);

        repMsg = "applesauce and spaghetti";
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc evalLisp(prog: string, val: int, arr1, arr2) {
        try {
            var A = arr1;
            var B = arr2;
            
            for (a,b) in zip(A,B) {
                var ast = parse(prog); // parse and check the program
                var env = new owned Env(); // allocate the env for variables
                // addEnrtry redefines values for already existing entries
                env.addEntry("elt",a); // add a symbol called "elt" and value for a
                
                // this version does the eval the in the enviroment which creates the symbol "ans"
                //var ans = env.lookup("ans").toValue(int).v; // retrieve value for ans
                //b = ans;

                // this version just returns the GenValue from the eval call
                var ans = eval(ast,env);
                b = ans.toValue(int).v; // put answer into b
            }
            writeln(A);
            writeln(B);
        }
        catch e: Error {
            writeln(e.message());
        }
    }
    
    use CommandMap;
    registerFunction("lispCode", lispMsg, getModuleName());
}
