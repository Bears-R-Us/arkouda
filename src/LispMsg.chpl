/* Processing of Arkouda lambda functions
   This allows users to write simple operations
   involving pdarrays and scalars to be computed
   in a single operation on the server side. This
   works by parsing the code, converting it to an
   AST, generating lisp code, then executing that
   lisp code on the server.
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

    use GenSymIO;
    use Message;

    private config const logLevel = ServerConfig.logLevel;
    const asLogger = new Logger(logLevel);
    const Tasks = {0..#numTasks};

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
        // TODO: If we support `|` in lisp, we don't want that to be delimeter
        var (retTypeStr, sizeStr, lispCode) = payload.splitMsgToTuple("|", 3);

        writeln("Ret type: ", retTypeStr);
        writeln("Size: ", sizeStr);
        writeln("Lisp code: ", lispCode);

        retTypeStr = retTypeStr.strip(" ");
        
        var size = sizeStr: int;

        var retName = st.nextName();

        if retTypeStr == "int64" {
          var ret = st.addEntry(retName, size, int);
          evalLisp(lispCode, ret.a, st);
        } else if retTypeStr == "float64" {
          var ret = st.addEntry(retName, size, real);
          evalLisp(lispCode, ret.a, st);
        }
        repMsg = "created " + st.attrib(retName);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    
    proc evalLisp(prog: string, ret: [] ?t, st) throws {
      try {
        coforall loc in Locales {
            on loc {
                coforall task in Tasks {
                    var lD = ret.domain.localSubdomain();
                    var tD = calcBlock(task, lD.low, lD.high);
                    var ast = parse(prog);
                    for i in tD {
                        var env = new owned Env();
                        env.addEntry("i", i);
                        
                        // Evaluate for this index
                        ret[i] = eval(ast, env, st).toValue(t).v;
                    }
                }
            }
        }
      } catch e {
        writeln(e!.message());
      }
    }
    use CommandMap;
    registerFunction("lispCode", lispMsg, getModuleName());
}
