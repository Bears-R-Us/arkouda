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
        var (lispCode) = payload.splitMsgToTuple(1);
        var ret = evalLisp(lispCode, st);
        var vname = st.nextName();
        st.addEntry(vname, new shared SymEntry(ret));
        repMsg = "created " + st.attrib(vname);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    
    // arrs is a tuple of the incoming arrays
    // arrNames is a list of names corresponding to arrs (so is same length as arrs)
    // vals are the values passed in
    // valNames are the names of those values (so is same length as vals)
    proc evalLisp(prog: string, st) {
      // TOOD: How do we want to construct ret?
      //       need size and type to know
      var ret: [0..#10] real;
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
                        ret[i] = eval(ast, env, st).toValue(real).v;
                    }
                }
            }
        }
      } catch e {
        writeln(e!.message());
      }
      return ret;
    }
    use CommandMap;
    registerFunction("lispCode", lispMsg, getModuleName());
}
