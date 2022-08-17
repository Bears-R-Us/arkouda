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
      var ret: [0..#10] real;
      try {
        for i in ret.domain {
          var ast = parse(prog);
          var env = new owned Env();
          env.addEntry("i", i);

          // Evaluate for this index
          var ans = eval(ast, env, st);
          ret[i] = ans.toValue(real).v;
        }
      } catch e {
        writeln(e!.message());
      }
      return ret;
    }

    /*
    proc evalLisp(prog: string, arrs ...?n) {
      // arrs is a list of arrays and their corresponding names
      var ret: [0..#arrs[0].size] real;
      try {
        for i in 0..#arrs[0].size {
          var ast = parse(prog);
          var env = new owned Env();

          for param j in 0..#n by 2{
            // arrs[j+1] is name, arrs[j][i] is val at current index of current array
            env.addEntry(arrs[j+1], arrs[j][i]);
          }
          var ans = eval(ast, env);
          ret[i] = ans.toValue(real).v;
        }
      }
        catch e: Error {
            writeln(e.message());
        }
        return ret;
        } */
    
    use CommandMap;
    registerFunction("lispCode", lispMsg, getModuleName());
}
