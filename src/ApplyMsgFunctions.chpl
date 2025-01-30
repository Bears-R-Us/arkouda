module ApplyMsgFunctions {
  use NumPyDType;
  use ArkoudaPythonCompat;

  use BigInteger;

  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use ServerErrorStrings;
  use ServerErrors;
  use AryUtil;
  use Message;
  use Reflection;

  @arkouda.registerCommand
  proc applyStr(const ref x: [?d] ?t, funcStr: string): [d] t throws {
    var ret = makeDistArray(d, t);
    if !pythonModuleSupported {
      throw new ErrorWithContext(
        "Python module not supported with this version of Chapel",
        getLineNumber(),
        getRoutineName(),
        getModuleName()
      );
    } else {
      coforall l in d.targetLocales() do on l {
        var interp = new Interpreter();
        var func = new Function(interp, funcStr);
        for sd in d.localSubdomains() {
          for i in sd {
            ret[i] = func(t, x[i]);
          }
        }
      }
    }
    return ret;
  }


  @arkouda.instantiateAndRegister
  proc applyPickle(cmd: string,
                   msgArgs: borrowed MessageArgs,
                   st: borrowed SymTab,
                   type array_dtype,
                   param array_nd: int,
                   type array_dtype_to): MsgTuple throws
      where array_dtype != BigInteger.bigint &&
            array_dtype_to != BigInteger.bigint {

    if !pythonModuleSupported {
      throw new ErrorWithContext(
        "Python module not supported with this version of Chapel",
        getLineNumber(),
        getRoutineName(),
        getModuleName()
      );
    } else {
      const pickleDataStr = msgArgs["pickleData"].toScalar(string);
      var pickleData: bytes;
      {
        // TODO: this is a big hack, ideally we properly implement the base64 module
        // instead of using Python as a polyfill.
        // even better, we should have a way to just send bytes from the client to the server
        var interp = new Interpreter();
        var mod = new Module(interp, "base64");
        var b64decode = new Function(mod, "b64decode");
        pickleData = b64decode(bytes, pickleDataStr);
      }

      var xSym = st[msgArgs["x"]]: SymEntry(array_dtype, array_nd);
      ref x = xSym.a;

      var d = x.domain;
      var ret = makeDistArray(d, array_dtype_to);

      coforall l in d.targetLocales() do on l {
        var interp = new Interpreter();
        var func = new Value(interp, pickleData);
        for sd in d.localSubdomains() {
          for i in sd {
            ret[i] = func(array_dtype_to, x[i]);
          }
        }
      }
      return st.insert(new shared SymEntry(ret));
    }
  }
}
