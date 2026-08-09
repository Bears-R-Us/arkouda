module ApplyMsg {
  use NumPyDType;
  use ArkoudaPythonCompat;

  use BigInteger;
  use BlockDist;

  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use ServerErrorStrings;
  use ServerErrors;
  use AryUtil;
  use Message;
  use Reflection;

  use Logging;
  use ServerConfig;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const applyLogger = new Logger(logLevel, logChannel);

  // each locale will have its own interpreter
  var interpreters = blockDist.createArray(0..<numLocales, owned Interpreter?);
  inline proc getInterpreter(): borrowed Interpreter {
    return interpreters[here.id]!;
  }

  @arkouda.registerCommand
  proc isPythonModuleSupported(): bool {
    return pythonModuleSupported;
  }

  @arkouda.registerCommand
  proc initPythonInterpreters() throws {
    if !pythonModuleSupported {
      throw new ErrorWithContext(
        "Python module not supported with this version of Chapel",
        getLineNumber(),
        getRoutineName(),
        getModuleName()
      );
    } else {
      applyLogger.debug(
        getModuleName(),
        getRoutineName(),
        getLineNumber(),
        "Initializing Python interpreters");

      forall interp in interpreters do
        interp = new Interpreter();
    }
  }

  @arkouda.registerCommand
  proc isVersionSupported(versionString: string): bool throws {
    if !pythonModuleSupported {
      throw new ErrorWithContext(
        "Python module not supported with this version of Chapel",
        getLineNumber(),
        getRoutineName(),
        getModuleName()
      );
      return false;
    } else {
      return versionString == ServerConfig.pythonVersion &&
             versionString == ArkoudaPythonCompat.pythonVersionString();
    }
  }

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
        var func = new Function(getInterpreter(), funcStr);
        applyLogger.debug(
          getModuleName(),
          getRoutineName(),
          getLineNumber(),
          "Loaded function on locale " + l.id:string + ", applying function");
        for sd in d.localSubdomains() {
          for i in sd {
            ret[i] = func(t, x[i]);
          }
        }
        applyLogger.debug(
          getModuleName(),
          getRoutineName(),
          getLineNumber(),
          "Finished applying function on locale " + l.id:string);
      }
    }
    return ret;
  }


  @arkouda.instantiateAndRegister
  @chplcheck.ignore("UnusedFormal")
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
        // TODO: this is a big hack,
        // ideally we properly implement the base64 module
        // instead of using Python as a polyfill.
        // even better, we should have a way to just
        // send bytes from the client to the server
        var mod = new Module(getInterpreter(), "base64");
        var b64decode = new Function(mod, "b64decode");
        pickleData = b64decode(bytes, pickleDataStr);
      }


      var xSym = st[msgArgs["x"]]: SymEntry(array_dtype, array_nd);
      ref x = xSym.a;

      var d = x.domain;
      var ret = makeDistArray(d, array_dtype_to);

      coforall l in d.targetLocales() do on l {
        var func = new Value(getInterpreter(), pickleData);
        applyLogger.debug(
          getModuleName(),
          getRoutineName(),
          getLineNumber(),
          "Loaded pickle data on locale " +
          l.id:string + ", applying function");
        for sd in d.localSubdomains() {
          for i in sd {
            ret[i] = func(array_dtype_to, x[i]);
          }
        }
        applyLogger.debug(
          getModuleName(),
          getRoutineName(),
          getLineNumber(),
          "Finished applying function on locale " + l.id:string);
      }
      return st.insert(new shared SymEntry(ret));
    }
  }
}
