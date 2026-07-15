module ApplyMsg {
  use NumPyDType;

  use Python;
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

  import this.Base64.b64Decode;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const applyLogger = new Logger(logLevel, logChannel);

  // each locale will have its own interpreter
  var interpreters = blockDist.createArray(0..<numLocales, owned Interpreter?);
  inline proc getInterpreter(): borrowed Interpreter {
    return interpreters[here.id]!;
  }

  proc pythonVersionString(): string {
    return CPythonInterface.PY_MAJOR_VERSION:string + "." +
           CPythonInterface.PY_MINOR_VERSION:string;
  }

  @arkouda.registerCommand
  proc initPythonInterpreters() throws {
    applyLogger.debug(
      getModuleName(),
      getRoutineName(),
      getLineNumber(),
      "Initializing Python interpreters");

    forall interp in interpreters do
      interp = new Interpreter();
  }

  @arkouda.registerCommand
  proc isVersionSupported(versionString: string): bool throws {
    return versionString == ServerConfig.pythonVersion &&
            versionString == pythonVersionString();
  }

  @arkouda.registerCommand
  proc applyStr(const ref x: [?d] ?t, funcStr: string): [d] t throws {
    var ret = makeDistArray(d, t);
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
    const pickleDataStr = msgArgs["pickleData"].toScalar(string);
    const pickleData: bytes = b64Decode(pickleDataStr);

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

  /*
    Copied and modified from https://github.com/jabraham17/Base64/blob/v0.1.0/src/Base64.chpl
  */
  module Base64 {
    /*
      Decode a Base64 string to binary data.
    */
    proc b64Decode(x: string): bytes do
      return b64DecodeImpl(x);

    private param padding = b"=";
    private proc decodeChar(ch: uint(8),
                            param plus: bytes, param slash: bytes): int {
      if ch >= 0x41 && ch <= 0x5A then return (ch - 0x41): int;        // A-Z
      else if ch >= 0x61 && ch <= 0x7A then return (ch - 0x61 + 26): int; // a-z
      else if ch >= 0x30 && ch <= 0x39 then return (ch - 0x30 + 52): int; // 0-9
      else if ch == plus.toByte() then return 62;
      else if ch == slash.toByte() then return 63;
      else return -1; // padding or invalid
    }

    private proc b64DecodeImpl(x: string,
                               param plus = b"+", param slash = b"/"): bytes {
      const len = x.size;
      if len == 0 then return b"";

      var result: bytes;
      var i = 0;

      while i < len {
        // Skip whitespace/newlines
        if x.byte[i] == 0x0A || x.byte[i] == 0x0D ||
          x.byte[i] == 0x20 || x.byte[i] == 0x09 {
          i += 1;
          continue;
        }

        // Need at least 2 valid chars for a group
        const a = decodeChar(x.byte[i], plus, slash);
        const b = if i+1 < len then decodeChar(x.byte[i+1], plus, slash) else 0;
        const c = if i+2 < len then decodeChar(x.byte[i+2], plus, slash) else 0;
        const d = if i+3 < len then decodeChar(x.byte[i+3], plus, slash) else 0;

        param mask = 0xFF:uint(8);
        result.appendByteValues((((a << 2) | (b >> 4)) & mask):uint(8));

        if i+2 < len && x.byte[i+2] != padding.toByte() {
          result.appendByteValues((((b << 4) | (c >> 2)) & mask):uint(8));
        }
        if i+3 < len && x.byte[i+3] != padding.toByte() {
          result.appendByteValues((((c << 6) | d) & mask):uint(8));
        }

        i += 4;
      }

      return result;
    }
  }
}
