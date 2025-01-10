module ApplyMsgFunctions {

  use Python;

  @arkouda.registerCommand
  proc apply(const ref x: [?d] ?t, funcStr: string): [] t throws {
    var ret = makeDistArray(d, t);
    coforall l in d.targetLocales() do on l {
      var interp = new Interpreter();
      var func = new Function(interp, funcStr);
      for sd in d.localSubdomains() {
        for i in sd {
          ret[i] = func(t, x[i]);
        }
      }
    }
    return ret;
  }


}
