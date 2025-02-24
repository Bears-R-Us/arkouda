module ArkoudaPythonCompat {
  public use Python;

  proc pythonVersionString(): string {
    return CPythonInterface.PY_MAJOR_VERSION:string + "." +
           CPythonInterface.PY_MINOR_VERSION:string;
  }

  proc pythonModuleSupported param do return true;
}
