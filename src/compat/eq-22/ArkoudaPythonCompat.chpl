module ArkoudaPythonCompat {
  proc pythonModuleSupported param do return false;

  class Interpreter {}
  class Value {}
  class Function: Value {}
  class Module: Value {}
}
