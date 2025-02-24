module ArkoudaPythonCompat {
  proc pythonModuleSupported param do return false;

  proc pythonVersionString(): string do return "";

  class Interpreter {}
  class Value {}
  class Function: Value {}
  class Module: Value {}
}
