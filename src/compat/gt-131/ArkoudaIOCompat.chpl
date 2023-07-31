module ArkoudaIOCompat {
  use IO;
  
  proc formatString(input) throws {
    return "%?".format(input);
  }
}
