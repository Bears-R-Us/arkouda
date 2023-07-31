module ArkoudaIOCompat {
  use IO;

  proc formatString(input) throws {
    return input:string;
  }

  proc formatJson(input): string throws {
    return "%jt".format(input);
  }
}
