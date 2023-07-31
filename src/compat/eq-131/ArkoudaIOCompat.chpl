module ArkoudaIOCompat {
  use IO;

  proc formatString(input) throws {
    return input:string;
  }

  proc formatJson(input): string throws {
    return "%jt".format(input);
  }

  proc formatJson(input:string, vals...?): string throws {
    var toUse = input.replace("%?", "%jt");
    return toUse.format((...vals));
  }
}
