module ArkoudaIOCompat {
  use IO;

  proc formatString(input) throws {
    return input:string;
  }

  proc formatJson(input): string throws {
    return "%jt".format(input);
  }

  proc string.doFormat(vals...?): string throws {
    var toUse = this.replace('%?', '%t');
    return toUse.format((...vals));
  }
}
