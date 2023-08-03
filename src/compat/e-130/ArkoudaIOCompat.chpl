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

  proc string.doFormat(vals...?): string throws {
    var toUse = this.replace('%?', '%t');
    return toUse.format((...vals));
  }

  proc jsonToTupleCompat(json: string, type t) throws {
    var f = openMemFile();
    var w = f.writer();
    w.write(json);
    w.close();
    var r = f.reader();
    var tup: t;
    r.readf("%jt", tup);
    r.close();
    return tup;
  }

  proc readfCompat(f: file, str: string, ref obj) throws {
    var nreader = f.reader();
    nreader.readf("%jt", obj);
  }
}
