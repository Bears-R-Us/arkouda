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

  proc jsonToPdArrayCompat(json: string, size: int) throws {
    var f = openMemFile();
    var w = f.writer();
    w.write(json);
    w.close();
    var r = f.reader();
    var array: [0..#size] string;
    r.readf("%jt", array);
    r.close();
    return array;
  }

  proc readfCompat(f: file, str: string, ref obj) throws {
    var nreader = f.reader();
    nreader.readf("%jt", obj);
  }

  proc writefCompat(fmt: ?t, const args ...?k) where isStringType(t) || isBytesType(t) {
    var newFmt = fmt.replace('%?', '%t');
    writef(newFmt, (...args));
  }

  proc getByteOrderCompat() throws {
    use IO;
    var writeVal = 1, readVal = 0;
    var tmpf = openMemFile();
    tmpf.writer(kind=iobig).write(writeVal);
    tmpf.reader(kind=ionative).read(readVal);
    return if writeVal == readVal then "big" else "little";
  }

  proc fileIOReaderCompat(infile) throws {
    return infile.reader(kind=ionative);
  }

  proc binaryCheckCompat(reader) throws {
    return reader.binary();
  }
}
