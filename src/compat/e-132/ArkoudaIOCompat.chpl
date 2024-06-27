module ArkoudaIOCompat {
  use IO, JSON, Set;

  proc formatString(input) throws {
    return "%?".format(input);
  }

  proc formatJson(input): string throws {
    var f = openMemFile();
    f.writer(serializer = new jsonSerializer()).write(input);
    return f.reader().readAll(string);
  }

  proc formatJson(input:string, vals...?): string throws {
    var f = openMemFile();
    f.writer(serializer = new jsonSerializer()).writef(input, (...vals));
    return f.reader().readAll(string);
  }

  proc jsonToTupleCompat(json: string, type t) throws {
    var f = openMemFile();
    var w = f.writer();
    w.write(json);
    w.close();
    var r = f.reader(deserializer=new jsonDeserializer());
    var tup: t;
    r.readf("%?", tup);
    r.close();
    return tup;
  }

  proc jsonToPdArrayCompat(json: string, size: int) throws {
    var f = openMemFile();
    var w = f.writer();
    w.write(json);
    w.close();
    var r = f.reader(deserializer=new jsonDeserializer());
    var array: [0..#size] string;
    r.readf("%?", array);
    r.close();
    return array;
  }

  proc writefCompat(fmt: ?t, const args ...?k) where isStringType(t) || isBytesType(t) {
    writef(fmt, (...args));
  }

  proc readfCompat(f: file, str: string, ref obj) throws {
    var nreader = f.reader(deserializer=new jsonDeserializer());
    nreader.readf("%?", obj);
  }

  proc getByteOrderCompat() throws {
    use IO;
    var writeVal = 1, readVal = 0;
    var tmpf = openMemFile();
    tmpf.writer(serializer = new binarySerializer(endian=ioendian.big)).write(writeVal);
    tmpf.reader(deserializer=new binaryDeserializer(endian=ioendian.native)).read(readVal);
    return if writeVal == readVal then "big" else "little";
  }

  proc fileIOReaderCompat(infile) throws {
    return infile.reader(deserializer=new binarySerializer(endian=ioendian.native));
  }

  proc binaryCheckCompat(reader) throws {
    return reader.deserializerType == binarySerializer;
  }

  proc writeUsedModulesJson(ref mods: set(string)) {
    const cfgFile = try! open("UsedModules.json", ioMode.cw),
          w = try! cfgFile.writer(locking=false, serializer = new jsonSerializer());

    try! w.write(mods);
  }
}
