module ArkoudaIOCompat {
  use IO, JSON;
  
  proc formatString(input) throws {
    return "%?".format(input);
  }

  proc formatJson(input): string throws {
    var f = openMemFile();
    f.writer(serializer = new JsonSerializer()).write(input);
    return f.reader().readAll(string);
  }

  proc formatJson(input:string, vals...?): string throws {
    var f = openMemFile();
    f.writer(serializer = new JsonSerializer()).writef(input, (...vals));
    return f.reader().readAll(string);
  }

  proc string.doFormat(vals...?): string throws {
    return this.format((...vals));
  }

  proc jsonToTupleCompat(json: string, type t) throws {
    var f = openMemFile();
    var w = f.writer();
    w.write(json);
    w.close();
    var r = f.reader(deserializer=new JsonDeserializer());
    var tup: t;
    r.readf("%?", tup);
    r.close();
    return tup;
  }
}
