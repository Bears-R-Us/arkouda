module IOUtils {
  use IO;
  use JSON;

  /*
      Format an argument as a JSON string
  */
  proc formatJson(input): string throws {
    var f = openMemFile();
    f.writer(serializer = new jsonSerializer(), locking=false).write(input);
    return f.reader(locking=false).readAll(string);
  }

  proc formatJson(input:string, vals...?): string throws {
    var f = openMemFile();
    f.writer(serializer = new jsonSerializer(), locking=false).writef(input, (...vals));
    return f.reader(locking=false).readAll(string);
  }

  /*
    Deserialize a Chapel array of strings from a JSON string
  */
  proc jsonToArray(json: string, type t, size: int) throws {
    var f = openMemFile();
    var w = f.writer(locking=false);
    w.write(json);
    w.close();
    var r = f.reader(deserializer=new jsonDeserializer(), locking=false);
    var array: [0..#size] t;
    r.readf("%?", array);
    r.close();
    return array;
  }

  /*
    Helper function to parse a JSON string as an item of the given type
  */
  proc parseJson(json: string, type t): t throws {
    var f = openMemFile();
    var w = f.writer(locking=false);
    w.write(json);
    w.close();
    var r = f.reader(deserializer=new jsonDeserializer(), locking=false);
    var tup: t;
    r.readf("%?", tup);
    r.close();
    return tup;
  }
}
