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

    // r.readf("%?", array);

    // temporary solution for tuples until frontend JSON serialization is fixed
    // (i.e., the frontend should not serialize an array of numbers with quotes around each number)
    var first = true;
    r.matchLiteral("[");
    for i in 0..<size {
      if first { first = false; } else { try { r.matchLiteral(","); } catch {} }
      if t != string then r.matchLiteral('"');
      array[i] = r.read(t);
      if t != string then r.matchLiteral('"');
    }
    r.matchLiteral("]");

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
    var item: t;
    r.readf("%?", item);
    r.close();
    return item;
  }

  // temporary solution for tuples until frontend JSON serialization is fixed
  // (i.e., the frontend should not serialize a tuple of numbers with quotes around each number)
  proc parseJson(json: string, param n: int, type t): n*t throws {
    var f = openMemFile();
    var w = f.writer(locking=false);
    w.write(json);
    w.close();
    var r = f.reader(deserializer=new jsonDeserializer(), locking=false);
    var tup: n*t;

    var first = true;
    r.matchLiteral("[");
    for i in 0..<n {
      if first { first = false; } else { try { r.matchLiteral(","); } catch {} }
      if t != string then r.matchLiteral('"');
      tup[i] = r.read(t);
      if t != string then r.matchLiteral('"');
    }
    r.matchLiteral("]");

    return tup;
  }
}
