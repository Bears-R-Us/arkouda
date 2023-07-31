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
}
