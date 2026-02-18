module ArkoudaJSONCompat {
  public use JSON;
  private use IO;

  proc toJson(arg): string throws {
    var memWriter = openMemFile();
    var writer = memWriter.writer(locking=false).withSerializer(jsonSerializer);
    writer.write(arg);
    writer.close();

    return memWriter.reader(locking=false).readAll(string);
  }
}
