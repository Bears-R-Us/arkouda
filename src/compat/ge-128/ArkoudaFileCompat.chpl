module ArkoudaFileCompat {
  use IO;
  proc file.appendWriter() throws {
    return this.writer(region=this.size..);
  }
}
