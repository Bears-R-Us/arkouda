module ArkoudaFileCompat {
  use IO;
  proc file.appendWriter() throws {
    return this.writer(start=this.size);
  }
}
