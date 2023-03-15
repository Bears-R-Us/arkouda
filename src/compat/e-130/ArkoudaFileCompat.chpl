module ArkoudaFileCompat {
  public use IO;
  proc fileReader.bytesRead(ref a, b) throws {
    this.readBytes(a,b);
  }
}
