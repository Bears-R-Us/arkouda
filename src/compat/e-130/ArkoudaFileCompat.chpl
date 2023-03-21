module ArkoudaFileCompat {
  public use IO;

  // We can't just use `readBytes` here because in previous
  // releases, `readBytes` had a different meaning, so we
  // need to override with a new name
  proc fileReader.bytesRead(ref a, b) throws {
    this.readBytes(a,b);
  }
}
