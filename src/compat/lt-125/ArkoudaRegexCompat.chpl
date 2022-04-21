module ArkoudaRegexCompat {
    // Chapel v1.24 and prior
    public use Regexp;
    proc reMatch.numBytes {
      return this.size;
    }
    proc reMatch.byteOffset {
      return this.offset;
    }
}
