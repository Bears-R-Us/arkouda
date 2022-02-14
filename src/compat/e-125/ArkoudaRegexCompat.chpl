module ArkoudaRegexCompat {
    // Chapel v1.25
    public use Regex;
    proc regexMatch.numBytes {
      return this.size;
    }
    proc regexMatch.byteOffset {
      return this.offset;
    }
}
