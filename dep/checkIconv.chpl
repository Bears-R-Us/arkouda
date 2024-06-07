use iconv;
use CTypes;

proc main() {
  var enc = "UTF-8": c_ptrConst(c_char);
  var cd = libiconv_open(enc, enc);
  libiconv_close(cd);
  writeln("Found libiconv version: ", _libiconv_version);
}
