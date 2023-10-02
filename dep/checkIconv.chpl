use iconv;
use ArkoudaCTypesCompat;

proc main() {
  var enc = "UTF-8": c_string_ptr;
  var cd = libiconv_open(enc, enc);
  libiconv_close(cd);
  writeln("Found libiconv version: ", _libiconv_version);
}
