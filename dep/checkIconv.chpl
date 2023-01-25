use iconv;

proc main() {
  var enc = "UTF-8";
  var cd = libiconv_open(enc.c_str(), enc.c_str());
  libiconv_close(cd);
  writeln("Found libiconv version: ", _libiconv_version);
}
