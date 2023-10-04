use idna;
use ArkoudaCTypesCompat;

proc main() {
  extern proc idn2_check_version(a): c_string_ptr;
  var idnaCVerStr = idn2_check_version(nil);
  writeln("Found idn2 version: ", idnaCVerStr:string);
}
