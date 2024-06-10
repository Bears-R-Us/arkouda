use idna;
use CTypes;

proc main() {
  extern proc idn2_check_version(a): c_ptrConst(c_char);
  var idnaCVerStr = idn2_check_version(nil);
  writeln("Found idn2 version: ", idnaCVerStr:string);
}
