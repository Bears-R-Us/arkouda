use idna, CTypes;

proc main() {
  extern proc idn2_check_version(a): c_string;
  var idnaCVerStr = idn2_check_version(nil);
  writeln("Found idn2 version: ", idnaCVerStr:string);
}
