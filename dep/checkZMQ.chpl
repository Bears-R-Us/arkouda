use ZMQ;
use ArkoudaIOCompat;

proc main() {
  var (Zmajor, Zminor, Zmicro) = ZMQ.version;
  writef("Found ZMQ version: %?.%?.%?\n", Zmajor, Zminor, Zmicro);
  return 0;
}
