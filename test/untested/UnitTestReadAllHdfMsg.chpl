use GenSymIO;
use MultiTypeSymbolTable;
use SegmentedArray;
use SegmentedMsg;
use UniqueMsg;

config const filename2 = "../converter/netflow_day-02.hdf";
config const filename3 = "../converter/netflow_day-03.hdf";
config const filename4 = "../converter/netflow_day-04.hdf";

proc main() {
  var st = new owned SymTab();
  var cmd = "readAllHdf";
  var reqMsg = "%i %i %jt | %jt".format(11, 3,
     ['start','duration', 'srcIP', 'dstIP', 'proto', 'srcPort', 'dstPort', 'srcPkts', 'dstPkts', 'srcBytes', 'dstBytes'],
     [filename2,filename3,filename4]);
  writeln(">>> ", reqMsg);
  var repMsg = readAllHdfMsg(cmd=cmd, payload=reqMsg.encode(), st);
  writeln("<<< ", repMsg);

  if repMsg.startsWith("Error") {
    halt();
  }
}
