use GenSymIO;
use MultiTypeSymbolTable;
use SegmentedArray;
use SegmentedMsg;
use UniqueMsg;

config const filename2 = "../../../data/hdf/netflow_day-02.h5";
config const filename3 = "../../../data/hdf/netflow_day-03.h5";
config const filename4 = "../../../data/hdf/netflow_day-04.h5";

proc main() {
  var st = new owned SymTab();
  var cmd = "readAllHdf";
  var reqMsg = "%s %i %i %jt | %jt".format(cmd, 7, 3,
     //['Time'],
     ['Time','Duration', 'Protocol', 'SrcPackets', 'DstPackets', 'SrcBytes', 'DstBytes'],
     //['Time','Duration', 'SrcDevice', 'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 'SrcBytes', 'DstBytes'],
     [filename2,filename3,filename4]);
  writeln(">>> ", reqMsg);
  var repMsg = readAllHdfMsg(reqMsg, st);
  writeln("<<< ", repMsg);

  if repMsg.startsWith("Error") {
    halt();
  }
}
