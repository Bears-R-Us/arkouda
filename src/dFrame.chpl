module Dframe {
  use HDF5;
  use IO;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use ServerErrorStrings;
  use FileSystem;
  use Sort;
  use CommAggregation;

  config const GenSymIO_DEBUG = false;
  config const SEGARRAY_OFFSET_NAME = "segments";
  config const SEGARRAY_VALUE_NAME = "values";

  proc dFrameMsg(reqMsg: string, st: borrowed SymTab): string {
    var repMsg: string;
    var (cmd, col, arrays) = reqMsg.splitMsgToTuple(3);
    var rname = st.nextName();

    return try! "created " + rname + " " + col + " " + arrays;
  }
}