use HDF5, SysCTypes;

proc main() {
  var H5major: c_uint, H5minor: c_uint, H5micro: c_uint;
  C_HDF5.H5get_libversion(H5major, H5minor, H5micro);
  writef("Found HDF5 version: %t.%t.%t\n", H5major, H5minor, H5micro);
  return 0;
}
