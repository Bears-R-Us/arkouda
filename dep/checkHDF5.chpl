use HDF5;

proc main() {
  var H5major: c_uint, H5minor: c_uint, H5micro: c_uint;
  C_HDF5.H5get_libversion(H5major, H5minor, H5micro);
  writeln("Found HDF5 version: %t".format((H5major:uint, H5minor:uint, H5micro:uint)));
  return 0;
}