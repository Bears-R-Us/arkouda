use HDF5;
use HDF5.C_HDF5;
use FileSystem;

config const filename:string;
config const dsetName:string;
/* config const bytearrayDset: string; */
/* config const offsetDset: string; */
config const groupName: string;
config const size: int = 100;

class DatasetNotFoundError: Error { proc init() {} }
class NotHDF5FileError: Error { proc init() {} }
class MismatchedAppendError: Error { proc init() {} }

/* Get the class of the HDF5 datatype for the dataset. */
proc get_dtype(filename: string, dsetName: string) throws {
  const READABLE = (S_IRUSR | S_IRGRP | S_IROTH);
  if !exists(filename) {
    throw new owned FileNotFoundError();
  }
  if !(getMode(filename) & READABLE) {
    throw new owned PermissionError();
  }
  var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, C_HDF5.H5P_DEFAULT);
  if file_id < 0 { // HF5open returns negative value on failure
    throw new owned NotHDF5FileError();
  }
  if !C_HDF5.H5Lexists(file_id, dsetName.c_str(), C_HDF5.H5P_DEFAULT) {
    throw new owned DatasetNotFoundError();
  }
  var dset = C_HDF5.H5Dopen(file_id, dsetName.c_str(), C_HDF5.H5P_DEFAULT);   
  var datatype = C_HDF5.H5Dget_type(dset);
  var dataclass = C_HDF5.H5Tget_class(datatype);
  C_HDF5.H5Tclose(datatype);
  C_HDF5.H5Dclose(dset);
  C_HDF5.H5Fclose(file_id);
  return (dataclass, datatype);
}

proc class2str(dataclass: hid_t): string {
  select dataclass {  
    when H5T_NO_CLASS {return "H5T_NO_CLASS";}
    when H5T_INTEGER {return "H5T_INTEGER";}
    when H5T_FLOAT {return "H5T_FLOAT";}
    when H5T_TIME {return "H5T_TIME";}
    when H5T_STRING {return "H5T_STRING";}
    when H5T_BITFIELD {return "H5T_BITFIELD";}
    when H5T_OPAQUE {return "H5T_OPAQUE";}
    when H5T_COMPOUND {return "H5T_COMPOUND";}
    when H5T_REFERENCE {return "H5T_REFERENCE";}
    when H5T_ENUM {return "H5T_ENUM";}
    when H5T_VLEN {return "H5T_VLEN";}
    when H5T_ARRAY {return "H5T_ARRAY";}
    when H5T_NCLASSES {return "H5T_NCLASSES";}
    otherwise { return "Unknown"; }
    }
}

proc readStrings(filename: string, dsetName: string, size:int) {
  var A: [0..#size] c_string;
  var file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  var dataset = H5Dopen(file_id, dsetName.c_str(), H5P_DEFAULT);
  var dataspace = H5Dget_space(dataset);
  var dsetOffset = [0: hsize_t];
  var dsetStride = [1: hsize_t];
  var dsetCount = [size: hsize_t];
  H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, c_ptrTo(dsetOffset), c_ptrTo(dsetStride), c_ptrTo(dsetCount), nil);
  var memOffset = [0: C_HDF5.hsize_t];
  var memStride = [1: C_HDF5.hsize_t];
  var memCount = [size: C_HDF5.hsize_t];
  var memspace = C_HDF5.H5Screate_simple(1, c_ptrTo(memCount), nil);
  C_HDF5.H5Sselect_hyperslab(memspace, C_HDF5.H5S_SELECT_SET, c_ptrTo(memOffset), c_ptrTo(memStride), c_ptrTo(memCount), nil);
  H5Dread(dataset, getHDF5Type(c_string), memspace, dataspace, H5P_DEFAULT, c_ptrTo(A));
  H5Sclose(memspace);
  H5Sclose(dataspace);
  var newA = [s in A] s:string;
  return newA;
}

proc readBytearray(filename: string, group: string, size: int) {
  var offsets: [0..#size] int;
  var file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  var dset_id = H5Dopen(file_id, group.c_str(), H5P_DEFAULT);
  if (dset_id < 0) {
    writeln("Verified that group is not a dataset");
  }
  var group_id = H5Gopen2(file_id, group.c_str(), H5P_DEFAULT);
  if (group_id < 0) {
    halt("Not a group");
  }
  var dtypeExists = H5Aexists(group_id, "segmented_string".c_str()): bool;
  if !dtypeExists {
    halt("Expected attr 'segmented_string'; not found");
  }
  var offsetDset = group+"/segments";
  var bytearrayDset = group+"/values";
  {
    var dataset = H5Dopen(file_id, offsetDset.c_str(), H5P_DEFAULT);
    var dataspace = H5Dget_space(dataset);
    var dsetOffset = [0: hsize_t];
    var dsetStride = [1: hsize_t];
    var dsetCount = [size: hsize_t];
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, c_ptrTo(dsetOffset), c_ptrTo(dsetStride), c_ptrTo(dsetCount), nil);
    var memOffset = [0: C_HDF5.hsize_t];
    var memStride = [1: C_HDF5.hsize_t];
    var memCount = [size: C_HDF5.hsize_t];
    var memspace = C_HDF5.H5Screate_simple(1, c_ptrTo(memCount), nil);
    C_HDF5.H5Sselect_hyperslab(memspace, C_HDF5.H5S_SELECT_SET, c_ptrTo(memOffset), c_ptrTo(memStride), c_ptrTo(memCount), nil);
    H5Dread(dataset, getHDF5Type(int), memspace, dataspace, H5P_DEFAULT, c_ptrTo(offsets));
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
  }

  var bytearrayize = offsets[size-1];
  var bytearray: [0..#bytearrayize] uint(8);
  {
    var dataset = H5Dopen(file_id, bytearrayDset.c_str(), H5P_DEFAULT);
    var dataspace = H5Dget_space(dataset);
    var dsetOffset = [0: hsize_t];
    var dsetStride = [1: hsize_t];
    var dsetCount = [bytearrayize: hsize_t];
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, c_ptrTo(dsetOffset), c_ptrTo(dsetStride), c_ptrTo(dsetCount), nil);
    var memOffset = [0: C_HDF5.hsize_t];
    var memStride = [1: C_HDF5.hsize_t];
    var memCount = [bytearrayize: C_HDF5.hsize_t];
    var memspace = C_HDF5.H5Screate_simple(1, c_ptrTo(memCount), nil);
    C_HDF5.H5Sselect_hyperslab(memspace, C_HDF5.H5S_SELECT_SET, c_ptrTo(memOffset), c_ptrTo(memStride), c_ptrTo(memCount), nil);
    H5Dread(dataset, getHDF5Type(uint(8)), memspace, dataspace, H5P_DEFAULT, c_ptrTo(bytearray));
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
  }
  H5Fclose(file_id);
  return (offsets, bytearray);
}

proc bytearrayToString(bytearray: [?D] uint(8)): string {
  var cBytes = c_ptrTo(bytearray);
  var s = new string(cBytes, D.size-1, D.size, isowned=false, needToCopy=true);
  return s;
}

proc main() {
  /* var (dataclass, datatype) = get_dtype(filename, bytearrayDset); */
  /* writeln("dtype = ", class2str(dataclass)); */
  var (offsets, bytearray) = readBytearray(filename, groupName, size);
  var size5 = offsets[6] - offsets[5];
  var bytearray5: [0..#size5] uint(8) = bytearray[offsets[5]..#size5];
  writeln("Item 5 = ", bytearrayToString(bytearray5));
  //  writeln("read in: [", A[0..#5], " ... ", A[size-5..], "]");
}