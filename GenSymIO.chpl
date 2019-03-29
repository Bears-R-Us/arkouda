module GenSymIO {
  use HDF5;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use FileSystem;
  config const GenSymIO_DEBUG = false;

  class DatasetNotFoundError: Error { proc init() {} }
  
  proc readhdfMsg(reqMsg: string, st: borrowed SymTab): string {
    var rep_msg: string;
    // req_msg = "readhdf <dsetName> <filename1> <filename2> ..."
    var fields = reqMsg.split();
    var cmd = fields[1];
    var dsetName = fields[2];
    var filenames = fields[3..];
    var dataclass: C_HDF5.H5T_INTEGER.type;
    var firstname = true;
    for fname in filenames {
      try {
	if firstname {
	  var dataclass = get_dtype(fname, dsetName);
	  firstname = false;
	} else {
	  if get_dtype(fname, dsetName) != dataclass {
	    return try! "Error: unexpected dtype in dataset %s of file %s".format(dsetName, fname);
	  }
	}
      } catch e: FileNotFoundError {
	return try! "Error: file not found: %s".format(fname);
      } catch e: PermissionError {
	return try! "Error: permission error on %s".format(fname);
      } catch e: DatasetNotFoundError {
	return try! "Error: dataset %s not found in file %s".format(dsetName, fname);
      } catch {
	// Need a catch-all for non-throwing function
	return "Error: unknown cause";
      }
    }
    var (subdoms, len) = get_subdoms(filenames, dsetName);
    var entry: shared GenSymEntry;
    if dataclass == C_HDF5.H5T_INTEGER {
      var entryInt = new shared SymEntry(len, int);
      read_files_into_distributed_array(entryInt.a, subdoms, filenames, dsetName);
      entry = entryInt;
    } else if dataclass == C_HDF5.H5T_FLOAT {
      var entryReal = new shared SymEntry(len, real);
      read_files_into_distributed_array(entryReal.a, subdoms, filenames, dsetName);
      entry = entryReal;
    } else {
      return try! "Error: detected unhandled datatype code %i".format(dataclass);
    }
    var rname = st.nextName();
    st.addEntry(rname, entry);
    return try! "created " + st.attrib(rname);
  }
  
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
    if !C_HDF5.H5Lexists(file_id, dsetName.c_str(), C_HDF5.H5P_DEFAULT) {
      throw new owned DatasetNotFoundError();
    }
    var dset = C_HDF5.H5Dopen(file_id, dsetName.c_str(), C_HDF5.H5P_DEFAULT);   
    var datatype = C_HDF5.H5Dget_type(dset);
    var dataclass = C_HDF5.H5Tget_class(datatype);
    C_HDF5.H5Tclose(datatype);
    C_HDF5.H5Dclose(dset);
    C_HDF5.H5Fclose(file_id);
    return dataclass;
  }
       
  /* Get the subdomains of the distributed array represented by each file, as well as the total length of the array. */
  proc get_subdoms(filenames: [?FD] string, dsetName: string) {
    var lengths: [FD] int;
    forall (i, filename) in zip(FD, filenames) with (ref lengths) {
      var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, C_HDF5.H5P_DEFAULT);
      var dims: [0..#1] C_HDF5.hsize_t; // Only rank 1 for now
      var dsetRank: c_int;
      // Verify 1D array
      C_HDF5.H5LTget_dataset_ndims(file_id, dsetName.c_str(), dsetRank);
      if dsetRank != 1 {
	// TODO: change this to a throw
	halt("Expected 1D array, got rank " + dsetRank);
      }
      // Read array length into dims[0]
      C_HDF5.HDF5_WAR.H5LTget_dataset_info_WAR(file_id, dsetName.c_str(), c_ptrTo(dims), nil, nil);
      C_HDF5.H5Fclose(file_id);
      lengths[i] = dims[0]: int;
    }
    // Compute subdomain of master array contained in each file
    var subdoms: [FD] domain(1);
    var offset = 0;
    for i in FD {
      subdoms[i] = {offset..#lengths[i]};
      offset += lengths[i];
    }
    return (subdoms, (+ reduce lengths));
  }
	
  /* This function gets called when A is a BlockDist array. */
  proc read_files_into_distributed_array(A, filedomains: [?FD] domain(1), filenames: [FD] string, dsetName: string) where (MyDmap == 1) {
    if GenSymIO_DEBUG {
      writeln("entry.a.targetLocales() = ", A.targetLocales());
      writeln("Filedomains: ", filedomains);
    }
    coforall loc in A.targetLocales() do on loc {
	// Create local copies of args
	var locFiles = filenames;
	var locFiledoms = filedomains;
	var locDset = dsetName;
	/* On this locale, find all files containing data that belongs in 
	   this locale's chunk of A */
	for (filedom, filename) in zip(locFiledoms, locFiles) {
	  var isopen = false;
	  var file_id: C_HDF5.hid_t;
	  var dataset: C_HDF5.hid_t;
	  // Look for overlap between A's local subdomains and this file
	  for locdom in A.localSubdomains() {
	    const intersection = domain_intersection(locdom, filedom);
	    if intersection.size > 0 {
	      // Only open the file once, even if it intersects with many local subdomains
	      if !isopen {
		file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, C_HDF5.H5P_DEFAULT);
		dataset = C_HDF5.H5Dopen(file_id, locDset.c_str(), C_HDF5.H5P_DEFAULT);
		isopen = true;
	      }
	      // do A[intersection] = file[intersection - offset]
	      var dataspace = C_HDF5.H5Dget_space(dataset);
	      var dsetOffset = [(intersection.low - filedom.low): C_HDF5.hsize_t];
	      var dsetStride = [intersection.stride: C_HDF5.hsize_t];
	      var dsetCount = [intersection.size: C_HDF5.hsize_t];
	      C_HDF5.H5Sselect_hyperslab(dataspace, C_HDF5.H5S_SELECT_SET, c_ptrTo(dsetOffset), c_ptrTo(dsetStride), c_ptrTo(dsetCount), nil);
	      var memOffset = [0: C_HDF5.hsize_t];
	      var memStride = [1: C_HDF5.hsize_t];
	      var memCount = [intersection.size: C_HDF5.hsize_t];
	      var memspace = C_HDF5.H5Screate_simple(1, c_ptrTo(memCount), nil);
	      C_HDF5.H5Sselect_hyperslab(memspace, C_HDF5.H5S_SELECT_SET, c_ptrTo(memOffset), c_ptrTo(memStride), c_ptrTo(memCount), nil);
	      if GenSymIO_DEBUG {
		writeln("Locale ", loc, ", intersection ", intersection, ", dataset slice ", (intersection.low - filedom.low, intersection.high - filedom.low));
	      }
	      // The fact that intersection is a subset of a local subdomain means there should be no communication in the read
	      local {
		C_HDF5.H5Dread(dataset, getHDF5Type(A.eltType), memspace, dataspace, C_HDF5.H5P_DEFAULT, c_ptrTo(A.localSlice(intersection)));
	      }
	      C_HDF5.H5Sclose(memspace);
	      C_HDF5.H5Sclose(dataspace);
	    }
	  }
	  if isopen {
	    C_HDF5.H5Dclose(dataset);
	    C_HDF5.H5Fclose(file_id);
	  }
	}
      }
  }
	
  /* This function is called when A is a CyclicDist array. */
  proc read_files_into_distributed_array(A, filedomains: [?FD] domain(1), filenames: [FD] string, dsetName: string) where (MyDmap == 0) {
    use CyclicDist;
    // Distribute filenames across locales, and ensure single-threaded reads on each locale
    var fileSpace: domain(1) dmapped Cyclic(startIdx=FD.low, dataParTasksPerLocale=1) = FD;
    forall fileind in fileSpace with (ref A) {
      var filedom: subdomain(A.domain) = filedomains[fileind];
      var filename = filenames[fileind];
      var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, C_HDF5.H5P_DEFAULT);
      // TODO: use select_hyperslab to read directly into a strided slice of A
      // Read file into a temporary array and copy into the correct chunk of A
      var AA: [1..filedom.size] A.eltType;
      readHDF5Dataset(file_id, dsetName, AA);
      A[filedom] = AA;
      C_HDF5.H5Fclose(file_id);
    }
  }
	
  proc domain_intersection(d1: domain(1), d2: domain(1)) {
    var low = max(d1.low, d2.low);
    var high = min(d1.high, d2.high);
    if (d1.stride !=1) && (d2.stride != 1) {
      //TODO: change this to throw
      halt("At least one domain must have stride 1");
    }
    var stride = max(d1.stride, d2.stride);
    return {low..high by stride};
  }
}

