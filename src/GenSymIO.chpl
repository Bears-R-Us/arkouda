module GenSymIO {
  use HDF5;
  use ServerConfig;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use ServerErrorStrings;
  use FileSystem;
  use Sort;
  config const GenSymIO_DEBUG = false;

  proc arrayMsg(reqMsg: string, st: borrowed SymTab): string {
    var repMsg: string;
    var fields = reqMsg.split(3);
    var cmd = fields[1];
    var dtype = str2dtype(fields[2]);
    var size = try! fields[3]:int;
    var data = fields[4];
    var tmpf:file; 
    try {
      tmpf = openmem();
      var tmpw = tmpf.writer(kind=iobig);
      tmpw.write(data);
      try! tmpw.close();
    } catch {
      return "Error: Could not write to memory buffer";
    }
    try {
      const entry: shared GenSymEntry = readEntry();
      const rname = st.nextName();
      st.addEntry(rname, entry);
      return try! "created " + st.attrib(rname);
    } catch err: UnhandledDataTypeError {
      return try! "Error: Unhandled data type %s".format(err.dtype);
    } catch {
      return "Error: Could not read from memory buffer into SymEntry";
    }

    class UnhandledDataTypeError: Error {
      var dtype: DType;
    }

    proc readEntry(): shared GenSymEntry throws {
      var tmpr = tmpf.reader(kind=iobig, start=0);
      if dtype == DType.Int64 {
	var entryInt = new shared SymEntry(size, int);
	tmpr.read(entryInt.a);
	tmpr.close(); tmpf.close();
	return entryInt;
      } else if dtype == DType.Float64 {
	var entryReal = new shared SymEntry(size, real);
	tmpr.read(entryReal.a);
	tmpr.close(); tmpf.close();
	return entryReal;
      } else if dtype == DType.Bool {
	var entryBool = new shared SymEntry(size, bool);
	tmpr.read(entryBool.a);
	tmpr.close(); tmpf.close();
	return entryBool;
      } else {
	tmpr.close();
	tmpf.close();
	throw new owned UnhandledDataTypeError(dtype);
      }
      tmpr.close();
      tmpf.close();
    }
  }

  proc tondarrayMsg(reqMsg: string, st: borrowed SymTab): string throws {
    var arraystr: string;
    var fields = reqMsg.split();
    var entry = st.lookup(fields[2]);
    var tmpf: file;
    try {
      tmpf = openmem();
      var tmpw = tmpf.writer(kind=iobig);
      if entry.dtype == DType.Int64 {
	tmpw.write(toSymEntry(entry, int).a);
      } else if entry.dtype == DType.Float64 {
	tmpw.write(toSymEntry(entry, real).a);
      } else if entry.dtype == DType.Bool {
	tmpw.write(toSymEntry(entry, bool).a);
      } else {
	return try! "Error: Unhandled dtype %s".format(entry.dtype);
      }
      tmpw.close();
    } catch {
      try! tmpf.close();
      return "Error: Unable to write SymEntry to memory buffer";
    }
    try {
      var tmpr = tmpf.reader(kind=iobig, start=0);
      tmpr.readstring(arraystr);
      tmpr.close();
      tmpf.close();
    } catch {
      return "Error: Unable to copy array from memory buffer to string";
    }
    //var repMsg = try! "Array: %i".format(arraystr.length) + arraystr;
    return arraystr;
  }

  class DatasetNotFoundError: Error { proc init() {} }
  class NotHDF5FileError: Error { proc init() {} }
  class MismatchedAppendError: Error { proc init() {} }

  proc decode_json(json: string, size: int) throws {
    var f = opentmp();
    var w = f.writer();
    w.write(json);
    w.close();
    var r = f.reader(start=0);
    var array: [0..#size] string;
    r.readf("%jt", array);
    r.close();
    f.close();
    return array;
  }

  proc lshdfMsg(reqMsg: string, st: borrowed SymTab): string {
    // reqMsg: "lshdf [<json_filename>]"
    use Spawn;
    const tmpfile = "/tmp/arkouda.lshdf.output";
    var repMsg: string;
    var fields = reqMsg.split(1);
    var cmd = fields[1];
    var jsonfile = fields[2];
    var filename: string;
    try {
      filename = decode_json(jsonfile, 1)[0];
    } catch {
      return try! "Error: could not decode json filenames via tempfile (%i files: %s)".format(1, jsonfile);
    }
    // Attempt to interpret filename as a glob expression and ls the first result
    var tmp = glob(filename);
    if GenSymIO_DEBUG {
      writeln(try! "glob expanded %s to %i files".format(filename, tmp.size));
    }
    if tmp.size <= 0 {
      return try! "Error: no files matching %s".format(filename);
    }
    filename = tmp[tmp.domain.first];
    var exitCode: int;
    try {
      if exists(tmpfile) {
	remove(tmpfile);
      }
      var cmd = try! "h5ls \"%s\" > \"%s\"".format(filename, tmpfile);
      var sub = spawnshell(cmd);
      // sub.stdout.readstring(repMsg);
      sub.wait();
      exitCode = sub.exit_status;
      var f = open(tmpfile, iomode.r);
      var r = f.reader(start=0);
      r.readstring(repMsg);
      r.close();
      f.close();
      remove(tmpfile);
    } catch {
      return "Error: failed to spawn process and read output";
    }
    
    if exitCode != 0 {
      return try! "Error: %s".format(repMsg);
    } else {
      return repMsg;
    }
  }
  
  proc readhdfMsg(reqMsg: string, st: borrowed SymTab): string {
    var repMsg: string;
    // reqMsg = "readhdf <dsetName> <nfiles> [<json_filenames>]"
    var fields = reqMsg.split(3);
    var cmd = fields[1];
    var dsetName = fields[2];
    var nfiles = try! fields[3]:int;
    var jsonfiles = fields[4];
    var filelist: [0..#nfiles] string;
    try {
      filelist = decode_json(jsonfiles, nfiles);
    } catch {
      return try! "Error: could not decode json filenames via tempfile (%i files: %s)".format(nfiles, jsonfiles);
    }
    var filedom = filelist.domain;
    var filenames: [filedom] string;
    if filelist.size == 1 {
      var tmp = glob(filelist[0]);
      if GenSymIO_DEBUG {
	writeln(try! "glob expanded %s to %i files".format(filelist[0], tmp.size));
      }
      if tmp.size == 0 {
	return try! "Error: no files matching %s".format(filelist[0]);
      }
      // Glob returns filenames in weird order. Sort for consistency
      // sort(tmp);
      filedom = tmp.domain;
      filenames = tmp;
    } else {
      filenames = filelist;
    }
    var dclasses: [filenames.domain] C_HDF5.hid_t;
    for (i, fname) in zip(filenames.domain, filenames) {
      try {
	dclasses[i] = get_dtype(fname, dsetName);
      } catch e: FileNotFoundError {
	return try! "Error: file not found: %s".format(fname);
      } catch e: PermissionError {
	return try! "Error: permission error on %s".format(fname);
      } catch e: DatasetNotFoundError {
	return try! "Error: dataset %s not found in file %s".format(dsetName, fname);
      } catch e: NotHDF5FileError {
	return try! "Error: cannot open as HDF5 file %s".format(fname);
      } catch {
	// Need a catch-all for non-throwing function
	return try! "Error: unknown cause";
      }
    }
    const dataclass = dclasses[dclasses.domain.first];
    for (i, dc) in zip(dclasses.domain, dclasses) {
      if dc != dataclass {
	return try! "Error: inconsistent dtype in dataset %s of file %s".format(dsetName, filenames[i]);
      }
    }
    if GenSymIO_DEBUG {
      writeln("Verified all dtypes across files");
    }
    var subdoms: [filenames.domain] domain(1);
    var len: int;
    try {
      (subdoms, len) = get_subdoms(filenames, dsetName);
    } catch e: HDF5RankError {
      return notImplementedError("readhdf", try! "Rank %i arrays".format(e.rank));
    } catch {
      return try! "Error: unknown cause";
    }
    if GenSymIO_DEBUG {
      writeln("Got subdomains and total length");
    }
    try {
    var entry: shared GenSymEntry = computeEntry(dataclass);
    var rname = st.nextName();
    st.addEntry(rname, entry);
    return try! "created " + st.attrib(rname);
    } catch e: UnhandledHDFSDataTypeError {
      return try! "Error: detected unhandled datatype code %i".format(dataclass);
    } catch {
      return "Unexpected error";
    }

    class UnhandledHDFSDataTypeError: Error {
    }

    // This assumes that dataclass has already been validated above in
    // validateDataClass() and that only expected dataclass values
    // will be sent in.  If this is not the case, a halt occurs.
    proc computeEntry(dataclass: C_HDF5.hid_t): shared GenSymEntry throws {
    if dataclass == C_HDF5.H5T_INTEGER {
      var entryInt = new shared SymEntry(len, int);
      if GenSymIO_DEBUG {
	writeln("Initialized int entry"); try! stdout.flush();
      }
      read_files_into_distributed_array(entryInt.a, subdoms, filenames, dsetName);
      return entryInt;
    } else if dataclass == C_HDF5.H5T_FLOAT {
      var entryReal = new shared SymEntry(len, real);
      if GenSymIO_DEBUG {
	writeln("Initialized float entry"); try! stdout.flush();
      }
      read_files_into_distributed_array(entryReal.a, subdoms, filenames, dsetName);
      return entryReal;
    } else {
      throw new owned UnhandledHDFSDataTypeError();
    }
  }
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
    return dataclass;
  }

  class HDF5RankError: Error {
    var rank: int;
    var filename: string;
    var dsetName: string;
  }
  
  /* Get the subdomains of the distributed array represented by each file, as well as the total length of the array. */
  proc get_subdoms(filenames: [?FD] string, dsetName: string) throws {
    var lengths: [FD] int;
    for (i, filename) in zip(FD, filenames) {
      var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, C_HDF5.H5P_DEFAULT);
      var dims: [0..#1] C_HDF5.hsize_t; // Only rank 1 for now
      var dsetRank: c_int;
      // Verify 1D array
      C_HDF5.H5LTget_dataset_ndims(file_id, dsetName.c_str(), dsetRank);
      if dsetRank != 1 {
	// TODO: change this to a throw
	// halt("Expected 1D array, got rank " + dsetRank);
	throw new owned HDF5RankError(dsetRank, filename, dsetName);
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
      writeln("entry.a.targetLocales() = ", A.targetLocales()); try! stdout.flush();
      writeln("Filedomains: ", filedomains); try! stdout.flush();
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

  proc tohdfMsg(reqMsg, st: borrowed SymTab): string throws {
    // reqMsg = "tohdf <arrayName> <dsetName> <mode> [<json_filename>]"
    var fields = reqMsg.split(4);
    var cmd = fields[1];
    var arrayName = fields[2];
    var dsetName = fields[3];
    var mode = try! fields[4]: int;
    var jsonfile = fields[5];
    var filename: string;
    try {
      filename = decode_json(jsonfile, 1)[0];
    } catch {
      return try! "Error: could not decode json filenames via tempfile (%i files: %s)".format(1, jsonfile);
    }
    var entry = st.lookup(arrayName);
    var warnFlag: bool;
    try {
    select entry.dtype {
      when DType.Int64 {
	var e = toSymEntry(entry, int);
	//C_HDF5.HDF5_WAR.H5LTmake_dataset_WAR(file_id, dsetName.c_str(), 1, c_ptrTo(dims), getHDF5Type(e.a.eltType), c_ptrTo(e.a));
	warnFlag = write1DDistArray(filename, mode, dsetName, e.a);
      }
      when DType.Float64 {
	var e = toSymEntry(entry, real);
	warnFlag = write1DDistArray(filename, mode, dsetName, e.a);
      }
      when DType.Bool {
	var e = toSymEntry(entry, bool);
	warnFlag = write1DDistArray(filename, mode, dsetName, e.a);
      }
      otherwise {
	return unrecognizedTypeError("tohdf", dtype2str(entry.dtype));
      }
    }
    } catch e: FileNotFoundError {
      return try! "Error: unable to open file for writing: %s".format(filename);
    } catch e: MismatchedAppendError {
      return "Error: appending to existing files must be done with the same number of locales. Try saving with a different directory or filename prefix?";
    } catch {
      return "Error: problem writing to file";
    }
    if warnFlag {
      return "Warning: possibly overwriting existing files matching filename pattern";
    } else {
      return "wrote array to file";
    }
  }

  proc write1DDistArray(filename, mode, dsetName, A) throws {
    /* Output is 1 file per locale named <filename>_<loc>, and a dataset 
       named <dsetName> is created in each one. If mode==1 (append) and the 
       correct number of files already exists, then a new dataset named 
       <dsetName> will be created in each. Strongly recommend only using 
       append mode to write arrays with the same domain. */

    var warnFlag = false;
    const fields = filename.split(".");
    var prefix:string;
    var extension:string;
    if fields.size == 1 {
      prefix = filename;
      extension = "";
    } else {
      prefix = ".".join(fields[1..fields.size-1]);
      extension = "." + fields[fields.size];
    }
    var filenames: [0..#A.targetLocales().size] string;
    for i in 0..#A.targetLocales().size {
      filenames[i] = try! "%s_LOCALE%s%s".format(prefix, i:string, extension);
    }
    var matchingFilenames = glob(try! "%s_LOCALE*%s".format(prefix, extension));
    // if appending, make sure number of files hasn't changed and all are present
    if (mode == 1) {
      var allexist = true;
      for f in filenames {
	allexist &= try! exists(f);
      }
      if !allexist || (matchingFilenames.size != filenames.size) {
	throw new owned MismatchedAppendError();
      }
    } else { // if truncating, create new file per locale
      if matchingFilenames.size > 0 {
	warnFlag = true;
      }
      for loc in 0..#A.targetLocales().size {
	// when done with a coforall over locales, only locale 0's file gets created correctly.
	// The other locales' files have corrupted headers.
	//filenames[loc] = try! "%s_LOCALE%s%s".format(prefix, loc:string, extension);
	var file_id: C_HDF5.hid_t;
	if GenSymIO_DEBUG {
	  writeln("Creating or truncating file");
	}
	file_id = C_HDF5.H5Fcreate(filenames[loc].c_str(), C_HDF5.H5F_ACC_TRUNC, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
	if file_id < 0 { // Negative file_id means error
	  throw new owned FileNotFoundError();
	}
	C_HDF5.H5Fclose(file_id);
      } 
    }
    coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) do on loc {
	const myFilename = filenames[idx];
	if GenSymIO_DEBUG {
	  writeln(try! "%s exists? %t".format(myFilename, exists(myFilename)));
	}
	var myFileID = C_HDF5.H5Fopen(myFilename.c_str(), C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
	const locDom = A.localSubdomain();
	var dims: [0..#1] C_HDF5.hsize_t;
	dims[0] = locDom.size: C_HDF5.hsize_t;
	var myDsetName = "/" + dsetName;
	
	use C_HDF5.HDF5_WAR;
	H5LTmake_dataset_WAR(myFileID, myDsetName.c_str(), 1, c_ptrTo(dims),
			     getHDF5Type(A.eltType), c_ptrTo(A.localSlice(locDom)));
	C_HDF5.H5Fclose(myFileID);
      }
    return warnFlag;
  }
    
}
