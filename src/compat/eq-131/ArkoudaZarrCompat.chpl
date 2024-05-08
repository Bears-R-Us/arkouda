module ArkoudaZarrCompat {
  use IO;
  use FileSystem;
  use JSON;
  use Map;
  use List;
  use Path;
  use CTypes;
  use BlockDist;
  use Time;

  require "blosc.h";
  require "-lblosc";

  private module Blosc {
    use CTypes;
    extern proc blosc_init();
    extern proc blosc_compress(clevel: c_int, doshuffle: c_int, typesize: c_size_t,
                              nbytes: c_size_t, src: c_ptrConst(void),
                              dest: c_ptr(void), destsize: c_size_t): int;
    extern proc blosc_decompress(src: c_ptrConst(void), dest: c_ptr(void), destsize: c_size_t): int;
    extern proc blosc_destroy();
    extern proc blosc_set_nthreads(nthreads_new: c_int) : c_int;
    extern proc blosc_get_nthreads() : c_int;
  }
  private use Blosc;

  /* Turns on/off profiling of Zarr IO */
  config param zarrProfiling = false;

  private var timerDomain: domain(string,parSafe=false) = {"Compression",
    "Decompression", "Opening File, Read", "Opening File, Write",
    "Creating Reader", "Reading File", "Creating Writer", "Writing File",
    "Reading to Update", "Copying In", "Creating Compressed Buffer"};
  private var times: [timerDomain] atomic real;

  /*
    Returns a map of profiling results for Zarr IO operations. The keys are
    the names of the operations and the values are the total time spent in
    each operation across all threads. Requires that zarrProfiling be set to
    true.
  */
  iter zarrProfilingResults() throws {
    for key in times.keys() do yield (key, times[key].read());
  }

  

  record zarrMetadataV2 {
    var zarr_format: int;
    var chunks: list(int);
    var dtype: string;
    var shape: list(int);
  }

  /* Unused until support is added for v3.0 stores */
  record zarrMetadataV3 {
    var zarr_format: int;
    var node_type: string;
    var shape: list(int);
    var data_type: string;
    var dimension_names: list(string);
  }

  private proc dtypeString(type dtype) throws {
    select dtype {
      when real(32) do return "f4";
      when real(64) do return "f8";
      when int(32) do return "i4";
      when int(64) do return "i8";
    }
    throw Error("Unexpected data type, only real and int types are supported.");
  }

  private proc getMetadata(directoryPath: string) throws {
    var metadataPath = joinPath(directoryPath, ".zarray");
    var r = openReader(metadataPath, deserializer = new jsonDeserializer(), locking=false);
    var md: zarrMetadataV2;
    r.readf("%?", md);
    return md;
  }

  private proc validateMetadata(metadata: zarrMetadataV2, type dtype, param dimCount) throws {
    //dimensionality matches
    if dimCount != metadata.shape.size then
      throw new Error("Expected metadata shape field to have %i dimensions: %?".format(dimCount, metadata.shape));
    if dimCount != metadata.chunks.size then
      throw new Error("Expected metadata chunks field to have %i dimensions: %?".format(dimCount, metadata.chunks));
    //positive, integer sizes
    for i in 0..<dimCount {
      if metadata.shape[i] <= 0 then
        throw new Error("Metadata shape field must have positive side lengths: %?".format(metadata.shape));
      if metadata.chunks[i] <= 0 then
        throw new Error("Metadata chunks field must have positive side lengths: %?".format(metadata.chunks));
    }

    var chplType: string;
    select metadata.dtype {
      when "i4", "<i4" do chplType = "int(32)";
      when "i8", "<i8" do chplType = "int(64)";
      when "f4", "<f4" do chplType = "real(32)";
      when "f8", "<f8" do chplType = "real(64)";
      otherwise {
        throw new Error("Only integer and floating point data types currently supported: %s".format(metadata.dtype));
      }
    }

    if chplType != dtype:string then
      throw new Error("Expected entries of type %s. Found %s".format(dtype:string, chplType));
  }


  private proc buildChunkPath(directoryPath: string, delimiter: string, const chunkIndices: ?dimCount * int) {
    var indexStrings: dimCount*string;
    for i in 0..<dimCount do indexStrings[i] = chunkIndices[i] : string;
    return joinPath(directoryPath, delimiter.join(indexStrings));
  }
  private proc buildChunkPath(directoryPath: string, delimiter: string, chunkIndex: int) {
    return joinPath(directoryPath, chunkIndex:string);
  }

  /* Returns the domain of chunks that the calling locale is responsible for */
  proc getLocalChunks(D: domain(?), localD: domain(?), chunkShape: ?dimCount*int): domain(dimCount) {

    const totalShape = D.shape;
    var chunkCounts: dimCount*int;
    for i in 0..<dimCount {
      chunkCounts[i] = ceil(totalShape[i]:real / chunkShape[i]: real) : int;
    }

    var localChunks: dimCount*range(int);
    for i in 0..<dimCount {
      var l = if dimCount != 1 then localD.low[i] else localD.low;
      var h = if dimCount != 1 then localD.high[i] else localD.high;
      var low = floor(l:real / chunkShape[i]:real):int;
      var high = floor(h / chunkShape[i]:real):int;
      localChunks[i] = max(low,0)..min(high,chunkCounts[i]-1);
    }
    const localChunkDomain: domain(dimCount) = localChunks;
    return localChunkDomain;
  }


  /* Returns the domain of the `chunkIndices`-th chunk for chunks of size `chunkShape` */
  private proc getChunkDomain(chunkShape: ?dimCount*int, chunkIndices: dimCount*int) {
    var thisChunkRange: dimCount*range(int);
    for i in 0..<dimCount {
      const start = chunkIndices[i] * chunkShape[i];
      thisChunkRange[i] = start..<start+chunkShape[i];
    }
    const thisChunkDomain: domain(dimCount) = thisChunkRange;
    return thisChunkDomain;
  }
  private proc getChunkDomain(chunkShape: ?dimCount*int, chunkIndices: int) {
    return getChunkDomain(chunkShape, (chunkIndices,));
  }


  /*
    Reads a chunk from storage and fills `arraySlice` with its corresponding
    values.

    :arg dimCount: Dimensionality of the array being read.

    :arg chunkPath: Relative or absolute path to the chunk being read.

    :arg chunkDomain: Array subdomain the chunk contains.

    :arg arraySlice: Reference to the portion of the array the calling locale stores.

    :throws Error: If the decompression fails
  */
  proc readChunk(param dimCount: int, chunkPath: string, chunkDomain: domain(dimCount), ref arraySlice: [] ?t) throws {
    var s: stopwatch;
    const f: file;
    // if the file does not exist, the chunk is empty
    if zarrProfiling then s.restart();
    try {
      f = open(chunkPath, ioMode.r);
    } catch {
      arraySlice = 0;
      return;
    }
    if zarrProfiling then times["Opening File, Read"].add(s.elapsed());

    if zarrProfiling then s.restart();
    const r = f.reader(deserializer = new binaryDeserializer(), locking=false);
    if zarrProfiling then times["Creating Reader"].add(s.elapsed());

    if zarrProfiling then s.restart();
    const compressedChunk = r.readAll(bytes); // TODO: stream straight through to blosc
    var readBytes = compressedChunk.size;
    if zarrProfiling then times["Reading File"].add(s.elapsed());
    if zarrProfiling then s.restart();
    var copyIn: [chunkDomain] t;
    var numRead = blosc_decompress(compressedChunk.c_str(), c_ptrTo(copyIn), copyIn.size*c_sizeof(t));
    if numRead <= 0 {
      throw new Error("Failed to decompress data from %?. Blosc error code: %?".format(chunkPath, numRead));
    }
    if zarrProfiling then times["Decompression"].add(s.elapsed());

    if zarrProfiling then s.restart();
    arraySlice = copyIn[arraySlice.domain];
    if zarrProfiling then times["Copying In"].add(s.elapsed());
  }

  /*
    Updates a chunk in storage with a locale's contribution to that chunk.
    The calling function is expected to manage synchronization among locales.
    If the locale contributes the entire chunk, it will immediately compress
    and write the chunk's data. If the contribution is partial, it decompresses
    the chunk, updates the necessary values, then compresses and writes the
    chunk to storage.

    :arg dimCount: Dimensionality of the array being written.

    :arg chunkPath: Relative or absolute path to the chunk being written.

    :arg chunkDomain: Array subdomain that the chunk contains.

    :arg arraySlice: The portion of the array that the calling locale
      contributes to this chunk.

    :arg bloscLevel: Compression level to use. 0 indicates no compression,
      9 (default) indicates maximum compression. Values outside of this range
      will be clipped to a value between 0 and 9.

    :throws Error: If the compression fails
  */
  proc writeChunk(param dimCount, chunkPath: string, chunkDomain: domain(dimCount), ref arraySlice: [] ?t, bloscLevel: int(32) = 9) throws {
    var s: stopwatch;

    //bloscLevel must be between 0 and 9
    var _bloscLevel = min(9,max(0,bloscLevel));

    // If this chunk is entirely contained in the array slice, we can write
    // it out immediately. Otherwise, we need to read in the chunk and update
    // it with the partial data before writing
    if zarrProfiling then s.restart();
    var copyOut: [chunkDomain] t;
    if (chunkDomain != arraySlice.domain) {
      readChunk(dimCount, chunkPath, chunkDomain, copyOut);
    }
    copyOut[arraySlice.domain] = arraySlice;
    if zarrProfiling then times["Reading to Update"].add(s.elapsed());


    // Create buffer for compressed bytes
    if zarrProfiling then s.restart();
    var compressedBuffer = allocate(t, copyOut.size + 16);
    if zarrProfiling then times["Creating Compressed Buffer"].add(s.elapsed());

    // Compress the chunk's data
    if zarrProfiling then s.restart();
    var bytesCompressed = blosc_compress(_bloscLevel, 0, c_sizeof(t), copyOut.size*c_sizeof(t), c_ptrTo(copyOut), compressedBuffer, (copyOut.size + 16) * c_sizeof(t));
    if bytesCompressed == 0 then
      throw new Error("Failed to compress bytes");
    if zarrProfiling then times["Compression"].add(s.elapsed());

    // Write it to storage
    if zarrProfiling then s.restart();
    const f = open(chunkPath, ioMode.cw);
    if zarrProfiling then times["Opening File, Write"].add(s.elapsed());
    if zarrProfiling then s.restart();
    const w = f.writer(serializer = new binarySerializer(),locking=false);
    if zarrProfiling then times["Creating Writer"].add(s.elapsed());

    if zarrProfiling then s.restart();
    w.writeBinary(compressedBuffer: c_ptr(void),bytesCompressed);
    if zarrProfiling then times["Writing File"].add(s.elapsed());
  }

  /*
    Reads a v2.0 zarr store from storage, returning a block distributed array.
    Each locale reads and decompresses the chunks with elements in its
    subdomain. This method assumes a shared filesystem where all nodes can
    access the store directory.

    :arg directoryPath: Relative or absolute path to the root of the zarr
      store. The store is expected to contain a '.zarray' metadata file

    :arg dtype: Chapel type of the store's data

    :arg dimCount: Dimensionality of the zarr array

    :arg bloscThreads: The number of threads to use during decompression
      (default=1)
  */
  proc readZarrArray(directoryPath: string, type dtype, param dimCount: int, bloscThreads: int(32) = 1) throws {
    var md = getMetadata(directoryPath);
    validateMetadata(md, dtype, dimCount);
    // Size and shape tuples
    var totalShape, chunkShape : dimCount*int;
    var chunkCounts: dimCount*int;
    var totalRanges,chunkRanges: dimCount*range(int);
    for i in 0..<dimCount {
      totalShape[i] = md.shape[i];
      chunkShape[i] = md.chunks[i];
      chunkCounts[i] = ceil(totalShape[i]:real / chunkShape[i]:real) : int;
      totalRanges[i] = 0..<totalShape[i];
      chunkRanges[i] = 0..<chunkCounts[i];
    }
    const fullChunkDomain: domain(dimCount) = chunkRanges;

    // Initialize the distributed domain and array
    const undistD : domain(dimCount) = totalRanges;
    const Dist = new blockDist(boundingBox=undistD);
    const D = Dist.createDomain(undistD);
    var A: [D] dtype;


    coforall loc in Locales do on loc {
      blosc_init();
      blosc_set_nthreads(bloscThreads);
      const hereD = A.localSubdomain();
      ref hereA = A[hereD];

      const localChunks = getLocalChunks(D, hereD, chunkShape);
      forall chunkIndices in localChunks {

        const chunkPath = buildChunkPath(directoryPath, ".", chunkIndices);

        const thisChunkDomain = getChunkDomain(chunkShape, chunkIndices);
        const thisChunkHere = hereD[thisChunkDomain];

        ref thisChunkSlice = hereA.localSlice(thisChunkHere);
        readChunk(dimCount, chunkPath, thisChunkDomain, thisChunkSlice);
      }
      blosc_destroy();
    }
    return A;
  }

  /*
    Writes an array to storage as a v2.0 zarr store. The array metadata and
    chunks will be stored within the `directoryPath` directory, which is created
    if it does not yet exist. The chunks will have the dimensions given in the
    `chunkShape` argument. This function writes chunks in parallel, and supports
    distributed execution. It assumes a shared filesystem where all nodes can
    access the store directory.

    :arg directoryPath: Relative or absolute path to the root of the zarr store.
      The directory and all necessary parent directories will be created if it
      does not exist.

    :arg A: The array to write to storage.

    :arg chunkShape: The dimension extents to use when breaking A into chunks.

    :arg bloscThreads: The number of threads to use during compression (default=1)

    :arg bloscLevel: Compression level to use. 0 indicates no compression,
      9 (default) indicates maximum compression.
  */
  proc writeZarrArray(directoryPath: string, ref A: [?domainType] ?dtype, chunkShape: ?dimCount*int, bloscThreads: int(32) = 1, bloscLevel: int(32) = 9) throws {

    // Create the metadata record that is written before the chunks
    var shape, chunks: list(int);
    for size in A.shape do shape.pushBack(size);
    for size in chunkShape do chunks.pushBack(size);
    const md: zarrMetadataV2 = new zarrMetadataV2(2, chunks, dtypeString(dtype), shape);

    // Clear the directory before writing
    if exists(directoryPath) then rmTree(directoryPath);
    mkdir(directoryPath, parents=true);

    // Write the metadata
    const metadataPath = joinPath(directoryPath, ".zarray");
    const w = openWriter(metadataPath, serializer = new jsonSerializer(),locking=true);
    w.writef("%?\n", md);

    // Normalize the array's domain to be zero-indexed
    var normalizedRanges: dimCount*range(int);
    for i in 0..<dimCount do
      normalizedRanges[i] = 0..<shape[i];
    const undistD: domain(dimCount) = normalizedRanges;
    const dist = new blockDist(boundingBox=undistD);
    const D = dist.createDomain(undistD);
    ref normA = A.reindex(D);

    // Locks to synchronize locales writing to the same chunks
    const allChunks = getLocalChunks(normA.domain, normA.domain, chunkShape);
    var locks: [allChunks] sync bool;

    // Write the chunks
    coforall loc in Locales do on loc {
      // Initialize blosc on each locale
      blosc_init();
      blosc_set_nthreads(bloscThreads);

      // Get the part of the array that belongs to this locale
      const hereD = normA.localSubdomain();
      ref hereA = normA[hereD];

      // Identify the range of chunks this locale will contribute to
      const localChunks = getLocalChunks(normA.domain, hereD, chunkShape);

      forall chunkIndices in localChunks {
        // Get the part of the array that contributes to this chunk
        const thisChunkDomain = getChunkDomain(chunkShape, chunkIndices);
        const thisChunkHere = hereD[thisChunkDomain];
        ref thisChunkSlice = hereA.localSlice(thisChunkHere);
        const chunkPath = buildChunkPath(directoryPath, ".", chunkIndices);
        locks[chunkIndices].writeEF(true);
        writeChunk(dimCount, chunkPath, thisChunkDomain, thisChunkSlice, bloscLevel=bloscLevel);
        locks[chunkIndices].readFE();
      }
    }
  }

}