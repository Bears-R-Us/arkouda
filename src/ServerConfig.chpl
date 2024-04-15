/* arkouda server config param and config const */
module ServerConfig
{
    use ZMQ only;
    use HDF5.C_HDF5 only H5get_libversion;
    use SymArrayDmapCompat only makeDistDom;

    public use IO;
    public use ArkoudaIOCompat;

    use ServerErrorStrings;
    use Reflection;
    use ServerErrors;
    use Logging;
    use MemoryMgmt;
    use CTypes;
    import NumPyDType.DType;
    use Math;

    use ArkoudaFileCompat;
    use ArkoudaMathCompat;
    private use ArkoudaCTypesCompat;

    import BigInteger.bigint;
    
    enum Deployment {STANDARD,KUBERNETES}

    enum ObjType {
      UNKNOWN=-1,
      ARRAYVIEW=0,
      PDARRAY=1,
      STRINGS=2,
      SEGARRAY=3,
      CATEGORICAL=4,
      GROUPBY=5,
      DATAFRAME=6,
      DATETIME=7,
      TIMEDELTA=8,
      IPV4=9,
      BITVECTOR=10,
      SERIES=11,
      INDEX=12,
      MULTIINDEX=13,
    };

    /*
      maximum array dimensionality supported by the server
      set by 'serverModuleGen.py' based on 'serverConfig.json'
    */
    config param MaxArrayDims: int = 1;

    /*
      Scalar (numpy) data types supported by the server
      set by 'serverModuleGen.py' based on 'serverConfig.json'
    */
    config param  SupportsUint8 = true,
                  SupportsUint16 = false,
                  SupportsUint32 = false,
                  SupportsUint64 = true,
                  SupportsInt8 = false,
                  SupportsInt16 = false,
                  SupportsInt32 = false,
                  SupportsInt64 = true,
                  SupportsFloat32 = false,
                  SupportsFloat64 = true,
                  SupportsComplex64 = false,
                  SupportsComplex128 = false,
                  SupportsBool = true;

    proc isSupportedType(type t) param: bool {
      if t == uint(8) then return SupportsUint8;
      if t == uint(16) then return SupportsUint16;
      if t == uint(32) then return SupportsUint32;
      if t == uint then return SupportsUint64;
      if t == int(8) then return SupportsInt8;
      if t == int(16) then return SupportsInt16;
      if t == int(32) then return SupportsInt32;
      if t == int(64) then return SupportsInt64;
      if t == real(32) then return SupportsFloat32;
      if t == real then return SupportsFloat64;
      if t == complex(64) then return SupportsComplex64;
      if t == complex(128) then return SupportsComplex128;
      if t == bool then return SupportsBool;
      if t == string then return true;
      if t == bytes then return true;
      if t == bigint then return true;
      return false;
    }

    proc isSupportedDType(dt: DType): bool {
      if dt == DType.UInt8 then return SupportsUint8;
      if dt == DType.UInt16 then return SupportsUint16;
      if dt == DType.UInt32 then return SupportsUint32;
      if dt == DType.UInt64 then return SupportsUint64;
      if dt == DType.Int8 then return SupportsInt8;
      if dt == DType.Int16 then return SupportsInt16;
      if dt == DType.Int32 then return SupportsInt32;
      if dt == DType.Int64 then return SupportsInt64;
      if dt == DType.Float32 then return SupportsFloat32;
      if dt == DType.Float64 then return SupportsFloat64;
      if dt == DType.Complex64 then return SupportsComplex64;
      if dt == DType.Complex128 then return SupportsComplex128;
      if dt == DType.Bool then return SupportsBool;
      if dt == DType.Strings then return true;
      if dt == DType.BigInt then return true;
      return false;
    }

    /*
    Type of deployment, which currently is either STANDARD, meaning
    that Arkouda is deployed bare-metal or within an HPC environment, 
    or on Kubernetes, defaults to Deployment.STANDARD
    */
    config const deployment = Deployment.STANDARD;

    /*
    Trace logging flag
    */
    config const trace = true;

    /*
    Global log level flag that defaults to LogLevel.INFO
    */
    config var logLevel = LogLevel.INFO;
    
    /*
    Global log channel flag that defaults to LogChannel.CONSOLE
    */
    config var logChannel = LogChannel.CONSOLE;
    
    /*
    Indicates whether arkouda_server commands should be logged.
    */
    config var logCommands = false;

    /*
    Port for zeromq
    */
    config const ServerPort = 5555;

    /*
    Memory usage limit -- percentage of physical memory
    */
    config const perLocaleMemLimit = 90;

    /*
    Bit width of digits for the LSD radix sort and related ops
     */
    config param RSLSD_bitsPerDigit = 16;
    
    /*
    Arkouda version
    */
    config param arkoudaVersion:string = "Please set during compilation";

    /*
    Write the server `hostname:port` to this file.
    */
    config const serverConnectionInfo: string = try! getEnv("ARKOUDA_SERVER_CONNECTION_INFO", "");

    /*
    Flag to shut down the arkouda server automatically when the client disconnects
    */
    config const autoShutdown = false;

    /*
    Flag to print the server information on startup
    */
    config const serverInfoNoSplash = false;

    /*
    Hostname where I am running
    */
    var serverHostname: string = try! get_hostname();

    proc get_hostname(): string {
      return here.hostname;
    }

    /*
     * Retrieves the hostname of the locale 0 arkouda_server process, which is useful for 
     * registering Arkouda with cloud environments such as Kubernetes.
     */
    proc getConnectHostname() throws {
        var hostname: string;
        on Locales[0] {
            hostname = here.name.strip('-0');
        }
        return hostname;
    }

    /*
     * Returns the version of Chapel arkouda was built with
     */
    proc getChplVersion() throws {
        use Version;
        // Prior to 1.28, chplVersion had a prepended version that has
        // since been removed
        return (chplVersion:string).replace('version ', '');
    }

    /*
    Indicates the version of Chapel Arkouda was built with
    */
    const chplVersionArkouda = try! getChplVersion();

    /*
    Indicates whether token authentication is being used for Akrouda server requests
    */
    config const authenticate : bool = false;

    /*
    Determines the maximum number of capture groups returned by Regex.matches
    */
    config param regexMaxCaptures = 20;

    config const saveUsedModules : bool = false,
                 usedModulesFmt : string = "cfg";

    private config const lLevel = ServerConfig.logLevel;
    
    private config const lChannel = ServerConfig.logChannel;

    const scLogger = new Logger(lLevel,lChannel);
   
    proc createConfig() {
        use ArkoudaCTypesCompat;

        class LocaleConfig {
            const id: int;
            const name: string;
            const numPUs: int;
            const maxTaskPar: int;
            const physicalMemory: int;

            proc init(id: int) {
                on Locales[id] {
                    this.id = here.id;
                    this.name = here.name;
                    this.numPUs = here.numPUs();
                    this.maxTaskPar = here.maxTaskPar;
                    this.physicalMemory = getPhysicalMemHere();
                }
            }
        }

        class Config {
            const arkoudaVersion: string;
            const chplVersion: string;
            const ZMQVersion: string;
            const HDF5Version: string;
            const serverHostname: string;
            const ServerPort: int;
            const numLocales: int;
            const numPUs: int;
            const maxTaskPar: int;
            const physicalMemory: int;
            const distributionType: string;
            const LocaleConfigs: [LocaleSpace] owned LocaleConfig;
            const authenticate: bool;
            const logLevel: LogLevel;
            const logChannel: LogChannel;
            const regexMaxCaptures: int;
            const byteorder: string;
            const autoShutdown: bool;
            const serverInfoNoSplash: bool;
        }

        var (Zmajor, Zminor, Zmicro) = ZMQ.version;
        var H5major: c_uint, H5minor: c_uint, H5micro: c_uint;
        H5get_libversion(H5major, H5minor, H5micro);
        
        const cfg = new owned Config(
            arkoudaVersion = (ServerConfig.arkoudaVersion:string),
            chplVersion = chplVersionArkouda,
            ZMQVersion = try! "%i.%i.%i".doFormat(Zmajor, Zminor, Zmicro),
            HDF5Version = try! "%i.%i.%i".doFormat(H5major, H5minor, H5micro),
            serverHostname = serverHostname,
            ServerPort = ServerPort,
            numLocales = numLocales,
            numPUs = here.numPUs(),
            maxTaskPar = here.maxTaskPar,
            physicalMemory = getPhysicalMemHere(),
            distributionType = (makeDistDom(10).type):string,
            LocaleConfigs = [loc in LocaleSpace] new owned LocaleConfig(loc),
            authenticate = authenticate,
            logLevel = logLevel,
            logChannel = logChannel,
            regexMaxCaptures = regexMaxCaptures,
            byteorder = try! getByteorder(),
            autoShutdown = autoShutdown,
            serverInfoNoSplash = serverInfoNoSplash
        );
        return try! formatJson(cfg);

    }
    private var cfgStr = createConfig();

    proc getConfig(): string {
        return cfgStr;
    }

    proc getEnv(name: string, default=""): string throws {
        use OS.POSIX;
        var envBytes = getenv(name.localize().c_str());
        var val = string.createCopyingBuffer(envBytes:c_string_ptr);
        if envBytes == nil || val.isEmpty() { val = default; }
        return val;
    }

    /*
    Get an estimate for how much memory can be allocated. Based on runtime with
    chpl_comm_regMemHeapInfo if using a fixed heap, otherwise physical memory
    */ 
    proc getPhysicalMemHere() {
        use ArkoudaMemDiagnosticsCompat, ArkoudaCTypesCompat;
        extern proc chpl_comm_regMemHeapInfo(start: c_ptr(c_ptr_void), size: c_ptr(c_size_t)): void;
        var unused: c_ptr_void;
        var heap_size: c_size_t;
        chpl_comm_regMemHeapInfo(c_ptrTo(unused), c_ptrTo(heap_size));
        if heap_size != 0 then
            return heap_size.safeCast(int);
        return here.physicalMemory(unit = MemUnits.Bytes);
    }

    /*
    Get the byteorder (endianness) of this locale
    */
    proc getByteorder() throws {
      use ArkoudaIOCompat;
      return getByteOrderCompat();
    }

    /*
    Get the memory used on this locale
    */
    proc getMemUsed() {
        use ArkoudaMemDiagnosticsCompat;
        return memoryUsed();
    }

    /*
    Get the memory limit for this server run
    returns either the memMax if set or a percentage of the physical memory per locale
    */
    proc getMemLimit():uint {
        if memMax:int > 0 {
            return memMax:uint;
        } else {
            return ((perLocaleMemLimit:real / 100.0) * getPhysicalMemHere()):uint; // checks on locale-0
        }
    }

    var memHighWater:uint = 0;
    
    /*
    check used + amount is over the memory limit
    throw error if we would go over the limit
    */
    proc overMemLimit(additionalAmount:int) throws {
        // must set config var "-smemTrack=true"(compile time) or "--memTrack=true" (run time)
        // to use memoryUsed() procedure from Chapel's Memory module
        proc checkStaticMemoryLimit(total: real) {
            if total > getMemLimit() {
                var pct = mathRound((total:real / getMemLimit():real * 100):uint);
                var msg = "cmd requiring %i bytes of memory exceeds %i limit with projected pct memory used of %i%%".doFormat(
                                   total * numLocales, getMemLimit() * numLocales, pct);
                scLogger.error(getModuleName(),getRoutineName(),getLineNumber(), msg);  
                throw getErrorWithContext(
                          msg=msg,
                          lineNumber=getLineNumber(),
                          routineName=getRoutineName(),
                          moduleName=getModuleName(),
                          errorClass="OverMemoryLimitError");
            }        
        }
        
        if (memTrack) {
            // this is a per locale total
            var total = getMemUsed() + (additionalAmount:uint / numLocales:uint);
            if (trace) {
                if (total > memHighWater) {
                    memHighWater = total;
                    scLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                    "memory high watermark = %i memory limit = %i projected pct memory used of %i%%".doFormat(
                           memHighWater:uint * numLocales:uint, 
                           getMemLimit():uint * numLocales:uint,
                           mathRound((memHighWater:real * numLocales / 
                                         (getMemLimit():real * numLocales)) * 100):uint));
                }
            }
            
            /*
             * If the MemoryMgmt.memMgmtType is STATIC (default), use the memory management logic based upon
             * a percentage of the locale0 host machine physical memory. 
             *
             * If DYNAMIC, use the new dynamic memory mgmt capability in the MemoryMgmt module that first determines 
             * for each locale if there's sufficient space within the memory currently allocated to the Arkouda 
             * Chapel process to accommodate the projected memory required by the cmd. If not, then MemoryMgmt 
             * checks the available memory on each locale to see if more can be allocated to the Arkouda-Chapel process.
             * If the answer is no on any locale, the cmd is not executed and MemoryMgmt logs the corresponding locales
             * server-side. More detailed client-side reporting can be implemented in a later version. 
             */
            if memMgmtType == MemMgmtType.STATIC {
                if total > getMemLimit() {
                    var pct = mathRound((total:real / getMemLimit():real * 100):uint);
                    var msg = "cmd requiring %i bytes of memory exceeds %i limit with projected pct memory used of %i%%".doFormat(
                                   total * numLocales, getMemLimit() * numLocales, pct);
                    scLogger.error(getModuleName(),getRoutineName(),getLineNumber(), msg);  
                    throw getErrorWithContext(
                              msg=msg,
                              lineNumber=getLineNumber(),
                              routineName=getRoutineName(),
                              moduleName=getModuleName(),
                              errorClass="OverMemoryLimitError");
                }
            } else {
                if !isMemAvailable(additionalAmount) {
                    var msg = "cmd requiring %i more bytes of memory exceeds available memory on one or more locales".doFormat(
                                                                                                     additionalAmount);
                    scLogger.error(getModuleName(),getRoutineName(),getLineNumber(), msg);  
                    throw getErrorWithContext(
                              msg=msg,
                              lineNumber=getLineNumber(),
                              routineName=getRoutineName(),
                              moduleName=getModuleName(),
                              errorClass="OverMemoryLimitError");
                }
            }
        }
    }

    proc string.splitMsgToTuple(param numChunks: int) {
      var tup: numChunks*string;
      var count = tup.indices.low;

      // fill in the initial tuple elements defined by split()
      for s in this.split(numChunks-1) {
        tup(count) = s;
        count += 1;
      }
      // if split() had fewer items than the tuple, fill in the rest
      if (count < numChunks) {
        for i in count..numChunks-1 {
          tup(i) = "";
        }
      }
      return tup;
    }

    proc string.splitMsgToTuple(sep: string, param numChunks: int) {
      var tup: numChunks*string;
      var count = tup.indices.low;

      // fill in the initial tuple elements defined by split()
      for s in this.split(sep, numChunks-1) {
        tup(count) = s;
        count += 1;
      }
      // if split() had fewer items than the tuple, fill in the rest
      if (count < numChunks) {
        for i in count..numChunks-1 {
          tup(i) = "";
        }
      }
      return tup;
    }

    proc bytes.splitMsgToTuple(param numChunks: int) {
      var tup: numChunks*bytes;
      var count = tup.indices.low;

      // fill in the initial tuple elements defined by split()
      for s in this.split(numChunks-1) {
        tup(count) = s;
        count += 1;
      }
      // if split() had fewer items than the tuple, fill in the rest
      if (count < numChunks) {
        for i in count..numChunks-1 {
          tup(i) = b"";
        }
      }
      return tup;
    }

    proc bytes.splitMsgToTuple(sep: bytes, param numChunks: int) {
      var tup: numChunks*bytes;
      var count = tup.indices.low;

      // fill in the initial tuple elements defined by split()
      for s in this.split(sep, numChunks-1) {
        tup(count) = s;
        count += 1;
      }
      // if split() had fewer items than the tuple, fill in the rest
      if (count < numChunks) {
        for i in count..numChunks-1 {
          tup(i) = b"";
        }
      }
      return tup;
    }

    proc getEnvInt(name: string, default: int): int {
      extern proc getenv(name) : c_string_ptr;
      var strval:string;
      try! strval = string.createCopyingBuffer(getenv(name.localize().c_str()));
      if strval.isEmpty() { return default; }
      return try! strval: int;
    }

    /*
     * String constants for use in constructing JSON formatted messages
     */
    const Q = '"'; // Double Quote, escaping quotes often throws off syntax highlighting.
    const QCQ = Q + ":" + Q; // `":"` -> useful for closing and opening quotes for named json k,v pairs
    const BSLASH = '\\';
    const ESCAPED_QUOTES = BSLASH + Q;

    proc appendToConfigStr(key:string, val:string) {
      var idx_close = cfgStr.rfind("}"):int;
      var tmp_json = cfgStr(0..idx_close-1);
      cfgStr = tmp_json + "," + Q + key + QCQ + val + Q + "}";
    }
}
