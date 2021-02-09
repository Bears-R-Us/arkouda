/* arkouda server config param and config const */
module ServerConfig
{
    use Memory;

    use ZMQ only;
    use HDF5.C_HDF5 only H5get_libversion;
    use SymArrayDmap only makeDistDom;

    public use IO;
    private use SysCTypes;

    use ServerErrorStrings;
    use Reflection;
    use Errors;
    use Logging;

    /*
    Trace logging flag
    */
    config const trace = true;

    /*
    Verbose debug flag
    */
    config const v = false;

    /*
    Port for zeromq
    */
    config const ServerPort = 5555;

    /*
    Memory usage limit -- percentage of physical memory
    */
    config const perLocaleMemLimit = 90;

    /*
    Arkouda version
    */
    config param arkoudaVersion:string;

    /*
    Write the server `hostname:port` to this file.
    */
    config const serverConnectionInfo: string = getEnv("ARKOUDA_SERVER_CONNECTION_INFO", "");

    const scLogger = new Logger();
  
    if v {
        scLogger.level = LogLevel.DEBUG;
    } else {
        scLogger.level = LogLevel.INFO;
    }

    /*
    Hostname where I am running
    */
    var serverHostname: string = try! get_hostname();

    proc get_hostname(): string {
      return here.hostname;
    }

    /*
    Indicates whether token authentication is being used for Akrouda server requests
    */
    config const authenticate : bool = false;
   
    proc getConfig(): string {
        use SysCTypes;

        class LocaleConfig {
            var id: int;
            var name: string;
            var numPUs: int;
            var maxTaskPar: int;
            var physicalMemory: int;
        }
        class Config {
            var arkoudaVersion: string;
            var ZMQVersion: string;
            var HDF5Version: string;
            var serverHostname: string;
            var ServerPort: int;
            var numLocales: int;
            var numPUs: int;
            var maxTaskPar: int;
            var physicalMemory: int;
            var distributionType: string;
            var LocaleConfigs: [LocaleSpace] owned LocaleConfig =
                [loc in LocaleSpace] new owned LocaleConfig();
            var authenticate: bool;
        }
        var (Zmajor, Zminor, Zmicro) = ZMQ.version;
        var H5major: c_uint, H5minor: c_uint, H5micro: c_uint;
        H5get_libversion(H5major, H5minor, H5micro);
        var cfg = new owned Config();
        cfg.arkoudaVersion = (ServerConfig.arkoudaVersion:string).replace("-", ".");
        cfg.ZMQVersion = try! "%i.%i.%i".format(Zmajor, Zminor, Zmicro);
        cfg.HDF5Version = try! "%i.%i.%i".format(H5major, H5minor, H5micro);
        cfg.serverHostname = serverHostname;
        cfg.ServerPort = ServerPort;
        cfg.numLocales = numLocales;
        cfg.numPUs = here.numPUs();
        cfg.maxTaskPar = here.maxTaskPar;
        cfg.physicalMemory = here.physicalMemory();
        cfg.distributionType = (makeDistDom(10).type):string;
        cfg.authenticate = authenticate; 

        for loc in Locales {
            on loc {
                cfg.LocaleConfigs[here.id].id = here.id;
                cfg.LocaleConfigs[here.id].name = here.name;
                cfg.LocaleConfigs[here.id].numPUs = here.numPUs();
                cfg.LocaleConfigs[here.id].maxTaskPar = here.maxTaskPar;
                cfg.LocaleConfigs[here.id].physicalMemory = here.physicalMemory();
            }
        }
        var res: string = try! "%jt".format(cfg);
        return res;
    }

    proc getEnv(name: string, default=""): string {
        extern proc getenv(name : c_string) : c_string;
        var val = getenv(name.localize().c_str()): string;
        if val.isEmpty() { val = default; }
        return val;
    }

    /*
    Get the memory limit for this server run
    returns a percentage of the physical memory per locale
    */
    proc getMemLimit():uint {
        return ((perLocaleMemLimit:real / 100.0) * here.physicalMemory()):uint; // checks on locale-0
    }

    var memHighWater:uint = 0;
    
    /*
    check used + amount is over the memory limit
    throw error if we would go over the limit
    */
    proc overMemLimit(additionalAmount:int) throws {
        // must set config var "-smemTrack=true"(compile time) or "--memTrack=true" (run time)
        // to use memoryUsed() procedure from Chapel's Memory module
        if (memTrack) {
            // this is a per locale total
            var total = memoryUsed() + (additionalAmount:uint / numLocales:uint);
            if (trace) {
                if (total > memHighWater) {
                    memHighWater = total;
                    scLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                    "memory high watermark = %i memory limit = %i".format(
                           memHighWater:uint * numLocales:uint, 
                           getMemLimit():uint * numLocales:uint));
                }
            }
            if total > getMemLimit() {
                var msg = "Error: Operation would exceed memory limit ("
                                             +total:string+","+getMemLimit():string+")";
                scLogger.error(getModuleName(),getRoutineName(),getLineNumber(), msg);  
                throw getErrorWithContext(
                          msg=msg,
                          lineNumber=getLineNumber(),
                          routineName=getRoutineName(),
                          moduleName=getModuleName(),
                          errorClass="ErrorWithContext");                                        
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

}
