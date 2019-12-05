/* arkouda server config param and config const */
module ServerConfig
{
    use Memory;
    
    use ZMQ only;
    use HDF5.C_HDF5 only H5get_libversion;
    use SymArrayDmap only makeDistDom;

    public use IO;

    use ServerErrorStrings;
    
    /*
    Verbose flag
    */
    config const v = true;

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
    Hostname where I am running 
    */ 
    var serverHostname: string = try! get_hostname();

    proc get_hostname(): string {
      // Want:
      //   return here.hostname;
      // but this isn't implemented yet; could use:
      //   return here.name;
      // but this munges the hostname when using local spawning with GASNet
      // so the following is used as a temporary workaround:
      extern proc chpl_nodeName(): c_string;
      var hostname = chpl_nodeName(): string;
      return hostname;
    }

    proc getConfig(): string {
        
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

    /*
    Get the memory limit for this server run
    returns a percentage of the physical memory per locale
    */
    proc getMemLimit():uint {
        return ((perLocaleMemLimit:real / 100.0) * here.physicalMemory()):uint; // checks on locale-0
    }

    /*
    check used + amount is over the memory limit
    throw error if we would go over the limit
    */
    proc overMemLimit(additionalAmount:int) throws {
        // must set config var "-smemTrack=true"(compile time) or "--memTrack=true" (run time)
        // to use memoryUsed() procedure from Chapel's Memory module
        if (memTrack) {
            var total = memoryUsed() + (additionalAmount:uint / numLocales:uint); // this is a per locale total
            if total > getMemLimit() {
                throw new owned ErrorWithMsg("Error: Operation would exceed memory limit ("
                                             +total:string+","+getMemLimit():string+")");
            }
        }
    }

}
