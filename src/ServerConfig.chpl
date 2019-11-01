/* arkouda server config param and config const */
module ServerConfig
{
    /*
    Verbose flag
    */
    config const v = true;

    /*
    Port for zeromq
    */
    config const ServerPort = 5555;

    /* 
    Arkouda version
    */
    config param arkoudaVersion = "0.0.9-2019-09-23";

    /*
    Configure MyDmap on compile line by "-s MyDmap=0" or "-s MyDmap=1"
    0 = Cyclic, 1 = Block. Cyclic may not work; we haven't tested it in a while.
    BlockDist is the default.
    */
    config param MyDmap = 1;

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
        
        use Memory;

        class LocaleConfig {
            var id: int;
            var name: string;
            var numPUs: int;
            var maxTaskPar: int;
            var physicalMemory: int;
        }
        class Config {
            var serverHostname: string;
            var ServerPort: int;
            var numLocales: int;
            var numPUs: int;
            var maxTaskPar: int;
            var physicalMemory: int;
            var LocaleConfigs: [LocaleSpace] owned LocaleConfig =
                        [loc in LocaleSpace] new owned LocaleConfig();
        }
        var cfg = new owned Config();

        cfg.serverHostname = serverHostname;
        cfg.ServerPort = ServerPort;
        cfg.numLocales = numLocales;
        cfg.numPUs = here.numPUs();
        cfg.maxTaskPar = here.maxTaskPar;
        cfg.physicalMemory = here.physicalMemory();
        
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
    
}
