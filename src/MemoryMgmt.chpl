module MemoryMgmt {

    use Subprocess;
    use FileSystem;
    use Reflection;
    use Math;

    use Logging;
    use ServerErrors;
    use MemDiagnostics;

    private config const logLevel = LogLevel.DEBUG;
    private config const logChannel = LogChannel.CONSOLE;
    const mmLogger = new Logger(logLevel,logChannel);

    /*
     * Indicates whether static locale host memory or dynamically-captured
     * memory allocated to the Arkouda process is used to estimate whether
     * sufficient memory is available to execute the requested command.
     */
    enum MemMgmtType {STATIC,DYNAMIC}

    /*
     * The percentage of currently available memory on each locale host 
     * that is the limit for memory allocated to each Arkouda locale.
     */
    config const availableMemoryPct: real = 90;

    /*
     * Config param that indicates whether the static memory mgmt logic in
     * ServerConfig or dynamic memory mgmt in this module will be used. The
     * default is static memory mgmt.
     */
    config const memMgmtType = MemMgmtType.STATIC;

    record LocaleMemoryStatus {
        var total_mem: uint(64);
        var avail_mem: uint(64);
        var pct_avail_mem: int;
        var arkouda_mem_alloc: uint(64);
        var mem_used: uint(64);
        var locale_id: int;
        var locale_hostname: string;
    }

    proc isSupportedOS() : bool  throws {
        return exists('/proc/meminfo');
    }

    proc getArkoudaPid() : string throws {
        if !isSupportedOS() {
            throw new owned ErrorWithContext("getArkoudaPid can only be invoked on Unix and Linux systems",
                                           getLineNumber(),
                                           getRoutineName(),
                                           getModuleName(),
                                           "UnsupportedOSError");
        }

        var pid = spawn(["pgrep","arkouda_server"], stdout=pipeStyle.pipe);

        var pid_string:string;
        var line:string;

        while pid.stdout.readLine(line) {
            pid_string = line.strip();
        }

        pid.close();
        return pid_string;
    }

    proc getArkoudaMemAlloc() : uint(64) throws {
        if !isSupportedOS() {
            throw new owned ErrorWithContext("getArkoudaMemAlloc can only be invoked on Unix and Linux systems",
                                           getLineNumber(),
                                           getRoutineName(),
                                           getModuleName(),
                                           "UnsupportedOSError");
        }

        var pid = getArkoudaPid();

        var sub = spawn(["pmap", pid], stdout=pipeStyle.pipe);

        var malloc_string:string;
        var line:string;

        while sub.stdout.readLine(line) {
            if line.find("total") > 0 {
                var splits = line.split('total');
                malloc_string = splits(1).strip().strip('K');
            }
        }

        sub.close();
        return malloc_string:uint(64) * 1000;
    }
    
    proc getAvailMemory() : uint(64) throws {
        if !isSupportedOS() {
            throw new owned ErrorWithContext("getAvailMemory can only be invoked on Unix and Linux systems",
                                           getLineNumber(),
                                           getRoutineName(),
                                           getModuleName(),
                                           "UnsupportedOSError");
        }

        var lines = openReader('/proc/meminfo').lines();
        var line : string;

        var memAvail:uint(64);

        for line in lines do {
            if line.find('MemAvailable:') >= 0 {
                var splits = line.split('MemAvailable:');
                memAvail = splits[1].strip().strip(' kB'):uint(64);
                break;
            }
        }

        return (Math.round(availableMemoryPct/100 * memAvail)*1000):uint(64);
    }
    
    proc getTotalMemory() : uint(64) throws {
        if !isSupportedOS() {
            throw new owned ErrorWithContext("getTotalMemory can only be invoked on Unix and Linux systems",
                                           getLineNumber(),
                                           getRoutineName(),
                                           getModuleName(),
                                           "UnsupportedOSError");
        }

        var lines = openReader('/proc/meminfo').lines();
        var line : string;

        var totalMem:uint(64);

        for line in lines do {
            if line.find('MemTotal:') >= 0 {
                var splits = line.split('MemTotal:');
                totalMem = splits[1].strip().strip(' kB'):uint(64);
                break;
            }
        }

        return totalMem*1000:uint(64);
    }
    
    proc getLocaleMemoryStatuses() throws {
        var memStatuses: [0..numLocales-1] LocaleMemoryStatus;
        
        coforall loc in Locales with (ref memStatuses) {
            on loc {
                var availMem = getAvailMemory();
                var totalMem = getTotalMemory();
                var pctAvailMem = (availMem:real/totalMem)*100:int;

                memStatuses[here.id] = new LocaleMemoryStatus(total_mem=totalMem,
                                                              avail_mem=availMem,
                                                              pct_avail_mem=pctAvailMem:int,
                                                              arkouda_mem_alloc=getArkoudaMemAlloc(),
                                                              mem_used=memoryUsed(),
                                                              locale_id=here.id,
                                                              locale_hostname=here.hostname);                                      
            }
        }
        return memStatuses;
    }

    proc localeMemAvailable(reqMemory) : bool throws {
        var arkMemAlloc = getArkoudaMemAlloc();
        var arkMemUsed = memoryUsed();
        var availMemory = getAvailMemory();

        mmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                 "locale: %s reqMemory: %i arkMemAlloc: %i arkMemUsed: %i availMemory: %i".format(here.id,
                                                                                                  reqMemory,
                                                                                                  arkMemAlloc,
                                                                                                  arkMemUsed,
                                                                                                  availMemory));
        var newArkoudaMemory = reqMemory:int + arkMemUsed:int;
        
        if newArkoudaMemory:int <= arkMemAlloc:int {
            return true;
        } else {
            if newArkoudaMemory:int <= availMemory {
                return true;
            } else {
                var msg = "Arkouda memory request %i on locale %s exceeds available memory %i".format(newArkoudaMemory,
                                                                                                      here.id,
                                                                                                      availMemory);
                mmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),msg);
                return false;
            }
        }
    }
    
    /*
     * Returns a boolean indicating whether there is either sufficient memory within the
     * memory allocated to Arkouda on each locale host; if true for all locales, returns true. 
     * 
     * If the reqMemory exceeds the memory currently allocated to at least one locale, each locale 
     * host is checked to see if there is memory available to allocate more memory to each
     * corresponding locale. If there is insufficient memory available on at least one locale,
     * returns false. If there is sufficient memory on all locales to allocate sufficient, 
     * additional memory to Arkouda to execute the command, returns true.
     */
    proc isMemAvailable(reqMemory) : bool throws {
      var overMemLimit : bool = false;

      coforall loc in Locales with (ref overMemLimit) {
        on loc {
            if !localeMemAvailable(reqMemory) {
                overMemLimit = true;
            }
        }
      }
      
      return if overMemLimit then false else true;
    }
}
