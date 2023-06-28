module MemoryMgmt {

    use Subprocess;
    use Memory.Diagnostics;
    use Logging;
    
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

    proc getArkoudaPid() : string throws {
        var pid = spawn(["pgrep","arkouda"], stdout=pipeStyle.pipe);

        var pid_string:string;
        var line:string;

        while pid.stdout.readLine(line) {
            pid_string = line.strip();
        }

        pid.close();
        return pid_string;
    }

    proc getArkoudaMemAlloc() : uint(64) throws {
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
        var aFile = open('/proc/meminfo', ioMode.r);
        var lines = aFile.reader().lines();
        var line : string;

        var memAvail:uint(64);

        for line in lines do {
            if line.find('MemAvailable:') >= 0 {
                var splits = line.split('MemAvailable:');
                memAvail = splits[1].strip().strip(' kB'):uint(64);
            }
        }

        return (AutoMath.round(availableMemoryPct/100 * memAvail)*1000):uint(64);
    }

    proc localeMemAvailable(reqMemory) : bool throws {
        var arkMemAlloc = getArkoudaMemAlloc();
        var arkMemUsed = memoryUsed();
        var availMemory = getAvailMemory();

        mmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                 "reqMemory: %i arkMemAlloc: %i arkMemUsed: %i availMemory: %i".format(reqMemory,
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

    proc main() {
      try {
          writeln("is mem available: %t".format(isMemAvailable(10000000)));
      } catch e: Error{
          try! writeln("error: %t".format(e));
      }
   }
}