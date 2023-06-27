module MemoryMgmt {

    use Subprocess;
    use Memory;

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

    proc getArkoudaMemAlloc() : string throws {
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
        return malloc_string;
    }
    
    proc getAvailMemory() : int throws {
        var aFile = open('/proc/meminfo', ioMode.r);
        var lines = aFile.reader().lines();
        var line : string;

        var memAvail:int;

        for line in lines do {
            if line.find('MemAvailable:') >= 0 {
                var splits = line.split('MemAvailable:');
                memAvail = splits[1].strip().strip(' kB'):int;
            }
        }
        
        return memAvail;
    }

    proc localeMemAvailable(reqMemory: int, memLimitPct: int) : bool throws {
        var arkMemAlloc = getArkoudaMemAlloc();
        var arkMemUsed = getMemory()/1000;
        var availMemory = getAvailMemory();

        if reqMemory + arkMemUsed <= arkMemAlloc {
            return true;
        } else {
            if reqMemory + arkMemUsed <= availMemory {
                return true;
            } else {
                return false;
            }
        }
    }
    
    proc memAvailable(reqMemory: int, memLimitPct: int) : bool throws {
      var overMemLimit : bool = false;

      coforall loc in Locales {
        on loc {
            if !localeMemAvailable(reqMemory, memLimitPct) {
                overMemLimit = true;
            }
        }
      }
      
      return if overMemLimit then false else true;
    }

    proc main() {
      try {
          var availMem = getAvailMemory();
          writeln("availMem %s".format(availMem));
      } catch e: Error{
          try! writeln("error: %s".format(e));
      }
   }
}