module repro
{
    use Memory;
    
    config const perLocaleMemLimit = 80;
    
    class ErrorWithMsg: Error {
        var msg: string;
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
            var total = memoryUsed() + additionalAmount:uint;
            if total > getMemLimit() {
                throw new owned ErrorWithMsg("Error: Operation would exceed memory limit ("
                                             +total:string+","+getMemLimit:string+")");
            }
        }
    }
    
    proc main() {
        var repMsg:string = "None";
        
        try {
            overMemLimit(2**40);
            repMsg = "Something";
        } catch (e: ErrorWithMsg) {
            repMsg = e.msg;
        } catch {
            repMsg = "Error: Unkown";
        }
        
        writeln(repMsg);
    }

}