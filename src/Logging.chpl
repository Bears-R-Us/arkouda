module Logging {
    use Set;
    use IO;
    use DateTime;

    /*
     * The LogLevel enum is used to provide a strongly-typed means of
     * configuring the logging level of a Logger object
     */
    enum LogLevel {DEBUG,INFO,WARN,ERROR,CRITICAL};
    
    /*
     * The Logger class provides structured log messages at various levels
     * of logging sensitivity analogous to other languages such as Python.
     */
    class Logger {
        var level = LogLevel.WARN;
        
        var warnLevels = new set(LogLevel,[LogLevel.WARN, LogLevel.INFO,
                           LogLevel.DEBUG]);
        
        var criticalLevels =  new set(LogLevel, [LogLevel.CRITICAL, LogLevel.WARN,
                                         LogLevel.INFO,LogLevel.DEBUG]);
                                         
        var errorLevels =  new set(LogLevel,[LogLevel.ERROR, LogLevel.CRITICAL,
                           LogLevel.WARN, LogLevel.INFO, LogLevel.DEBUG]);
                             
        var infoLevels = new set(LogLevel,[LogLevel.INFO,LogLevel.DEBUG]);  
        
        var printDate: bool = true;
        
        proc debug(moduleName, routineName, lineNumber, msg: string) throws {
            if level == LogLevel.DEBUG  {
                writeln(generateLogMessage(moduleName, routineName, lineNumber, 
                                            msg, "DEBUG"));
                stdout.flush();
            }
        }
        
        proc info(moduleName, routineName, lineNumber, msg: string) throws {
            if infoLevels.contains(level) {
                writeln(generateLogMessage(moduleName, routineName, lineNumber, 
                                            msg, "INFO"));
                stdout.flush();
            }
        }
        
        proc warn(moduleName, routineName, lineNumber, msg: string) throws {
            if warnLevels.contains(level) {
                writeln(generateLogMessage(moduleName, routineName, lineNumber, 
                                            msg, "WARN"));
                stdout.flush();
            }
        }
        
        proc critical(moduleName, routineName, lineNumber, msg: string) throws {
            writeln(generateLogMessage(moduleName, routineName, lineNumber, 
                                            msg, "CRITICAL"));
            stdout.flush();
        }
        
        proc error(moduleName, routineName, lineNumber, msg: string) throws {
            writeln(generateLogMessage(moduleName, routineName, lineNumber, 
                                            msg, "ERROR"));
            stdout.flush();
        }
        
        proc generateLogMessage(moduleName: string, routineName, lineNumber, 
                           msg, level: string) throws {
             if printDate {
                 return "%s [%s] %s Line %i %s [Chapel] %s".format(
                 generateDateTimeString(), moduleName,routineName,lineNumber, 
                                     level,msg);
             } else {
                 return "[%s] %s Line %i %s [Chapel] %s".format(moduleName, 
                 routineName,lineNumber,level,msg);            
             }
        }
         
        proc generateDateTimeString() throws {
            var dts = datetime.today():string;
            var vals = dts.split("T");
            var cd = vals(0);
            var rawCms = vals(1).split(".");
            return "%s:%s".format(cd,rawCms(0));        
        }
    }
}