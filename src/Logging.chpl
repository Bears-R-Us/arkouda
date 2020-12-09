module Logging {
    use Set;
    use IO;

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
        var warnLevels = new set(LogLevel,[LogLevel.WARN,LogLevel.INFO,LogLevel.DEBUG]);
        var criticalLevels =  new set(LogLevel, [LogLevel.CRITICAL,LogLevel.WARN,
                                         LogLevel.INFO,LogLevel.DEBUG]);
        var errorLevels =  new set(LogLevel,[LogLevel.ERROR, LogLevel.CRITICAL,LogLevel.WARN,
                             LogLevel.INFO, LogLevel.DEBUG]);
        var infoLevels = new set(LogLevel,[LogLevel.INFO,LogLevel.DEBUG]);  
        
        proc debug(moduleName, routineName, lineNumber, msg: string) throws {
            if level == LogLevel.DEBUG  {
                writeln("[%s] %s Line %i DEBUG [Chapel] %s".format(moduleName, 
                    routineName, lineNumber, msg));
                stdout.flush();
            }
        }
        
        proc info(moduleName, routineName, lineNumber, msg: string) throws {
            if infoLevels.contains(level) {
                writeln("[%s] %s Line %i INFO [Chapel] %s".format(moduleName, 
                    routineName, lineNumber, msg));
                stdout.flush();
            }
        }
        
        proc warn(moduleName, routineName, lineNumber, msg: string) throws {
            if warnLevels.contains(level) {
                writeln("[%s] %s Line %i WARN [Chapel] %s".format(moduleName, 
                    routineName, lineNumber, msg));
                stdout.flush();
            }
        }
        
        proc critical(moduleName, routineName, lineNumber, msg: string) throws {
            writeln("[%s] %s Line %i CRITICAL [Chapel] %s".format(moduleName, 
                    routineName, lineNumber, msg));
            stdout.flush();
        }
        
        proc error(moduleName, routineName, lineNumber, msg: string) throws {
            writeln("[%s] %s Line %i ERROR [Chapel] %s".format(moduleName, 
                    routineName, lineNumber, msg));
            stdout.flush();
        }
    }
}