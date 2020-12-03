module Logging {
    use Set;
    
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
        var warnLevels = new set(LogLevel,[LogLevel.WARN,LogLevel.DEBUG,LogLevel.INFO]);
        var debugLevels = new set(LogLevel,[LogLevel.DEBUG,LogLevel.INFO]);  
        
        proc debug(moduleName, routineName, lineNumber, msg: string) throws {
            if debugLevels.contains(level) {
                writeln("[%s] %s Line %i DEBUG [Chapel] %s".format(moduleName, 
                    routineName, lineNumber, msg));
            }
        }
        
        proc info(moduleName, routineName, lineNumber, msg: string) throws {
            if level == LogLevel.INFO {
                writeln("[%s] %s Line %i INFO [Chapel] %s".format(moduleName, 
                    routineName, lineNumber, msg));
            }
        }
        
        proc warn(moduleName, routineName, lineNumber, msg: string) throws {
            if warnLevels.contains(level) {
                writeln("[%s] %s Line %i WARN [Chapel] %s".format(moduleName, 
                    routineName, lineNumber, msg));
            }
        }
        
        proc critical(moduleName, routineName, lineNumber, msg: string) throws {
            writeln("[%s] %s Line %i CRITICAL [Chapel] %s".format(moduleName, 
                    routineName, lineNumber, msg));
        }
        
        proc error(moduleName, routineName, lineNumber, msg: string) throws {
            writeln("[%s] %s Line %i ERROR [Chapel] %s".format(moduleName, 
                    routineName, lineNumber, msg));
        }
    }
}