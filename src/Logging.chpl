module Logging {
    use Set;
    use FileSystem;
    use Reflection;
    use ServerErrors;
    use Time;

    import IO.{format, stdout, file, open, ioMode};

    /*
     * The LogLevel enum is used to provide a strongly-typed means of
     * configuring the logging level of a Logger object
     */
    enum LogLevel {DEBUG,INFO,WARN,ERROR,CRITICAL};
    
    /*
     * The LogChannel enum is used to provide a strongly-typed means of
     * configuring the channel such as stdout or file where log messages
     * are written.
     */
    enum LogChannel {CONSOLE,FILE};

    /*
     * The OutputHandler class defines the interface for all derived 
     * classes that write log messages to various channels.
     */
    class OutputHandler {
        proc write(message: string) throws {}
    }
    
    /*
     * The ConsoleOutputHandler writes log messages to the Arkouda console.
     */
    class ConsoleOutputHandler : OutputHandler {
        override proc write(message: string) throws {
            writeln(message);
            stdout.flush();
        }
    }

    /*
     * The FileOutputHandler writes log messages to the configured filePath.
     */    
    class FileOutputHandler : OutputHandler {
        var filePath: string;
        
        proc init(filePath: string) {
            super.init();
            this.filePath = filePath;
        }

        override proc write(message: string) throws {
            this.writeToFile(this.filePath, message);
        }
        
        /*
         * Writes to file, creating file if it does not exist
         */
        proc writeToFile(filePath : string, line : string) throws {
            var writer;
            if exists(filePath) {
                var aFile = open(filePath, ioMode.rw);
                writer = aFile.writer(region=aFile.size.., locking=false);
            } else {
                var aFile = open(filePath, ioMode.cwr);
                writer = aFile.writer(locking=false);
            }

            writer.writeln(line);
            writer.flush();
            writer.close();
        }
    }
    
    /*
     * getOutputHandler is a factory method for OutputHandler implementations.
     */
    proc getOutputHandler(channel: LogChannel) : owned OutputHandler throws {
        if channel == LogChannel.CONSOLE {
            return new owned ConsoleOutputHandler();
        } else {
            return new owned FileOutputHandler("%s/.arkouda/arkouda.log".format(here.cwd()));
        }
    }
    
    /*
     * The Logger class provides structured log messages at various levels
     * of logging sensitivity analogous to other languages such as Python.
     */
    class Logger {
        var level: LogLevel = LogLevel.INFO;
        
        var warnLevels = new set(LogLevel,[LogLevel.WARN, LogLevel.INFO,
                           LogLevel.DEBUG]);
        
        var criticalLevels =  new set(LogLevel, [LogLevel.CRITICAL, LogLevel.WARN,
                                         LogLevel.INFO,LogLevel.DEBUG]);
                                         
        var errorLevels =  new set(LogLevel,[LogLevel.ERROR, LogLevel.CRITICAL,
                           LogLevel.WARN, LogLevel.INFO, LogLevel.DEBUG]);
                             
        var infoLevels = new set(LogLevel,[LogLevel.INFO,LogLevel.DEBUG]);  
        
        var printDate: bool = true;
        
        var outputHandler: owned OutputHandler = try! getOutputHandler(LogChannel.CONSOLE);
        
        proc init() {}
       
        proc init(level: LogLevel) {
            this.level = level;
        }
        
        proc init(level: LogLevel, channel: LogChannel) {
            this.level = level;
            this.outputHandler = try! getOutputHandler(channel);
        }

        proc debug(moduleName, routineName, lineNumber, msg: string) throws {
            try {
                if this.level == LogLevel.DEBUG  {
                    this.outputHandler.write(generateLogMessage(moduleName, routineName, lineNumber, 
                                            msg, "DEBUG"));
                }
            } catch (e: Error) {
                writeln(generateErrorMsg(moduleName, routineName, lineNumber, e));
            }
        }
        
        proc info(moduleName, routineName, lineNumber, msg: string) throws {
            try {
                if infoLevels.contains(level) {
                    this.outputHandler.write(generateLogMessage(moduleName, routineName, lineNumber, 
                                            msg, "INFO"));
                }
            } catch (e: Error) {
                writeln(generateErrorMsg(moduleName, routineName, lineNumber, e));
            }
        }
        
        proc warn(moduleName, routineName, lineNumber, msg: string) throws {
            try {
                if warnLevels.contains(level) {
                    this.outputHandler.write(generateLogMessage(moduleName, routineName, lineNumber, 
                                            msg, "WARN"));
                }
            } catch (e: Error) {
                writeln(generateErrorMsg(moduleName, routineName, lineNumber, e));
            }
        }
        
        proc critical(moduleName, routineName, lineNumber, msg: string) throws {
            try {
                this.outputHandler.write(generateLogMessage(moduleName, routineName, lineNumber, 
                                            msg, "CRITICAL"));
            } catch (e: Error) {
                writeln(generateErrorMsg(moduleName, routineName, lineNumber, e));
            }            
        }
        
        proc error(moduleName, routineName, lineNumber, msg: string) throws {
            try {
                this.outputHandler.write(generateLogMessage(moduleName, routineName, lineNumber, 
                                            msg, "ERROR"));
            } catch (e: Error) {
                writeln(generateErrorMsg(moduleName, routineName, lineNumber, e));
            }
        }
        
        proc generateErrorMsg(moduleName: string, routineName, lineNumber, 
                           error) throws {
            return "Error in logging message for %s %s %i: %?".format(
                    moduleName, routineName, lineNumber, error.message());                
        }
        
        proc generateLogMessage(moduleName: string, routineName, lineNumber, 
                           msg, level: string) throws {
            var lineStr: string = if lineNumber != 0 then "Line %i ".format(lineNumber) else "";
             if printDate {
                 return "%s [%s] %s %s%s [Chapel] %s".format(
                 generateDateTimeString(), moduleName,routineName,lineStr, 
                                     level,msg);
             } else {
                 return "[%s] %s %s%s [Chapel] %s".format(moduleName, 
                 routineName,lineStr,level,msg);            
             }
        }
         
        proc generateDateTimeString() throws {
            const t = dateTime.now();
            return "%i-%02i-%02i %02i:%02i:%02i".format(t.year, t.month, t.day, t.hour, t.minute, t.second);
        }
    }
}
