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
    proc getOutputHandler(channel: LogChannel) : owned OutputHandler  {
        if channel == LogChannel.CONSOLE {
            return new owned ConsoleOutputHandler();
        } else {
            return new owned FileOutputHandler((try! here.cwd()) + "/.arkouda/arkouda.log");
        }
    }
    
    /*
     * The Logger class provides structured log messages at various levels
     * of logging sensitivity analogous to other languages such as Python.
     */
    class Logger {
        var level: LogLevel = LogLevel.INFO;
        
        var printDate: bool = true;
        
        var outputHandler: owned OutputHandler = getOutputHandler(LogChannel.CONSOLE);
        
        proc init() {}
       
        proc init(level: LogLevel) {
            this.level = level;
        }
        
        proc init(level: LogLevel, channel: LogChannel) {
            this.level = level;
            this.outputHandler = getOutputHandler(channel);
        }

        // Emit `msg` into the handler.
        proc emit(msg: string...) {
            try {
                this.outputHandler.write(if msg.size == 1 then msg[0]
                                         else "".join(msg));
            } catch (e: Error) {
                writeln("Error while logging message <", (...msg), "> : ", e.message());
            }
        }

        // Emit `msg`, depending on `level`.
        proc report(moduleName: string, routineName: string, lineNumber: int,
                    level: LogLevel, msg:string...)
        {
            if level < this.level then return;

            emit((...generateLogMessage(moduleName, routineName, lineNumber,
                                        level, (...msg))));
        }

        inline proc debug(moduleName, routineName, lineNumber, msg...) do
          report(moduleName, routineName, lineNumber, LogLevel.DEBUG, (...msg));

        inline proc info(moduleName, routineName, lineNumber, msg...) do
          report(moduleName, routineName, lineNumber, LogLevel.INFO, (...msg));

        inline proc warn(moduleName, routineName, lineNumber, msg...) do
          report(moduleName, routineName, lineNumber, LogLevel.WARN, (...msg));

        inline proc error(moduleName, routineName, lineNumber, msg...) do
          report(moduleName, routineName, lineNumber, LogLevel.ERROR, (...msg));
        
        inline proc critical(moduleName, routineName, lineNumber, msg...) do
          report(moduleName, routineName, lineNumber, LogLevel.CRITICAL, (...msg));
        
        proc generateLogMessage(moduleName: string, routineName: string, lineNumber: int,
                                level: LogLevel, msg: string...) {
            var lineStr = if lineNumber != 0 then "Line " + lineNumber:string + " " else "";
            var dateStr = if printDate then generateDateTimeString(" ") else "";
            return (dateStr, "[", moduleName, "] ", routineName, " ", lineStr, level:string,
                    " [Chapel ] ", (...msg));
        }
         
        proc generateDateTimeString(tail = "") {
            const t = dateTime.now();
            try {
                return "%i-%02i-%02i %02i:%02i:%02i%s".format(t.year, t.month, t.day,
                                                              t.hour, t.minute, t.second, tail);
            } catch {
                return "".join(t.year:string, "-", t.month:string, "-", t.day:string, " ",
                               t.hour:string, ":", t.minute:string, ":", t.second:string, tail);
            }
        }
    }
}
