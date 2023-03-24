module LogMsg
{
    use ServerConfig;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const clLogger = new Logger(logLevel, logChannel);

    proc clientLogMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        const logMsg: string = msgArgs.getValueOf("log_msg");
        const logLvl: Logging.LogLevel = msgArgs.getValueOf("log_lvl").toUpper(): Logging.LogLevel;

        select logLvl {
            when Logging.LogLevel.DEBUG {
                clLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),logMsg);
            }
            when Logging.LogLevel.INFO {
                clLogger.info(getModuleName(),getRoutineName(),getLineNumber(),logMsg);
            }
            when Logging.LogLevel.WARN {
                clLogger.warn(getModuleName(),getRoutineName(),getLineNumber(),logMsg);
            }
            when Logging.LogLevel.ERROR {
                clLogger.error(getModuleName(),getRoutineName(),getLineNumber(),logMsg);
            }
            when Logging.LogLevel.CRITICAL {
                clLogger.critical(getModuleName(),getRoutineName(),getLineNumber(),logMsg);
            }
            otherwise {
                var errorMsg = "Unknown Log Type Found.";
                clLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
        return new MsgTuple("Log Message Written Successfully.", MsgType.NORMAL);

    }

    use CommandMap;
    registerFunction("clientlog", clientLogMsg, getModuleName());
}