module LogMsg
{
    use ServerConfig;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    
    use MultiTypeSymbolTable;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const clLogger = new Logger(logLevel, logChannel);

    proc clientLogMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        const logMsg: string = msgArgs.getValueOf("log_msg");
        const logLvl: Logging.LogLevel = msgArgs.getValueOf("log_lvl").toUpper(): Logging.LogLevel;
        const tag: string = msgArgs.getValueOf("tag");

        const mod_replace: string = "ClientGeneratedLog";

        select logLvl {
            when Logging.LogLevel.DEBUG {
                clLogger.debug(mod_replace,tag,0,logMsg);
            }
            when Logging.LogLevel.INFO {
                clLogger.info(mod_replace,tag,0,logMsg);
            }
            when Logging.LogLevel.WARN {
                clLogger.warn(mod_replace,tag,0,logMsg);
            }
            when Logging.LogLevel.ERROR {
                clLogger.error(mod_replace,tag,0,logMsg);
            }
            when Logging.LogLevel.CRITICAL {
                clLogger.critical(mod_replace,tag,0,logMsg);
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