module StringMatchingMsg {
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use StringMatching;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerConfig;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const smLogger = new Logger(logLevel, logChannel);
    

    proc stringMatchingMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var mode = msgArgs.getValueOf("mode");
        var query = msgArgs.getValueOf("query");
        var data = msgArgs.getValueOf("dataset");
        var algo = msgArgs.getValueOf("algorithm");

        var repMsg = "";

        select mode {
            when "single" {
                select algo {
                    when "levenshtein" {
                        repMsg = "%i".format(match_levenshtein(query, data, 0, 0));
                    }
                    when "jaro" {
                        repMsg = "%r".format(match_jaro(query, data));
                    }
                    when "jaccard" {
                        repMsg = "%r".format(match_jaccard(query, data));
                    }
                }
            }
            when "multi" {
                repMsg = segstring_match(query, data, algo, st);
            }
            when "many" {
                repMsg = segstring_match_many(query, data, algo, st);
            }
        }

        smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }


    use CommandMap;
    registerFunction("stringMatching", stringMatchingMsg, getModuleName());
}