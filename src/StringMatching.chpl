module StringMatching {
    use SegmentedString;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerConfig;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const smLogger = new Logger(logLevel, logChannel);


    // One to One Matching
    proc match_levenshtein(query, dataStr, s1: int, s2: int): int {
        if s1 == query.size || s2 == dataStr.size {
            return query.size - s1 + dataStr.size - s2;
        }  

        // matching chars
        if query[s1] == dataStr[s2]{
            return match_levenshtein(query, dataStr, s1 + 1, s2 + 1);
        }
            
        return 1 + min(
            match_levenshtein(query, dataStr, s1, s2 + 1),      // insert character
            match_levenshtein(query, dataStr, s1 + 1, s2),      // delete character
            match_levenshtein(query, dataStr, s1 + 1, s2 + 1)   // replace character
        );
    }

    // One to Many Matching
    proc segstring_match_levenshtein(query: string, dataName: string, st: borrowed SymTab): string throws {
        var segString = getSegString(dataName, st);
        var distances: [0..#segString.size] int;

        coforall i in 0..#segString.size {
            var dataStr = segString[i];
            
            distances[i] = match_levenshtein(query, dataStr, 0, 0);
        }

        var name = st.nextName();
        var distEntry = new shared SymEntry(distances);
        st.addEntry(name, distEntry);

        return "created " + st.attrib(name);
    }

    // TODO: Many One to One matching
    proc segstring_many_match_levenshtein(queryName: string, dataName: string, st: borrowed SymTab): string throws {
        var dataString = getSegString(dataName, st);
        var queryString = getSegString(queryName, st);
        var distances: [0..#dataString.size] int;

        coforall i in 0..#dataString.size {
            var dataStr = dataString[i];
            var queryStr = queryString[i];
            
            distances[i] = match_levenshtein(queryStr, dataStr, 0, 0);
        }

        var name = st.nextName();
        var distEntry = new shared SymEntry(distances);
        st.addEntry(name, distEntry);

        return "created " + st.attrib(name);
    }

}