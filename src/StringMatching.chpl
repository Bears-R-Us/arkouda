module StringMatching {
    use SegmentedString;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerConfig;
    use ArraySetops;
    use Unique;

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

    // One to One Matching
    proc match_jaro(s1: string, s2: string): real throws {
        // If the strings are equal
        if (s1 == s2) {
            return 1.0;
        }
    
        // Length of two strings
        var len1 = s1.size;
        var len2 = s2.size;
    
        // Maximum distance up to which matching is allowed
        var max_dist = (floor(max(len1, len2) / 2) - 1): int;
    
        // Count of matches
        var match = 0.0;
    
        // Hash for matches
        var hash_s1: [0..#len1] int = 0;
        var hash_s2: [0..#len2] int = 0;
    
        // Traverse through the first string
        forall i in 0..#len1 with (+ reduce match) {
            var x = max(0, i - max_dist);
            var y = min(len2-1, i + max_dist + 1);

            for j in x..y {
                // If there is a match
                if (s1[i] == s2[j] && hash_s2[j] == 0) {
                    hash_s1[i] = 1;
                    hash_s2[j] = 1;
                    match += 1.0;
                    break;
                }
            }

            // var char = s1[i].toByte();
            // var sub: [x..y] uint(8) = s2[x..y].bytes();
            // var matches: [x..y] bool = [j in sub] char == j;
            // var (maxVal, maxInd) = maxloc reduce zip (matches, x..y);
            // var count = 0;
            // while maxVal { 
            //     if hash_s2[maxInd] == 0 {
            //         hash_s1[i] = 1;
            //         hash_s2[maxInd] = 1;
            //         match += 1.0;
            //         break;
            //     } else {
            //         count += 1;
            //         (maxVal, maxInd) = maxloc reduce zip (matches[x+count..y], x+count..y);
            //     }
            // }
        }
    
        // If there is no match
        if (match == 0.0) {
            return 0.0;
        }
    
        // Number of transpositions
        var t = 0.0;
    
        // Count number of occurrences where two characters match but there is a third matched character
        // in between the indices 
        forall i in 0..#len1 with (+ reduce t) {
            if (i <= len2 && hash_s1[i] == 1) {
                var maxVal: int;
                var maxInd: int;
                if (i < len2) {
                    (maxVal, maxInd) = maxloc reduce zip (hash_s2[i..], i..len2);
                } else {
                    maxInd = len2-1;
                    maxVal = hash_s2[maxInd];
                }
    
                if maxVal == 1 && s1[i] != s2[maxInd] {
                    t += 1.0;
                }
            }
        }

        t /= 2.0;
    
        // Return the Jaro Distance. (1 - Jaro Similarity)
        return 1.0 - (match / len1: real + match / len2: real + (match - t) / match) / 3.0;
    }

    // One to One Matching
    proc match_jaccard(s1, s2): real throws {
        var b1 = s1.bytes();
        var b2 = s2.bytes();
        
        const N1 = b1.size - 1;
        const N2 = b2.size - 1;

        var c1 = makeDistArray(N1, uint);
        var c2 = makeDistArray(N2, uint);

        cobegin {
            forall (c, i) in zip(c1, 0..#N1) {
                c += b1[i]:uint;
                c <<= 8;
                c += b1[i+1];
            }
            forall (c, i) in zip(c2, 0..#N2) {
                c += b2[i]:uint;
                c <<= 8;
                c += b2[i+1];
            }
        }

        var u1 = uniqueSort(c1, false);
        var u2 = uniqueSort(c2, false);

        var inter = intersect1d(u1, u2, true);
        var u = (u1.size + u2.size) - inter.size;

        return inter.size: real / u: real;
    }

    // One to Many Matching
    proc segstring_match(query: string, dataName: string, algo: string, st: borrowed SymTab): string throws {
        var segString = getSegString(dataName, st);
        var distances: [0..#segString.size] real;

        forall i in 0..#segString.size {
            var dataStr = segString[i];
            
            select algo {
                when "levenshtein" {
                    distances[i] = match_levenshtein(query, dataStr, 0, 0): real;
                }
                when "jaro" {
                    distances[i] = match_jaro(query, dataStr);
                }
                when "jaccard" {
                    distances[i] = match_jaccard(query, dataStr);
                }
            }
        }

        var name = st.nextName();
        var distEntry = new shared SymEntry(distances);
        st.addEntry(name, distEntry);

        return "created " + st.attrib(name);
    }

    // Many One to One matching
    proc segstring_match_many(queryName: string, dataName: string, algo: string, st: borrowed SymTab): string throws {
        var dataString = getSegString(dataName, st);
        var queryString = getSegString(queryName, st);
        var distances: [0..#dataString.size] real;

        forall i in 0..#dataString.size {
            var dataStr = dataString[i];
            var queryStr = queryString[i];
            
            select algo {
                when "levenshtein" {
                    distances[i] = match_levenshtein(queryStr, dataStr, 0, 0): real;
                }
                when "jaro" {
                    distances[i] = match_jaro(queryStr, dataStr);
                }
                when "jaccard" {
                    distances[i] = match_jaccard(queryStr, dataStr);
                }
            }
        }

        var name = st.nextName();
        var distEntry = new shared SymEntry(distances);
        st.addEntry(name, distEntry);

        return "created " + st.attrib(name);
    }
}