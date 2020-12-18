
module Security {
    use Random;
    use Random.PCGRandom only PCGRandomStream;
    use FileIO;
    use FileSystem;
    use Path;
    private use IO;

    proc generateToken(len: int=32) : string {
        var alphanum = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j","k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
            ];

        var upper_bound = alphanum.size;
        var indices : [0..upper_bound-1] int;
        for i in 0..upper_bound-1 do
            indices[i] = i;

        var ret : [0..len-1] string;
        var r = new owned PCGRandomStream(int);
        var rindices = try! r.choice(indices, len);

        for i in 1..len-1 do
            ret[i] = alphanum[rindices[i]];
        return ''.join(ret);
    }

    proc getArkoudaToken(tokensPath : string) : string throws {
        var token : string;

        if exists(tokensPath) {
            token = getLineFromFile(tokensPath, -1);
            if token.isEmpty() {
                return setArkoudaToken(tokensPath);
            } else {
                return token;
            } 
        } else {
            return setArkoudaToken(tokensPath);
        } 
   }

   proc setArkoudaToken(tokensPath : string, len : int=32) : string throws {
       var token = generateToken(len);
       appendFile(filePath=tokensPath, line=token);
       return token;
   }       
}
