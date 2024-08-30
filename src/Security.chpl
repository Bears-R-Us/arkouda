
module Security {
    use Random;
    use FileIO;
    use FileSystem;
    use Path;
    use ServerConfig;
    private use IO;

    proc generateToken(len: int=32) : string {
        var alphanum = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j","k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
            ];

        return ''.join(try! sample(alphanum, len-1, withReplacement=true));
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
       // First see if there is token via env variable
       var token : string = getEnv(name='ARKOUDA_SERVER_TOKEN');
       
       if token.isEmpty() {
           // No token env variable, so generate it 
           token = generateToken(len);
       }
       appendFile(filePath=tokensPath, line=token);
       return token;
   }       
}
