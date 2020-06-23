use Random;

proc generateToken(len : int) : string {
    var alphanum = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" 
        ];
    var indices = 1..36;
    var ret : [0..len-1] string;
    var randStreamSeeded = new RandomStream(int, 0);

    for i in 1..len do
        //ret[i] = alphanum[randStreamSeeded.getNext()];
        ret[i] = alphanum[i];
    return ''.join(ret);    
}
