/* arkouda server config param and config const */
module ServerConfig
{
    /*
    Verbose flag
    */
    config const v = true;

    /*
    Port for zeromq
    */
    config const ServerPort = 5555;

    /* 
    Arkouda version
    */
    config param arkoudaVersion = "0.0.9-2019-08-22";

    /*
    Configure MyDmap on compile line by "-s MyDmap=0" or "-s MyDmap=1"
    0 = Cyclic, 1 = Block. Cyclic may not work; we haven't tested it in a while.
    BlockDist is the default.
    */
    config param MyDmap = 1;

}
