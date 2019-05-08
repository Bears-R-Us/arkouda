module ServerConfig
{
    // verbose flag
    config const v = true;

    // port for zeromq
    config const ServerPort = 5555;

    // arkouda version
    config param arkoudaVersion = "0.0.8pre";

    // configure MyDmap on compile line by "-s MyDmap=0" or "-s MyDmap=1"
    // 0 = Cyclic, 1 = Block, Cyclic may not work now haven't tested it in a while
    // BlockDist is the default now
    config param MyDmap = 1;

}
