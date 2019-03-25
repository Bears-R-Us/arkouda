module ServerConfig
{
    // verbose flag
    config const v = true;

    // port for zeromq
    config const ServerPort = 5555;

    // arkouda version
    config param arkouda_version = "0.0.6pre";

    // configure MyDmap on compile line by "-s MyDmap=0" or "-s MyDmap=1"
    // 0 = Cyclic, 1 = Block, Block still does not work the way we want set at init() time
    config param MyDmap = 1;

}
