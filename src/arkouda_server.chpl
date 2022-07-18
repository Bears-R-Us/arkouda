/* arkouda server
backend chapel program to mimic ndarray from numpy
This is the main driver for the arkouda server */

use ServerConfig;
use IO;
use Logging;
use ServerDaemon;

private config const logLevel = ServerConfig.logLevel;
const asLogger = new Logger(logLevel);

proc main() {
    try! getServerDaemon().run();
}