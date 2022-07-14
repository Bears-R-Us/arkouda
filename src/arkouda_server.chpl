/* arkouda server
backend chapel program to mimic ndarray from numpy
This is the main driver for the arkouda server */

use FileIO;
use Security;
use ServerConfig;
use Time only;
use ZMQ only;
use Memory;
use FileSystem;
use IO;
use Logging;
use Path;
use MultiTypeSymbolTable;
use MultiTypeSymEntry;
use MsgProcessing;
use GenSymIO;
use Reflection;
use SymArrayDmap;
use ServerErrorStrings;
use Message;
use ServerDaemon;

use CommandMap, ServerRegistration;

private config const logLevel = ServerConfig.logLevel;
const asLogger = new Logger(logLevel);

proc main() {
    var daemon = new ArkoudaServerDaemon();
    try! daemon.run();
}
