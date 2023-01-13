/* arkouda server
backend chapel program to mimic ndarray from numpy
This is the main driver for the arkouda server */

use ServerConfig;
use IO;
use Reflection;
use Logging;
use ServerDaemon;

private config const logLevel = ServerConfig.logLevel;
private config const logChannel = ServerConfig.logChannel;
const asLogger = new Logger(logLevel, logChannel);

/**
 * The main method serves as the Arkouda driver that invokes the run 
 * method on the configured list of ArkoudaServerDaemon objects
 */
proc main() {
    coforall daemon in getServerDaemons() {
        daemon.run();
    }
}