/* arkouda server
backend chapel program to mimic ndarray from numpy
This is the main driver for the arkouda server */

use ServerConfig;
use IO;
use Reflection;
use Logging;
use ServerDaemon;

private config const logLevel = ServerConfig.logLevel;
const asLogger = new Logger(logLevel);

/**
 * The main method serves as the Arkouda driver that invokes the run 
 * method on the configured list of ArkoudaServerDaemon objects
 */
proc main() {
    asLogger.info(getModuleName(), 
                  getRoutineName(), 
                  getLineNumber(),
                  'Starting Arkouda Server Daemons');
    coforall daemon in getServerDaemons() {
        daemon.run();
    }
}