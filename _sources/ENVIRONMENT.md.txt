# Environment Variables

There are a number of environment variables you can use (depending on role, i.e. user|developer)
to configure your environment.  This document will highlight the various environment variables
available in separate sections.

## Running

These env vars are used when running the `arkouda_server`.

- ARKOUDA_SERVER_CONNECTION_INFO : Set if you want the Arkouda server to write `server:port` info to file on startup
- To tune buffers used for message aggregation during sorting on non-crazy systems, you can set the following.  They are per
task aggregation buffers so there is no contention between competing tasks.
  - ARKOUDA_SERVER_AGGREGATION_DST_BUFF_SIZE : Used for tuning buffers associated with communication aggregation
  - ARKOUDA_SERVER_AGGREGATION_SRC_BUFF_SIZE : Used for tuning the buffers associated with communication aggregation
  - ARKOUDA_SERVER_AGGREGATION_YIELD_FREQUENCY : Configure the frequency when Aggregators yield, default every 1024 messages.
  
## Compilation / Makefile

These env vars can be used to configure your build of Arkouda when running `make`

### Chapel Compiler Flags

- CHPL_FLAGS : A number of flags will be added automatically to the `chpl` compiler in the Makefile, you can add your
  own, additional ones here.
  - `-smemTrack=true -lhdf5 -lhdf5_hl -lzmq -liconv -lidn2`, will add `--fast` unless one of the following env vars are set.
- ARKOUDA_DEVELOPER : Setting this to 1 or true will add the `-O1` flag to CHPL_FLAGS.  NOTE: _mutually exclusive_ with
  `ARKOUDA_QUICK_COMPILE`
- ARKOUDA_QUICK_COMPILE : Setting this to 1 or true will add the following flags.  NOTE: _mutually exclusive_ with
  `ARKOUDA_DEVELOPER`
  - `--no-checks --no-loop-invariant-code-motion --no-fast-followers --ccflags="-O0"`
- ARKOUDA_PRINT_PASSES_FILE : Setting this adds `--print-passes-file <file>` to the Chapel compiler flags and writes
  the associated "pass timing" output to the specified file.  This is mainly used in the nightly testing infrastructure.
- CHPL_DEBUG_FLAGS : We add `--print-passes` automatically, but you can add additional flags here.
- REGEX_MAX_CAPTURES : Set this to an integer to change the maximum number of capture groups accessible using ``Match.group``
  (set to 20 by default)

### Dependency Paths

Most folks install anaconda and link to these libraries through Makefile.paths instructions.  If you have an alternative
setup you can set them explicitly via:

- ARKOUDA_ZMQ_PATH : Path to ZMQ library
- ARKOUDA_HDF5_PATH : Path to HDF5 library
- ARKOUDA_ARROW_PATH : Path to Arrow library
- ARKOUDA_ICONV_PATH : Path to iconv library
- ARKOUDA_IDN2_PATH : Path to idn2 library
- LD_LIBRARY_PATH : Path to environment `lib` directory.
- ARKOUDA_SKIP_CHECK_DEPS : Setting this will skip the automated checks for dependencies (i.e. ZMQ, HDF5). This is
  useful for developers doing repeated Arkouda builds since they should have already verified the deps have been set up.

### Adding a Module from Outside the Arkouda src Directory

- ARKOUDA_SERVER_USER_MODULES : Absolute path or string of absolute paths separated by a space to modules outside of the Arkouda source directory to be included in the Arkouda build. The module name must also be included in `ServerModules.cfg` for the function to be registered with the server.

## Testing

Also see the python tests [README](tests/README.md) for more information on Python based unit & functional testing.

- VERBOSE : Setting this to `1` will add the `--print-passes` Chapel compiler flag when running _Chapel_-based unit tests.
- ARKOUDA_VERBOSE : Client env to set verbosity
- ARKOUDA_SERVER_HOST : Client env var of the Arkouda Server hostname
- ARKOUDA_SERVER_PORT : Client env var of the Arkouda Server port
- ARKOUDA_CLIENT_TIMEOUT : Client env var to control the client timeout for unit testing.
- ARKOUDA_FULL_STACK_TEST : Client testing option
- TEST_DATA_URL : Client testing variable for ReadAllTest/read_all_tests.py
- ARKOUDA_NUMLOCALES : Client unit testing option to set the number of Chpl server locales for unit tests
- ARKOUDA_SERVER_CONNECTION_INFO : Client env var to specify where the `ak-server-info` file is found
- ARKOUDA_HOME : This is used by `make check` tests to specify the location of Arkouda's server executable and
                  server_util/test module.  **_WARNING_**: The env var is subject to future change since it is mainly an
                  internal use variable.

## Python Client

- ARKOUDA_CLIENT_DIRECTORY : Sets the parent directory of where the client will look for `.arkouda/tokens.txt`
- ARKOUDA_TUNNEL_SERVER : Env var to control ssh tunnel server url
- ARKOUDA_KEY_FILE : Client env var for keyfile when using ssh tunnel
- ARKOUDA_PASSWORD : Client env var for password when using ssh tunnel
- ARKOUDA_LOG_LEVEL : Client env var to control client side Logging Level
- ARKOUDA_CLIENT_MODE: Client env var controlling client mode (UI or API), where UI mode displays the Arkouda client splash message. 
