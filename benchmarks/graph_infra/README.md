This directory contains infrastructure required by the Chapel graphing scripts

- .perfkeys files contain the strings to search for in benchmark output. These
  keys are then stored in .dat files.
- .graph files contain the graph information (title, perfkeys, graphkeys, .dat
  file)
- The GRAPHFILE file is a meta file that lists the .graph files

Benchmark output and a .perfkey file is used by `computePerfStats` to create or
append to a .dat file. `genGraphs` then takes the .dat files and the meta
information in the .graph file to generate interactive graphs. To view the
graphs locally you can do:

    cd benchmarks/datdir/html
    python3 -m http.server 8000
    open http://localhost:8000/ (or navigate to localhost:8000 in your browser)
