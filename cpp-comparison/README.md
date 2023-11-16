## Programs included
### `read-parquet-low-level.cpp`
This file is using Arrow's undocumented low-level API to read data in batches into C++ arrays. This is the same API we are using in Arkouda today, so the performance here _should_ be comparable to Arkouda.

This program requires 4 command line arguments, filename, base column name, batch size, and number of columns and you'd run it like:
```
./a.out test-file_LOCALE0000 col 8192 2
```

For both this program and the next, by "base column name" I am assuming that the columns you'd like to read will all be the same name with a number on the end. In other words, I am assuming you have something like "col1", "col2", "col3"..., where "col" is the base column name, and the program is automatically appending the column number on the end. Not the best way to do this, but it was the best I could come up with! 

The batch size determines how many values that Parquet should be trying to read in at once and is also controllable in Arkouda with the `ARKOUDA_SERVER_PARQUET_BATCH_SIZE` env variable. 8192 is recommended by Arrow and is what we are using in Arkouda by default, but you can play around with it to see if the impacts. Generally, lower values perform worse, while higher values are just truncated to 8192.

### `read-parquet-standard.cpp`
This file is using Arrow's documented standrd Arrow API, which is not as optimized as the low-level API. The performance here should be a little worse than Arkouda and the low-level API version.

This program requires 3 command line arguments: filename, base column name, and number of columns (so, same as the last, but without the batch size).
```
./a.out test-file_LOCALE0000 col 2
```

### `build-df-write.py`
This file is building a dataframe with 2 columns, named `col1` and `col2` with integer columns of size 10**8. If you use this file, the programs will "just work" as I've given them to you (you might need to update the path to where you'd like it to go though).

### `time-ak-read.py`
I spent almost no time on this, since I assume you already know how to read and time for Arkouda reads, but this is what I used...

## Compiling
Compilation command: `g++ read-parquet.cpp -O3 -std=c++17 <include/link flags>`

If you are able to compile Arkouda yourself, an easy way to figure out which flags to use on your system would be to run `make compile-arrow-cpp` in Arkouda and steal the compilation flags from the compilation command that Arkouda is running, since Arkouda is already finding your Arrow install already. You will likely see `-I`, `-L`, and `-l` flags, depending on where you run and copy-pasting those onto the command above into the `<include/link flags>` section should do the trick.

If you aren't able to build Arkouda to steal the flags, but know where your Arrow build lives, you should be able to compile with:
```
g++ read-parquet-low-level.cpp -O3 -std=c++17 -larrow -lparquet -I/path/to/arrow/include -L/path/to/arrow/lib
```

If you aren't able to install Arkouda and don't know where your Arrow build lives, you can try running:
```
g++ read-parquet-low-level.cpp -O3 -std=c++17 -larrow -lparquet
```
And hopefully it will be able to find Arrow and Parquet for you. If none of those work, let me know and we can figure it out!

## Debugging failed compiles/runs

Compiling and running C/C++ programs on big machines can be... not as fun as Chapel. Here are some potential issues:

1. at runtime: `error while loading shared libraries: libarrow.so.900: cannot open shared object file: No such file or directory`
- oh no! You likely have to append the path to the shared object file to the `LD_LIBRARY_PATH`
- so, if your `so` file is located at `/path/to/arrow/libarrow.so.900`, you'd need to do something like `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/arrow/`

...

## Launching your jobs
When you launch Arkouda (or any Chapel program build where `CHPL_LAUNCHER!=none`), Chapel is building up the command for the workload manager on your machine (likely Slurm), which is launching the server onto compute nodes. When you run `./arkouda_server -nl<node-count>`, try running with the `--verbose` flag to see what command it is using to launch onto the compute nodes! (and then you can copy the important things there to launch whatever job you want, i.e., the C++ programs)

Most large machines typically have a login-node that is used for things like launching, building, etc., but you don't want to execute programs on there, since there could be other jobs contending with yours, so you might not be getting full machine resources, which would mean that your performance numbers are going to have a lot of variance (and you don't want to slow things down for other users!). To solve this problems, we have "workload managers" like Slurm and Qsub, which users submit jobs to from the login-node, and then they will reserve certain nodes to satisfy the jobs that users request.

So, say you have a machine with 4 nodes and one guy is running a job that takes all 4 nodes and another guy tries to run his program during that run. Slurm will queue it up and then, once the other run finishes and the nodes are free, the second job will reserve those nodes and run the program so there won't be simultaneous runs on any nodes fighting for machine resources! This is useful, but it makes running things a little bit harder than on your latop.

There are two ways that I'd recommend considering running your jobs with:
1. `salloc -N=1` and then `ssh <node-received>`
  - reserve a single node for yourself and then ssh onto the machine so you can do whatever you want without worrying about other jobs
  - this is probably the easiest way to run the C++ programs, since you'll be compiling and building just like usual
  - if you want your Arkouda server to be running on this same node, you'll have to `unset CHPL_LAUNCHER` and then rebuild Arkouda (which is probably a hastle)
  - when in doubt, `salloc -h`!
2. `srun <my-command>`
  - this is what Chapel is doing for you with `CHPL_LAUNCHER!=none` (remember, running `./arkouda_server -nl<node-count> --verbose` will show you launch command)
  - you can also launch your C++ programs with this, something like `srun ./a.out <args>` and it will be magically run on a compute node
  - you might get different nodes for your jobs, which probably doesn't matter, but if there are different node partitions with different hardware, you'll either want to constrain the `srun` to give you the same partition or request a specific node with `--nodelist`
  - when in doubt, `srun -h`!

## Comparing to Chapel
When you are using a distributed build of Chapel, there is going to be some overhead from the distributed checks, communication checks, and that kind of thing, even if you are running on a single locale. This means that the single-locale performance of Arkouda is going to have a little bit of additional overhead, but it shouldn't be too significant in this case.

If you have the ability to build Arkouda and use ones other the default system install, using the regular environment, but with `export CHPL_COMM=none`, this will give you a non-distributed build that won't have the extra distributed overhead.

If you can't do that, that's OK too...

### Results I collected
Reading 1 columns using low-level API: 1.112s
Reading 2 columns using low-level API: 1.897s

Reading 1 columns using standard API: 1.39s
Reading 2 columns using standard API: 2.251s

Reading 1 columns using Arkouda: 0.6521358489990234
Reading 2 columns using Arkouda: 1.4953250885009766
