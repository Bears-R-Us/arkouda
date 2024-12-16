#   These are the steps for an offline build of arkouda on almalinux:9.0 docker container


#   Step 1:  Make sure the environment variables are set.  See bashrc.chpl for an example.
#   Step 2:  Install the chapel.  An example script is given in install_chapel.sh
#   Step 3:  Install the python dependencies in requirements.txt.  If the dependencies are pre-downloaded into a <pip_deps> directory:
    
    >   pip install --no-index --find-links <pip_deps> -r requirements.txt

#   Step 4:  Set environment variable, export ARROW_DEPENDENCIES_SOURCE_DIR=<build_dir>/arrow_dependencies

#   Step 5:  Install dependencies from source.  If the source is in <build_dir>:
    >   cd arkouda
    >   make install-deps DEP_BUILD_DIR=<build_dir>

#   Step 6:  pip install the arkouda project
    >   python3 -m pip install -e .[dev] 
    
#   Step 7:  make
    >   make

#   Step 8:  verify the unit tests run:
    >   make test
