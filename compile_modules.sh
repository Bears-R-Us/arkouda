FILE_NAME=$1
CLASS_NAME=$(echo $1 | cut -d'.' -f 1)
echo $FILE_NAME
echo $CLASS_NAME
chpl --print-passes  --fast -smemTrack=true -lhdf5 -lhdf5_hl -lzmq \
-Idep/./zeromq-install/include \
-Ldep/./zeromq-install/lib \
--ldflags="-Wl,-rpath,./dep/zeromq-install/lib" \
-I./dep/hdf5-install/include \
-L./dep/hdf5-install/lib \
--ldflags="-Wl,-rpath,./dep/hdf5-install/lib" \
-M src/ $FILE_NAME -o $CLASS_NAME