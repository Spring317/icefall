
#/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(dirname $(find $CONDA_PREFIX -name libnvrtc.so.12 | head -n 1))
export PYTHONPATH=~/icefall/:$PYTHONPATH

