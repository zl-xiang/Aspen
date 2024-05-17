#!/bin/bash

echo Time is $(date)
echo Directory is $(pwd)

TIME_STAMP=$(date +"%Y-%m-%d-%H-%M-%S")
ROOT_DIR=$(dirname "$0") # Using relative path to script directory
SETUP=5-uni
SCHEMA=cora
SUB=lb

WDPATH=${ROOT_DIR}/ERA/src
CPATH=${ROOT_DIR}/.lace-asp
EXPGPATH=${WDPATH}/experiment/${SETUP}
EXPATH=${EXPGPATH}/${SCHEMA}
LOGPATH=${EXPATH}/logs

cd "${WDPATH}" || exit
export PYTHONPATH=$(pwd)

echo Activating conda env ...
CONDA_BASE=$(conda info --base)
if [ -d "$CONDA_BASE" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${CPATH}" || exit
else
    echo "Error: Conda base directory not found."
    exit 1
fi

echo Starting job ...
start=$(date +%s)
python -u mains_explain_.py -c -l "${EXPATH}/${SCHEMA}.lp" --main --lb --ternary --no_show --presimed --schema "${SCHEMA}" >"${LOGPATH}/${SCHEMA}-${SUB}-ter-${TIME_STAMP}.log"
stop=$(date +%s)
finish=$((stop - start))
echo Job-Time $finish seconds
echo End Time is $(date)
