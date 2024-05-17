#!/usr/bin/env bash

echo Time is `date`
echo Directory is `pwd`
#
env

# Define the name of the Conda environment
ENV_NAME="aspen"



TIME_STAMP=$(date +"%Y-%m-%d-%H-%M-%S")
ROOT_DIR=../../../..
SCHEMA=dblp
SUB=lb

WDPATH=${ROOT_DIR}/src
EXPGPATH=${WDPATH}/experiment
EXPATH=${EXPGPATH}/${SCHEMA}
LOGPATH=${EXPATH}/logs


cd ${WDPATH}
export PYTHONPATH=`pwd`

echo activating conda env ...


# Source Conda's shell initialization script
source "$CONDA_PREFIX/etc/profile.d/conda.sh"

conda init bash

# Activate the Conda environment by name
conda activate "$ENV_NAME"

echo Starting job ...
start="$(date +%s)"
#echo $csv_list
python -u mains_explain_.py -c -l ${EXPATH}/${SCHEMA}.lp  --main --lb --no_show --ternary --presimed --schema ${SCHEMA}  > ${LOGPATH}/${SCHEMA}-${SUB}-ter-${TIME_STAMP}.log
stop="$(date +%s)"
finish=$(( $stop-$start ))
echo Job-Time $finish seconds
echo End Time is `date`

