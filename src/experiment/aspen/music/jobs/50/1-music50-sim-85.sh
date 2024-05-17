#!/usr/bin/env bash

echo Time is `date`
echo Directory is `pwd`
#
env

TIME_STAMP=$(date +"%Y-%m-%d-%H-%M-%S")
ROOT_DIR=/home/zhiliangxiang/Academic/project/lace-asp
SETUP=5-uni
SCHEMA=music
DATA=50
SPEC=85
SUB=sim-${SPEC}

#output_dir=/scratch/$USER/dataset/I-RAVEN
WDPATH=${ROOT_DIR}/ERA/src
CPATH=${ROOT_DIR}/.lace-asp
EXPGPATH=${WDPATH}/experiment/${SETUP}
EXPATH=${EXPGPATH}/${SCHEMA}
LPATH=${EXPGPATH}/${SCHEMA}/thresh
LOGPATH=${EXPATH}/logs

#rm -rf ${output_dir}
#mkdir -p ${output_dir}
cd ${WDPATH}
export PYTHONPATH=`pwd`

echo activating conda env ...

source activate ${CPATH}
# batch size 8, scale mean and std to fit 3x3 grid , n epochs incremented by 2
echo Starting job ...
start="$(date +%s)"
#echo $csv_list
python -u mains_explain_.py -c -l ${LPATH}/${SCHEMA}-${SPEC}.lp  --getsim --typed_eval --ternary --schema ${SCHEMA} --data ${DATA} > ${LOGPATH}/${SCHEMA}-${SUB}-${DATA}-ter-${TIME_STAMP}.log
stop="$(date +%s)"
finish=$(( $stop-$start ))
echo Job-Time $finish seconds
echo End Time is `date`

