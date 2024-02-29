#!/usr/bin/env bash

echo Time is `date`
echo Directory is `pwd`
#
env

TIME_STAMP=$(date +"%Y-%m-%d-%H-%M-%S")
ROOT_DIR=../../../..
SETUP=5-uni
SCHEMA=dblp
SUB=ub-rec

#output_dir=/scratch/$USER/dataset/I-RAVEN
WDPATH=${ROOT_DIR}/src
CPATH=${ROOT_DIR}/.lace-asp
EXPGPATH=${WDPATH}/experiment/${SETUP}
EXPATH=${EXPGPATH}/${SCHEMA}
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
python -u mains_explain_.py -c -l ${EXPATH}/${SCHEMA}.lp --typed_eval --no_show --presimed --ternary --rec-track --ub --schema ${SCHEMA} > ${LOGPATH}/${SCHEMA}-${SUB}-ter-${TIME_STAMP}.log
stop="$(date +%s)"
finish=$(( $stop-$start ))
echo Job-Time $finish seconds
echo End Time is `date`

