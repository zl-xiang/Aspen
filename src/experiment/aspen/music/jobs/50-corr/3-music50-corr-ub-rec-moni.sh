#!/usr/bin/env bash

echo Time is `date`
echo Directory is `pwd`
#
env


ROOT_DIR=/home/zhiliangxiang/Academic/project/lace-asp
SETUP=5-uni
SCHEMA=music
DATA=50
SUB=ub-rec

#output_dir=/scratch/$USER/dataset/I-RAVEN
WDPATH=${ROOT_DIR}/ERA/src
CPATH=${ROOT_DIR}/.lace-asp
EXPGPATH=${WDPATH}/experiment/${SETUP}
EXPATH=${EXPGPATH}/${SCHEMA}
LOGPATH=${EXPATH}/logs
#rm -rf ${output_dir}
#mkdir -p ${output_dir}
cd ${WDPATH}
export PYTHONPATH=`pwd`

echo Loading modules ...
module purge
module load anaconda
module list
echo Loading succeed ...
csv_files=`ls ${OPATH}`
echo activating conda env ...
source activate ${CPATH}
# batch size 8, scale mean and std to fit 3x3 grid , n epochs incremented by 2
echo Starting job ...
start="$(date +%s)"
#echo $csv_list
python mains_explain_.py -c -l ${EXPATH}/${SCHEMA}-${SUB}.lp --typed_eval --data 50 --presimed --rec-track --schema music > ${LOGPATH}/${SCHEMA}-${SUB}.out
stop="$(date +%s)"
finish=$(( $stop-$start ))
echo Job-Time $finish seconds
echo End Time is `date`

