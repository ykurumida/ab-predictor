#!/bin/bash
#============ PBS Options ============
#$ -S /bin/sh
#$ -l s_vmem=10G 
#$ -l mem_req=10G
#$ -l d_rt=15:00:00
#$ -cwd
#$ -t 1-100:1
#$ -j y
#$ -pe def_slot 1
#============ Shell Script ============

export PATH=$HOME/.local/bin:$HOME/bin:$PATH
source $HOME/.bash_profile
#source /lustre6/home/ykurumida/local/virtualenv/combo_190422/bin/activate
source /lustre6/home/ykurumida/local/venv/seqlogo_190507/bin/activate
mkdir ../Output
SEED=${SGE_TASK_ID}
OUT_DIR=1
mkdir ../Output/${OUT_DIR}
#python3 ../script/run.py ../../data/P049_data.csv ${SEED} > ../output/10_fold_floating/log.${SEED} 
python3 ../code/corr_run.py ../Input/ddG.csv ${SEED} >& ../Output/${OUT_DIR}/log.${SEED} 

