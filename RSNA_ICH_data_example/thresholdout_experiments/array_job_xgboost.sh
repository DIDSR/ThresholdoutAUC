#$ -cwd
#$ -S /bin/sh
#$ -j y
#$ -o sysout
#$ -N xgbthresh
#$ -P CDRHID0007
#$ -pe thread 8
#$ -l h_vmem=5G
#$ -l h_rt=048:00:00
#$ -t 1-100


echo "Running job $JOB_NAME ($JOB_ID) on $HOSTNAME"

source /projects/mikem/applications/centos7/gcc/source.sh
source /projects/mikem/applications/R-4.0.2/set_env.sh
export R_LIBS_USER=/home/alexej.gossmann/R-4.0.2_packages:$R_LIBS_USER

LOG_DIR=log/xgb_random/
#mkdir -p $LOG_DIR
LOG_FILE=log/xgb_random/"$JOB_ID.$SGE_TASK_ID".txt

RSCRIPT=holdout_reuse_xgboost.R
R CMD BATCH --no-save --no-restore $RSCRIPT $LOG_FILE
