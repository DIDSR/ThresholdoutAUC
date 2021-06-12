#$ -cwd
#$ -S /bin/sh
#$ -j y
#$ -o sysout
#$ -N glmnaive
#$ -P CDRHID0007
#$ -l h_vmem=32G  # 48G # a small number of tasks may need more RAM -- adjust accordingly when rerunning any failed tasks
#$ -l h_rt=048:00:00  # 096:00:00 # a small number of tasks may need more time than 48 hours -- adjust accordingly when rerunning any failed tasks
#$ -t 1-100


echo "Running job $JOB_NAME ($JOB_ID) on $HOSTNAME"

source /projects/mikem/applications/centos7/gcc/source.sh
source /projects/mikem/applications/R-4.0.2/set_env.sh
export R_LIBS_USER=/home/alexej.gossmann/R-4.0.2_packages:$R_LIBS_USER

LOG_DIR=log/glmnet_random/
#mkdir -p $LOG_DIR
LOG_FILE=log/glmnet_random/"$JOB_ID.$SGE_TASK_ID".txt

RSCRIPT=holdout_reuse_glmnet.R
R CMD BATCH --no-save --no-restore $RSCRIPT $LOG_FILE
