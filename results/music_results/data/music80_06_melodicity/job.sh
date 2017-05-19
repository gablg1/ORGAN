#!/bin/bash

#SBATCH -n 8   # Number of cores
#SBATCH -N 1  # Ensure that all cores are on one machine
#SBATCH -t 20-0:00   # Runtime in D-HH:MM
#SBATCH -p aagk80       # Partition to submit to
#SBATCH --constraint=cuda-7.5
#SBATCH --mem-per-cpu=1024   # Memory pool for each core
#SBATCH -J music80_06_melodicity
#SBATCH -o OGAN.out
#SBATCH --gres=gpu
#SBATCH --mail-type=FAIL,END  # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=gabrielguimaraes@college.harvard.edu

export OMP_NUM_THREADS=8
# define variables
module load cuda/7.5-fasrc02 cudnn/7.0-fasrc02
module load gcc/4.9.3-fasrc01 tensorflow/1.0.0-fasrc04
source activate tfgpu

# variables for the job
JOB_NAME=music80_06_melodicity
PY_SCRIPT='train_ogan.py'
CUR_DIR=$(pwd)
SCRATCH=/scratch
RESULTS=$(pwd)

#Create start dummy file
touch $CUR_DIR/$JOB_NAME.start

echo "setuping up directories"
echo "  at ${SCRATCH}/${JOB_NAME}/${SLURM_JOB_ID}"
#setup scratch dirs
cd $SCRATCH
mkdir -p $JOB_NAME/$SLURM_JOB_ID
cd $JOB_NAME/$SLURM_JOB_ID
cp -Rv $CUR_DIR/* .

# run python script
echo "running python"
python ${PY_SCRIPT}  2>&1 | tee ${CUR_DIR}/results.out
# move results of calc
echo "copy results"
mv ${JOB_NAME}.out $RESULTS
cp -r * $RESULTS

#Create finished dummy file:
touch $CUR_DIR/$JOB_NAME.done