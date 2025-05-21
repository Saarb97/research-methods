#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 2-0:00:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
##SBATCH --job-name 				### name of the job
#SBATCH --output job-%J.out			### output log for running job - %J for job number
##SBATCH --gpus=1
#SBATCH --gpus=rtx_6000:1				### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1
##SBATCH --tmp=100G

# Note: the following 4 lines are commented out
#SBATCH --mail-user=saarbu@post.bgu.ac.il	### user's email for sending job status messages
#SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=59G
#SBATCH --cpus-per-task=6				### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load cuda/12.4
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate my_env				### activate a conda environment, replace my_env with your conda environment

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0  # Already set, but included for clarity
export LD_LIBRARY_PATH=/storage/modules/packages/cuda/cuda-12.4/lib64:$LD_LIBRARY_PATH  # Add CUDA 12.4 libraries

##echo "=== NVIDIA Driver and GPU Info ==="
##nvidia-smi > nvidia_smi_output.txt 2>&1 || echo "nvidia-smi failed"
##echo "=== CUDA Toolkit Version ==="
##nvcc --version > nvcc_version.txt 2>&1 || echo "nvcc not found"
##echo "=== Environment Variables ==="
##env | grep -E "CUDA|PATH|LD_LIBRARY_PATH" > env_vars.txt
##echo "=== System Logs (NVIDIA-related) ==="
##dmesg | grep -i nvidia > dmesg_nvidia.txt 2>&1 || echo "dmesg failed (possible permission issue)"


python -u main.py
echo "Python script finished running."

