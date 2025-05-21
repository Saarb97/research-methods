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
##SBATCH --gpus=rtx_6000:1				### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1
##SBATCH --tmp=100G

# Note: the following 4 lines are commented out
#SBATCH --mail-user=saarbu@post.bgu.ac.il	### user's email for sending job status messages
#SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6				### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate my_env				### activate a conda environment, replace my_env with your conda environment
python openai_embedding.py

##Final-Research-Project/imdb/imdb_model.py ###jupyter lab			### run my_code.py python file and send my_arg1 argument to it
##export SLURM_SCRATCH_DIR=/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}
##cp /storage/*.img $SLURM_SCRATCH_DIR
##mkdir $SLURM_SCRATCH_DIR/test
##cp -r $SLURM_SCRATCH_DIR $SLURM_SUBMIT_DIR 
