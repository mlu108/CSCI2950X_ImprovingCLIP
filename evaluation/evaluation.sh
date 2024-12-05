#!/bin/bash
# This is an example batch script for slurm on Hydra
#
# The commands for slurm start with #SBATCH
# All slurm commands need to come before the program # you want to run. In this example, 'echo "Hello World!"
# is the command we are running.
#
# This is a bash script, so any line that starts with # is # a comment. If you need to comment out an #SBATCH line, use # infront of the #SBATCH
#
# To submit this script to slurm do:
# sbatch batch.script
#
# Once the job starts you will see a file MySerialJob-****.out
# The **** will be the slurm JobID
# --- Start of slurm commands -----------
# set the partition to run on the gpus partition. The Hydra cluster has the following partitions: compute, gpus, debug, tstaff
#SBATCH --partition=gpus

# request 1 gpu resource
#SBATCH --gres=gpu:1
 

# Request an hour of runtime. Default runtime on the compute parition is 1hr.
#SBATCH --time=1:00:00
# Request a certain amount of memory (4GB):
#SBATCH --mem=4G
# Specify a job name:
#SBATCH -J MySerialJob
# Specify an output file
# %j is a special variable that is replaced by the JobID when the job starts
#SBATCH -o MySerialJob-%j.out #SBATCH -e MySerialJob-%j.out
#----- End of slurm commands ----
# # Run a command

# nvidia-smi
# nvcc --version
# which python3

# sbatch -p gpu --gres=gpu:1 --time=4:00:00 -N 1 -o /users/sboppana/data/sboppana/CSCI2950X_ImprovingCLIP/evaluation/results/results1.out --mem=32G ./evaluation.sh

python3 /users/sboppana/data/sboppana/CSCI2950X_ImprovingCLIP/evaluation/benchmark_clip_model.py