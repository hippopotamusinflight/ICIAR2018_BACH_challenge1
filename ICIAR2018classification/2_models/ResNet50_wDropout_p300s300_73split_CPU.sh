#!/bin/bash
#SBATCH --job-name=ResNet50_wDropout_p300s300_73split
#SBATCH --mail-type=ALL
#SBATCH --mail-user=19mh17@queensu.ca
#SBATCH --account=con-hpcg1553 
#SBATCH --partition=reserved 
#SBATCH -c 80 
#SBATCH --mem 512g 
#SBATCH -t 12:00:00 
#SBATCH --output=./logs/%j-%N-%x.out
#SBATCH --error=./logs/%j-%N-%x.err

job_name="ResNet50_wDropout_p300s300_73split"

echo "job ${job_name} started at "$(date) 
time1=$(date +%s)

source /global/home/hpc4535/kerasTFpy36/bin/activate
module load cuda cudnn 

ICIAR_dir="/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/2_pretrained_models_test/"

python ${ICIAR_dir}ResNet50_wDropout_p300s300_73split.py


time2=$(date +%s)
echo "\njob ${job_name} ended at "$(date) 
echo "job took $(((time2-time1)/3600)) hours $(((time2-time1)%3600)) minutes"
