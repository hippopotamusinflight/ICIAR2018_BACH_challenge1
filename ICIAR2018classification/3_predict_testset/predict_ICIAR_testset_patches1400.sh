#!/bin/bash 
#SBATCH --job-name=predict_ICIAR_testset_patches1400
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=19mh17@queensu.ca 
#SBATCH --account=con-hpcg1553 
#SBATCH --partition=reserved 
#SBATCH -c 12 
#SBATCH --mem 512g 
#SBATCH -t 6:00:00 
#SBATCH --output=./logs/%j-%N-%x.out
#SBATCH --error=./logs/%j-%N-%x.err

job_name="predict_ICIAR_testset_patches1400" 

echo "job ${job_name} started at "$(date) 
time1=$(date +%s)

# module load python/3.6
source /global/home/hpc4535/kerasTFpy36/bin/activate

ICIAR_dir="/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/5_predict_photos/"

python ${ICIAR_dir}predict_ICIAR_testset_patches1400.py

time2=$(date +%s)
echo "job ${job_name} ended at "$(date) 
echo "job took $(((time2-time1)/60)) minutes $(((time2-time1)%60)) seconds"
