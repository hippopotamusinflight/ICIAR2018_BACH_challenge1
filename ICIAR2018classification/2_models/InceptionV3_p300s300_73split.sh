#!/bin/bash
#SBATCH --job-name=InceptionV3_BNandDO_predict_4_classes_p300s300_73split
#SBATCH --mail-type=ALL
#SBATCH --mail-user=19mh17@queensu.ca
#SBATCH --account=gpu-hpcg1553

#SBATCH --qos=avin
#SBATCH --partition=gpu-vin

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem 128g
#SBATCH --time 1-00:00:00

#SBATCH --output=./logs/%j-%N-%x.out
#SBATCH --error=./logs/%j-%N-%x.err

job_name="InceptionV3_BNandDO_predict_4_classes_p300s300_73split"

echo "job ${job_name} started at "$(date) 
time1=$(date +%s)

source /global/home/hpc4535/kerasTFpy36/bin/activate
module load cuda cudnn 

ICIAR_dir="/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/2_pretrained_models_test/"

python ${ICIAR_dir}InceptionV3_BNandDO_predict_4_classes_p300s300_73split.py


time2=$(date +%s)
echo "\njob ${job_name} ended at "$(date) 
echo "job took $(((time2-time1)/3600)) hours $(((time2-time1)%3600)) minutes"
