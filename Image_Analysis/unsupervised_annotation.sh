#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --mem=5g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=burns756@umn.edu
#SBATCH -o /home/hirschc1/burns756/Pericarp/Data/Images/Annotation_Images/Ground_Truth_Cooked/Split_GT_Images/%j.out
#SBATCH -e /home/hirschc1/burns756/Pericarp/Data/Images/Annotation_Images/Ground_Truth_Cooked/Split_GT_Images/%j.err

echo ${CS}
echo ${METRIC}
echo ${LINKAGE}

# Run python script
python3 unsupervised_annotation.py ${CS} ${METRIC} ${LINKAGE}

echo 'Finished'
