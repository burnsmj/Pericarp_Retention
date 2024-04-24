#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=burns756@umn.edu
#SBATCH -o /home/hirschc1/burns756/Pericarp/%j.out
#SBATCH -e /home/hirschc1/burns756/Pericarp/%j.err

# Change conda environments
# conda activate Research_Env

# Run python script
python3 kernel_correction_and_splitting.py

echo 'Finished'
