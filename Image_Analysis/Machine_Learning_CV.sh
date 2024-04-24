#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50gb
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=burns756@umn.edu
#SBATCH -o /home/hirschc1/burns756/Pericarp/Outputs/%j.out
#SBATCH -e /home/hirschc1/burns756/Pericarp/Outputs/%j.err

# Load python
module load python3/3.8.3_anaconda2020.07_mamba

# Run script
python3 pixel_classification.py
