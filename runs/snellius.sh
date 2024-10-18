#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 20:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G

now=$(date)
echo "$now"

# Define dirs
export GW_ID="GW191204_110529"
export data_dir=$HOME/projects/jim_catalogue/data/outdir/$GW_ID
export outdir=$HOME/projects/jim_catalogue/runs/outdir/$GW_ID

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
conda activate /home/twouters2/miniconda3/envs/jim-dev3

# Run the script
python $HOME/projects/jim_catalogue/runs/run.py --event-id $GW_ID

echo "DONE"