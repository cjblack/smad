#!/bin/bash

# Generic options:

#SBATCH --account=bdrap08 # Run job under project <project>
#SBATCH --time=10:00:0        # Run time

# Recommend to add
# Add your own information here --> #SBATCH --output=/users/username/logs/smad_train_C57_climbing_noise_%j.out # Set output file location
# Add your own information here --> #SBATCH --error=/users/username/logs/smad_train_C57_climbing_noise_%j.err # Set error file location
# Add your own information here --> #SBATCH --mail-user=user@emailaddress.com
# #SBATCH --mail-type=ALL

# Node resources:
# (choose between 1-4 gpus per node)

#SBATCH --partition=gpu    # Choose either "gpu", "test" or "infer" partition type
#SBATCH --nodes=1           # Resources from a single node
#SBATCH --gres=gpu:1        # One GPU per node (plus 25% of node CPU and RAM per GPU)

# Run commands:
SECONDS=0

# Load necessary modules

module load cuda/11.4.1

# Activate conda environment
echo "ACTIVATING CONDA ENVIRONMENT"
# Add your system specific code for activating relevant conda environment
conda activate smad

# Run python script

echo "CHANGING TO SMAD DIRECTORY"
# Add your smad location here --> cd smad/project/location

echo "RUNNING SCRIPT"
python3 -u -m tests.smad_training_C57_climbing_with_noise

echo "Elapsed time: $(($SECONDS / 3600))h $((($SECONDS % 3600) / 60))m $(($SECONDS % 60))s"