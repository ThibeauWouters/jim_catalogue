#!/bin/bash

# List of GW event IDs (you can replace these with actual IDs)
gw_ids=('GW191103_012549' 'GW191105_143521' 'GW191109_010717')

# Define the path to the template script
template_file="template.sh"

# Loop over each GW event ID
for gw_id in "${gw_ids[@]}"
do
  # Create a unique SLURM script for each GW event
  new_script="slurm_scripts/submit_${gw_id}.sh"
  cp $template_file $new_script
  
  # Replace the placeholder with the actual GW_ID
  sed -i "s/{{{GW_ID}}}/$gw_id/g" $new_script
  
  # Submit the job to SLURM
  sbatch $new_script
  
  echo "Submitted job for $gw_id"
done
