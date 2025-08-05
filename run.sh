#!/bin/bash

# Export environment variables with paths
export OUTPUT_DIR="/scratch/sg/Priyanshu/TripCraft-main/output_7day_3"    # Path to your output directory
export MODEL_NAME="qwen"     #qwen    #phi4             # Model name (you can change it as needed)
export OPENAI_API_KEY="YOUR_OPENAI_KEY"             # Your OpenAI API key
# export GOOGLE_API_KEY="YOUR_GOOGLE_KEY"                 # Your Google API key
export DAY="7day"                                           # 3day/5day/7day
export SET_TYPE="7day_gpt4o_orig"                            # Set type- name of folder in O/P directory where generated outputs get saved 
export STRATEGY="direct_og"                                # direct_og / direct_param
export CSV_FILE="/scratch/sg/Priyanshu/7day_disruption.csv"  # Path to your CSV file

# Navigate to the planner directory
cd tools/planner

# Run the Python script with the environment variables
python sole_planning_mltp.py \
    --day $DAY \
    --set_type $SET_TYPE \
    --output_dir $OUTPUT_DIR \
    --csv_file $CSV_FILE \
    --model_name $MODEL_NAME \
    --strategy $STRATEGY
