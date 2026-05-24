#!/bin/bash

# Define Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Header
echo -e "${BLUE}========================================================${NC}"
echo -e "${BLUE}          GLIOBLASTOMA SURVIVAL PIPELINE                ${NC}"
echo -e "${BLUE}========================================================${NC}"

echo -e "${CYAN}Sourcing conda and activating environment...${NC}"
conda activate ARA_env
cd app

# Experiment Runner Function
run_exp() {
    echo -e "\n${YELLOW}--------------------------------------------------------${NC}"
    echo -e "${GREEN}RUNNING:${NC} $1"
    echo -e "${YELLOW}--------------------------------------------------------${NC}"
    $2
}

run_exp "No masks - Upenn - No Radiomics" \
"python main.py train --ssl_dataset upenn --exp_name Upenn-No_masks-No_Radiomics"

run_exp "Train masks - Upenn - No Radiomics" \
"python main.py train -mtr --ssl_dataset upenn --exp_name Upenn-Train_masks-No_Radiomics"

run_exp "No masks - Upenn - Radiomics" \
"python main.py train --ssl_dataset upenn --use_radiomics --exp_name Upenn-No_masks-Radiomics"

run_exp "Train masks - Upenn - Radiomics" \
"python main.py train -mtr --ssl_dataset upenn --use_radiomics --exp_name Upenn-Train_masks-Radiomics"


run_exp "All masks - Upenn - No Radiomics" \
"python main.py train -mtr -mts --ssl_dataset upenn --exp_name Upenn-All_masks-No_Radiomics"

run_exp "All masks - Upenn - Radiomics" \
"python main.py train -mtr -mts --ssl_dataset upenn --use_radiomics --exp_name Upenn-All_masks-Radiomics"






run_exp "No masks - brats - No Radiomics" \
"python main.py train --ssl_dataset brats --exp_name brats-No_masks-No_Radiomics"

run_exp "Train masks - brats - No Radiomics" \
"python main.py train -mtr --ssl_dataset brats --exp_name brats-Train_masks-No_Radiomics"

run_exp "No masks - brats - Radiomics" \
"python main.py train --ssl_dataset brats --use_radiomics --exp_name brats-No_masks-Radiomics"

run_exp "Train masks - brats - Radiomics" \
"python main.py train -mtr --ssl_dataset brats --use_radiomics --exp_name brats-Train_masks-Radiomics"


run_exp "All masks - brats - No Radiomics" \
"python main.py train -mtr -mts --ssl_dataset brats --exp_name brats-All_masks-No_Radiomics"

run_exp "All masks - brats - Radiomics" \
"python main.py train -mtr -mts --ssl_dataset brats --use_radiomics --exp_name brats-All_masks-Radiomics"





echo -e "\n${BLUE}========================================================${NC}"
echo -e "${GREEN}             ALL EXPERIMENTS COMPLETED                 ${NC}"
echo -e "${BLUE}========================================================${NC}"