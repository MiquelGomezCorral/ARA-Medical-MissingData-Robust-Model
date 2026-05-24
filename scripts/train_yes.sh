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

run_exp "No Masks - Emb D1 - No Radiomics - D Dropout" \
"python main.py train -mts -mtr --dynamic_dropout --base_name 'No_Masks-D_Dropout'"


# ======================================================================

run_exp "All masks - Emb D1 - No Radiomics - D Dropout" \
"python main.py train --ssl_dataset upenn -mts -mtr --dynamic_dropout --base_name 'All_masks-D_Dropout'"

run_exp "All masks - Emb D3 - No Radiomics - D Dropout" \
"python main.py train --ssl_dataset upenn -mts -mtr --dynamic_dropout --pos_embed 3d --base_name 'All_masks-Emb_D3-D_Dropout'"



run_exp "All masks - Emb D1 - Radiomics - D Dropout" \
"python main.py train --ssl_dataset upenn -mts -mtr --dynamic_dropout --use_radiomics --base_name 'All_masks-Radiomics-D_Dropout'"

run_exp "All masks - Emb D3 - Radiomics - D Dropout" \
"python main.py train --ssl_dataset upenn -mts -mtr --dynamic_dropout --pos_embed 3d --use_radiomics --base_name 'All_masks-Emb_D3-Radiomics-D_Dropout'"





echo -e "\n${BLUE}========================================================${NC}"
echo -e "${GREEN}             ALL EXPERIMENTS COMPLETED                 ${NC}"
echo -e "${BLUE}========================================================${NC}"