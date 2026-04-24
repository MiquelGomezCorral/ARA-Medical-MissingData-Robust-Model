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

# 1. Baseline
run_exp "Standard Training (No Masks)" \
"python main.py train --ssl_epochs 20 --survival_epochs 20"

# 2. Masked Train Only
run_exp "Masked Training Partition (-mtr)" \
"python main.py train -mtr --ssl_epochs 20 --survival_epochs 20"

# 3. Masked Test Only
run_exp "Masked Testing Partition (-mts)" \
"python main.py train -mts --ssl_epochs 20 --survival_epochs 20"

# 4. Full Masking
run_exp "Full Masking (Train + Test)" \
"python main.py train -mtr -mts --ssl_epochs 20 --survival_epochs 20"

echo -e "\n${BLUE}========================================================${NC}"
echo -e "${GREEN}             ALL EXPERIMENTS COMPLETED                 ${NC}"
echo -e "${BLUE}========================================================${NC}"