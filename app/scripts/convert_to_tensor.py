import os
import nibabel as nib
import torch
from tqdm import tqdm

from maikol_utils.print_utils import print_warn
from src.config import Configuration

def convert_niigz_to_tensor(CONFIG: Configuration):
    # Loop through all patients
    for patient_id in tqdm(os.listdir(CONFIG.mr_nf_structural)):
        patient_path = os.path.join(CONFIG.mr_nf_structural, patient_id)
        
        # Ensure it is a directory and only process baseline scans (_11)
        if not os.path.isdir(patient_path) or not patient_id.endswith('_11'): 
            print_warn(f"Skipping {patient_path} as it is not a directory or does not end with '_11'")
            continue

        # Extract the base ID (e.g., UPENN-GBM-00001) from the folder name
        base_id = patient_id.split('_')[0] 

        # Create an empty tensor to hold all 4 modalities (Shape: 4, X, Y, Z)
        modalities = ['T1', 'T1GD', 'T2', 'FLAIR']
        tensor_list = []

        for mod in modalities:
            file_path = os.path.join(patient_path, f"{patient_id}_{mod}.nii.gz")
            img_data = nib.load(file_path).get_fdata()
            tensor_list.append(torch.tensor(img_data, dtype=torch.float32))
        
        # Stack into a single 4D tensor and save
        stacked_tensor = torch.stack(tensor_list)
        torch.save(stacked_tensor, os.path.join(CONFIG.mr_nf_tensors, f"{base_id}.pt"))