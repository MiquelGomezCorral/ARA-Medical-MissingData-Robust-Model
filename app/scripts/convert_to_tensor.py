import os
import nibabel as nib
import torch
from tqdm import tqdm

from maikol_utils.print_utils import print_warn
from src.config import Configuration

def convert_niigz_to_tensor(CONFIG: Configuration):
    # Collect candidate folders by base ID and prefer _11 over _21.
    selected_patients = {}

    for patient_id in os.listdir(CONFIG.mr_nf_structural):
        patient_path = os.path.join(CONFIG.mr_nf_structural, patient_id)

        if not os.path.isdir(patient_path):
            print_warn(f"Skipping {patient_path} as it is not a directory")
            continue

        if not (patient_id.endswith('_11') or patient_id.endswith('_21')):
            continue

        base_id, visit_suffix = patient_id.rsplit('_', 1)
        existing = selected_patients.get(base_id)

        if existing is None:
            selected_patients[base_id] = (patient_id, visit_suffix)
            continue

        # Keep _11 whenever present; only use _21 if _11 does not exist.
        _, existing_suffix = existing
        if existing_suffix == '21' and visit_suffix == '11':
            selected_patients[base_id] = (patient_id, visit_suffix)

    for base_id, (patient_id, _) in tqdm(selected_patients.items()):
        patient_path = os.path.join(CONFIG.mr_nf_structural, patient_id)

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