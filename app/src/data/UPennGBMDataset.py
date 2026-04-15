import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import Configuration

class UPennGBMDataset(Dataset):
    '''
        bins = [0, 180, 365, 730, float('inf')]
        labels = [0, 1, 2, 3] # Short, Mid, Long, Exceptional
    '''
    def __init__(self, CONFIG: Configuration, transform=None):
        self.transform = transform
        
        self.tensor_dir = CONFIG.mr_nf_tensors
        self.mr_data = CONFIG.mr_data

        self.bins = CONFIG.bins
        self.labels = CONFIG.labels
        self.process_tabular()
        

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_row = self.df.iloc[idx]
        # 1. Load Image Tensor (C, X, Y, Z)
        tensor_path = os.path.join(self.tensor_dir, f"{patient_row['ID']}.pt")
        image = torch.load(tensor_path, weights_only=True)

        # 2. Get Tabular and Label
        tabular = torch.tensor(
            patient_row[self.tabular_cols].values.astype('float32'),
            dtype=torch.float32
        )
        label = torch.tensor(
            patient_row[self.label_cols],
            dtype=torch.long
        )

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,     # Shape: (4, X, Y, Z)
            'tabular': tabular, # Shape: (N_features,)
            'label': label      # Scalar
        }
    

    def process_tabular(self):
        df = pd.read_csv(self.mr_data)

        # ===================================================================================
        #                             Get a single record per patient
        # ===================================================================================
        # 1. Create a temporary column with the Patient Root ID (e.g., UPENN-GBM-00612)
        df['Patient'] = df['ID'].str.split('_').str[0]
        # 2. Sort by ID so that _11 comes before _21 for the same patient
        df = df.sort_values(by='ID')
        # 3. Drop duplicates based on the Root ID, keeping the first one found
        # This keeps _11 if it exists; otherwise, it keeps _21.
        df = df.drop_duplicates(subset='Patient', keep='first')
        df['ID'] = df['Patient']

        # ===================================================================================
        #                             CLEAN AND PREPARE SURVIVAL DATA
        # ===================================================================================

        # Ensure numerical
        df['Survival_days'] = pd.to_numeric(df['Survival_from_surgery_days_UPDATED'], errors='coerce')
        df = df.dropna(subset=['Survival_days'])

        # Define bins and labels
        # bins [0, 180, 365, 730, infinity]
      
        df['Survival_Class'] = pd.cut(df['Survival_days'], bins=self.bins, labels=self.labels, include_lowest=True)
        
        # Cast to int for PyTorch CrossEntropyLoss
        df['Survival_Class'] = df['Survival_Class'].astype(int)


        # ===================================================================================
        #                             CLEAN OTHER DATA
        # ===================================================================================
        # Gender (no missing data)
        df['Gender'] = df['Gender'].map({'F': 0, 'M': 1}).astype(int)
        df['Age_at_scan_years'] = df['Age_at_scan_years'].astype(float) / 100.0

        # DUMMIES
        df['KPS'] = pd.to_numeric(df['KPS'], errors='coerce')
        # VERY LITTLE DATA, so keep it in just 3 beans
        def bin_kps(score):
            if pd.isna(score):
                return 'Unk'
            elif score <= 80:
                return 'High'
            else:
                return 'Low'
        df['KPS'] = df['KPS'].apply(bin_kps)
                

        df['IDH1'] = df['IDH1'].replace(['NOS/NEC'], 'Unk').fillna('Unk')

        df['MGMT'] = df['MGMT'].replace(['Not Available', 'Indeterminate'], 'Unk').fillna('Unk')

        df['GTR'] = df['GTR_over90percent'].replace(['Not Available', 'Not Applicable'], 'Unk').fillna('Unk')

        # 4. Create Dummies for these columns
        clinical_cols = ['KPS', 'IDH1', 'MGMT', 'GTR']
        df = pd.get_dummies(df, columns=clinical_cols, prefix=clinical_cols)

        # Parse to INT for numerical data
        for col in df.columns:
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(int)

        self.label_cols = 'Survival_Class'
        self.tabular_cols = [
            'Gender',
            'Age_at_scan_years',
            'KPS_High', 'KPS_Low', 'KPS_Unk',
            'IDH1_Mutated', 'IDH1_Wildtype', 'IDH1_Unk',
            'MGMT_Methylated', 'MGMT_Unmethylated', 'MGMT_Unk',
            'GTR_Y', 'GTR_N', 'GTR_Unk'
        ]
        self.df = df[['ID'] + self.tabular_cols + [self.label_cols]]
        