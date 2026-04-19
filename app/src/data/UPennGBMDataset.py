import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from src.config import Configuration

class UPennGBMDataset(Dataset):
    '''
        bins = [0, 180, 365, 730, float('inf')]
        labels = [0, 1, 2, 3] # Short, Mid, Long, Exceptional
    '''
    def __init__(self, CONFIG: Configuration, transform=None, partition='train', cache=False):
        self.transform = transform
        self.partition = partition

        self.CONFIG = CONFIG

        self.tensor_dir = CONFIG.mr_nf_tensors_96
        self.mr_data = CONFIG.mr_data

        self.bins = CONFIG.bins
        self.labels = CONFIG.labels
        self.process_tabular()
    

        self.cache = {}
        if cache:
            print(f"[{partition}] caching {len(self.df)} tensors in RAM...")
            for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"Caching {partition}"):
                path = os.path.join(self.tensor_dir, f"{row['ID']}.pt")
                self.cache[row['ID']] = torch.load(path, weights_only=True)
            print(f"[{partition}] cache ready.")
            

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_row = self.df.iloc[idx]
        pid = patient_row['ID']
    
        # Load from cache or disk
        image = self.cache[pid] if self.cache else torch.load(
            os.path.join(self.tensor_dir, f"{pid}.pt"), weights_only=True
        )

        for c in range(image.shape[0]):
            channel = image[c]
            mean = channel.mean()
            std  = channel.std().clamp(min=1e-5)   # clamp avoids div-by-zero on blank channels
            image[c] = (channel - mean) / std

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

        # ADD THIS BLOCK: Ensure all required columns exist
        for col in self.tabular_cols:
            if col not in df.columns:
                df[col] = 0

        # Now this will never throw a KeyError
        df = df[['ID'] + self.tabular_cols + [self.label_cols]]
        df.fillna(0, inplace=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        test_end = int(self.CONFIG.test_split)
        val_end = test_end + int(self.CONFIG.val_split)

        if self.partition == 'test':
            self.df = df.iloc[:test_end]
            
        elif self.partition == 'val':
            self.df = df.iloc[test_end:val_end]
            
        elif self.partition == 'train':
            self.df = df.iloc[val_end:]
            
        else:
            raise ValueError(f"Invalid partition: {self.partition}")