import os
import json
import pickle
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
        self.CONFIG = CONFIG

        self.transform = transform
        self.partition = partition

        self.apply_mask = CONFIG.masked_train if partition == 'train' else CONFIG.masked_test

        self.tensor_dir = CONFIG.mr_nf_tensors_96
        self.mr_data = CONFIG.mr_data
        self.dropout_data_path = CONFIG.dropout_data_path

        self.bins = CONFIG.bins
        self.labels = CONFIG.labels
        self.dropout_reference, self.dropout_by_id = self._load_dropout_data()
        self.process_tabular()
    

        self.cache = {}
        if cache:
            print(f"[{partition}] caching {len(self.df)} tensors in RAM...")
            for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"Caching {partition}"):
                self.cache[row['ID']] = self._load_sample(row['ID'])
            print(f"[{partition}] cache ready.")
            

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_row = self.df.iloc[idx]
        pid = patient_row['ID']
    
        # Load from cache or disk
        image = self.cache[pid] if self.cache else self._load_sample(pid)

        # image = self._normalize_image_dict(image)
        image, image_mask = self._apply_image_mask(pid, image)

        # 2. Get Tabular, mask, and Label
        tabular_values = torch.tensor(
            patient_row[self.tabular_cols].values.astype('float32'),
            dtype=torch.float32
        )
        tabular_mask = torch.ones(len(self.tabular_cols), dtype=torch.float32)
        tabular_values, tabular_mask = self._apply_tabular_mask(pid, tabular_values, tabular_mask)
        label = torch.tensor(
            patient_row[self.label_cols],
            dtype=torch.long
        )

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,             # expect torch.Size([4, D, D, D]) con D = 96
            'image_mask': image_mask,   # Shape: (N_channels,)
            'tabular': tabular_values,  # Shape: (N_features,)
            'tabular_mask': tabular_mask,
            'label': label             # Scalar
        }
    
    

    def _load_dropout_data(self):
        with open(self.dropout_data_path, "r", encoding="utf-8") as f:
            dropout_data = json.load(f)

        return dropout_data.get("reference", {}), dropout_data.get("ids", {})


    def _sample_path(self, pid):
        pkl_path = os.path.join(self.tensor_dir, f"{pid}.pkl")
        if os.path.exists(pkl_path):
            return pkl_path

        # pt_path = os.path.join(self.tensor_dir, f"{pid}.pt")
        # if os.path.exists(pt_path):
        #     return pt_path

        raise FileNotFoundError(f"No tensor file found for {pid} in {self.tensor_dir}")


    def _load_sample(self, pid):
        path = self._sample_path(pid)
        with open(path, "rb") as f:
            sample = pickle.load(f)

        if not isinstance(sample, dict):
            raise ValueError(f"Expected a dict sample for {pid}, got {type(sample).__name__}")

        return sample


    # def _normalize_image_dict(self, image_dict):
    #     normalized = {}
    #     for feature_name, tensor in image_dict.items():
    #         channel = tensor.clone().float()
    #         mean = channel.mean()
    #         std = channel.std().clamp(min=1e-5)
    #         normalized[feature_name] = (channel - mean) / std
    #     return normalized


    def _apply_image_mask(self, pid, image_dict):
        image_mask = {
            feature_name: torch.tensor(1.0, dtype=torch.float32)
            for feature_name in image_dict.keys()
        }

        if self.apply_mask:
            patient_dropout = self.dropout_by_id.get(pid, {})
            
            for feature_name, reference_value in self.dropout_reference.items():
                if feature_name == "TABULAR":
                    continue

                if feature_name != "RADIOMIC":
                    if patient_dropout.get(feature_name, False) and feature_name in image_dict:
                        image_dict[feature_name] = torch.zeros_like(image_dict[feature_name])
                        image_mask[feature_name] = torch.tensor(0.0, dtype=torch.float32)
                    continue

                radiomic_reference = reference_value if isinstance(reference_value, dict) else {}
                radiomic_dropout = patient_dropout.get("RADIOMIC", {})

                for group_name, group_spec in radiomic_reference.items():
                    if not radiomic_dropout.get(group_name, False):
                        continue

                    prefixes = group_spec[0] if isinstance(group_spec, list) and group_spec else []
                    for feature_key in image_dict.keys():
                        if any(feature_key.startswith(prefix) for prefix in prefixes):
                            image_dict[feature_key] = torch.zeros_like(image_dict[feature_key])
                            image_mask[feature_key] = torch.tensor(0.0, dtype=torch.float32)

        keys = list(image_dict.keys())
        image_tensor = torch.stack([image_dict[k] for k in keys], dim=0)
        mask_tensor  = torch.stack([image_mask[k] for k in keys], dim=0)
        return image_tensor, mask_tensor


    def _apply_tabular_mask(self, pid, tabular_values, tabular_mask):
        if not self.apply_mask:
            return tabular_values, tabular_mask

        patient_dropout = self.dropout_by_id.get(pid, {})
        tabular_dropout = patient_dropout.get("TABULAR", {})
        tabular_reference = self.dropout_reference.get("TABULAR", {})

        for group_name, group_spec in tabular_reference.items():
            if not tabular_dropout.get(group_name, False):
                continue

            columns = group_spec[0] if isinstance(group_spec, list) and group_spec else []
            for column_name in columns:
                if column_name not in self.tabular_cols:
                    continue
                column_idx = self.tabular_cols.index(column_name)
                tabular_values[column_idx] = 0.0
                tabular_mask[column_idx] = 0.0

        return tabular_values, tabular_mask


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

        with open(self.CONFIG.partition_ids_path, "r", encoding="utf-8") as f:
            partition_ids = json.load(f)

        if self.partition not in partition_ids:
            raise ValueError(
                f"Partition '{self.partition}' not found in {self.CONFIG.partition_ids_path}. "
                f"Available partitions: {list(partition_ids.keys())}"
            )

        target_ids = partition_ids[self.partition]
        if not isinstance(target_ids, list):
            raise ValueError(
                f"Partition '{self.partition}' must map to a list of IDs in "
                f"{self.CONFIG.partition_ids_path}"
            )

        df_by_id = df.set_index('ID')
        existing_ids = [pid for pid in target_ids if pid in df_by_id.index]
        missing_ids = [pid for pid in target_ids if pid not in df_by_id.index]

        if missing_ids:
            print(
                f"[{self.partition}] Warning: {len(missing_ids)} IDs from partition file "
                f"are not present in tabular data and will be skipped."
            )

        self.df = df_by_id.loc[existing_ids].reset_index()