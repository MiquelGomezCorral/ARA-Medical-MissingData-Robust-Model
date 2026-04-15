"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
from dataclasses import dataclass

from maikol_utils.file_utils import make_dirs
import yaml

@dataclass 
class Configuration:
    """Configuration class for the project."""
    # ===================================================================
    #                       PATHS
    # ===================================================================
    DATA_PATH: str = os.path.join("..", "data")
    MODELS_PATH: str = os.path.join("..", "models")
    LOGS_PATH: str = os.path.join("..", "logs")
    CONFIGS_PATH: str = os.path.join("..", "configs")
    yaml_config_name: str = None

    mr_path: str = os.path.join(DATA_PATH, "MR", 'metadata')
    mr_data: str = os.path.join(mr_path, "UPENN-GBM_clinical_info_v2.1.csv")

    mr_nf_path: str = os.path.join(DATA_PATH, "MR_NIfTI")
    mr_nf_structural: str = os.path.join(mr_nf_path, "images_structural")
    mr_nf_segm: str = os.path.join(mr_nf_path, "images_segm")
    mr_nf_tensors: str = os.path.join(mr_nf_path, "images_tensors")

    # ===================================================================
    #                       PARAMETER
    # ===================================================================

    exp_name: str = "base_name"
    seed:     int = 42

    gym_id:          str = None
    learning_rate: float = 2.5e-4
    total_timesteps: int = 25_000

    torch_deterministic: bool = True
    cuda:                bool = True

    track_run:         bool = False
    wandb_project_name: str = "RL"
    wandb_entity:       str = None

    def __post_init__(self):
        # Basic setup: create folders and load yaml config if provided
        make_dirs([
            self.DATA_PATH, self.MODELS_PATH, self.LOGS_PATH, self.CONFIGS_PATH,
            self.mr_path, self.mr_nf_path, self.mr_nf_structural, self.mr_nf_segm, self.mr_nf_tensors
        ])
        if self.yaml_config_name:
            self._load_yaml_configuration(self.yaml_config_name)

        # More stuff 
        ...

        
    def _load_yaml_configuration(self, yaml_file: str) -> None:
        """Load config values from a YAML file under CONFIGS_PATH."""
        config_path = os.path.join(self.CONFIGS_PATH, yaml_file)

        with open(config_path, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file) or {}

        for key, value in yaml_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
