"""Main file for scripts with arguments and call other functions."""

import dotenv
import argparse
from src.config import Configuration
from maikol_utils.other_utils import args_to_dataclass
from maikol_utils.print_utils import print_separator

from scripts import convert_niigz_to_tensor, train_3d_vit

def cmd_convert_to_tensor(args: argparse.Namespace):
    """Call convert_niigz_to_tensor with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    print_separator("START CONVERT TO TENSOR", sep_type="START")
    convert_niigz_to_tensor(CONFIG)
    print_separator("END CONVERT TO TENSOR", sep_type="START")


def cmd_train(args: argparse.Namespace):
    """Train the SSL encoder, survival model, or both."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    print_separator("START TRAINING", sep_type="START")
    train_3d_vit(CONFIG)
    print_separator("END TRAINING", sep_type="START")

# ======================================================================================
#                                       ARGUMENTS
# ======================================================================================
if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(prog="app", description="Main Application CLI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--config", type=str, default=None, help="Name of the config file at configs/ (default: None, but config.yaml exists)")
    subparsers = parser.add_subparsers(dest="function", required=True)


    # ======================================================================================
    #                                       train
    # ======================================================================================
    p_train_model = subparsers.add_parser("train", help="Train SSL, survival, or both")
    p_train_model.add_argument("--stage", type=str, choices=("ssl", "survival", "both"), default="survival", help="Which training stage to run")
    p_train_model.add_argument("--ssl-checkpoint-name", type=str, default="ssl_checkpoint.pt", help="Filename used when saving the SSL checkpoint")
    p_train_model.add_argument("--ssl-checkpoint", type=str, default=None, help="Optional SSL checkpoint filename to load into the survival model")
    p_train_model.add_argument("--ssl-epochs", type=int, default=50, help="Number of SSL epochs")
    p_train_model.add_argument("--ssl-batch-size", type=int, default=4, help="SSL batch size")
    p_train_model.add_argument("--ssl-learning-rate", type=float, default=1e-4, help="SSL learning rate")
    p_train_model.add_argument("--ssl-weight-decay", type=float, default=1e-4, help="SSL weight decay")
    p_train_model.add_argument("--ssl-num-workers", type=int, default=2, help="SSL dataloader workers")
    p_train_model.add_argument("--ssl-temperature", type=float, default=0.5, help="NT-Xent temperature")
    p_train_model.add_argument("--ssl-proj-dim", type=int, default=128, help="SSL projection dimension")
    p_train_model.add_argument("--ssl-noise-std", type=float, default=0.05, help="SSL augmentation noise std")
    p_train_model.add_argument("--ssl-crop-scale", type=float, default=0.85, help="SSL random crop scale")
    p_train_model.add_argument("--checkpoint-name", type=str, default="survival_checkpoint.pt", help="Filename used when saving the survival checkpoint")
    p_train_model.add_argument("--epochs", type=int, default=50, help="Number of survival training epochs")
    p_train_model.add_argument("--batch-size", type=int, default=4, help="Survival batch size")
    p_train_model.add_argument("--learning-rate", type=float, default=1e-4, help="Survival learning rate")
    p_train_model.add_argument("--weight-decay", type=float, default=1e-4, help="Survival weight decay")
    p_train_model.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing for the survival loss")
    p_train_model.add_argument("--num-workers", type=int, default=2, help="Survival dataloader workers")
    p_train_model.add_argument("--embed-dim", type=int, default=256, help="Shared token dimension")
    p_train_model.add_argument("--num-heads", type=int, default=8, help="Attention heads")
    p_train_model.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    p_train_model.add_argument("--patch-size", type=int, default=16, help="ViT patch size")
    p_train_model.add_argument("--vit-depth", type=int, default=4, help="ViT depth")
    p_train_model.add_argument("--vol-size", type=int, default=96, help="Input volume size")
    p_train_model.add_argument("--tabular-tokens", type=int, default=8, help="Number of tabular tokens")
    p_train_model.add_argument("--tabular-hidden", type=int, default=128, help="Tabular tokenizer hidden size")
    p_train_model.add_argument("--freeze-encoder", action="store_true", help="Freeze the image encoder when training survival")
    p_train_model.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    p_train_model.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    p_train_model.add_argument("--early-stopping-min-delta", type=float, default=1e-4, help="Early stopping minimum delta")
    p_train_model.set_defaults(func=cmd_train)

    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)
