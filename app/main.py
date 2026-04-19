"""Main file for scripts with arguments and call other functions."""

import dotenv
import argparse
from src.config import Configuration
from maikol_utils.other_utils import args_to_dataclass
from maikol_utils.print_utils import print_separator

from scripts import convert_niigz_to_tensor, train_fast_resnet
from src.training.train_ssl import train_ssl
from src.training.train_survival import train_multimodal_survival

def cmd_convert_to_tensor(args: argparse.Namespace):
    """Call convert_niigz_to_tensor with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    print_separator("START CONVERT TO TENSOR", sep_type="START")
    convert_niigz_to_tensor(CONFIG)
    print_separator("END CONVERT TO TENSOR", sep_type="START")

def cmd_test(args):
    """Call test functions."""
    ...


def cmd_train_resnet_fast(args: argparse.Namespace):
    """Train lightweight 3D ResNet quickly with tensor inputs."""
    config: Configuration = args_to_dataclass(args, Configuration)
    print_separator("START FAST RESNET TRAINING", sep_type="START")
    train_fast_resnet(
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        input_size=args.input_size,
        base_channels=args.base_channels,
        use_amp=not args.no_amp,
    )
    print_separator("END FAST RESNET TRAINING", sep_type="START")


def cmd_train(args: argparse.Namespace):
    """Train the SSL encoder, survival model, or both."""
    config: Configuration = args_to_dataclass(args, Configuration)
    print_separator("START TRAINING", sep_type="START")

    if args.stage in ("ssl", "both"):
        train_ssl(
            config=config,
            epochs=args.ssl_epochs,
            batch_size=args.ssl_batch_size,
            learning_rate=args.ssl_learning_rate,
            weight_decay=args.ssl_weight_decay,
            num_workers=args.ssl_num_workers,
            embed_dim=args.embed_dim,
            patch_size=args.patch_size,
            vit_depth=args.vit_depth,
            num_heads=args.num_heads,
            dropout=args.dropout,
            vol_size=args.vol_size,
            temperature=args.ssl_temperature,
            proj_dim=args.ssl_proj_dim,
            noise_std=args.ssl_noise_std,
            crop_scale=args.ssl_crop_scale,
            checkpoint_name=args.ssl_checkpoint_name,
            enable_early_stopping=not args.no_early_stopping,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        )

    if args.stage in ("survival", "both"):
        ssl_checkpoint = args.ssl_checkpoint
        if ssl_checkpoint is None and args.stage == "both":
            ssl_checkpoint = args.ssl_checkpoint_name

        train_multimodal_survival(
            config=config,
            ssl_checkpoint=ssl_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            num_workers=args.num_workers,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            patch_size=args.patch_size,
            vit_depth=args.vit_depth,
            vol_size=args.vol_size,
            tabular_tokens=args.tabular_tokens,
            tabular_hidden=args.tabular_hidden,
            freeze_encoder=args.freeze_encoder,
            enable_early_stopping=not args.no_early_stopping,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            checkpoint_name=args.checkpoint_name,
        )

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
    #                                       read_extract
    # ======================================================================================
    p_read = subparsers.add_parser("convert-to-tensor", help="Convert NIfTI files to PyTorch tensors")
    p_read.set_defaults(func=cmd_convert_to_tensor)

    # ======================================================================================
    #                                       test
    # ======================================================================================
    p_test = subparsers.add_parser("test", help="Test script with any code")
    p_test.set_defaults(func=cmd_test)

    # ======================================================================================
    #                                 train-resnet-fast
    # ======================================================================================
    p_train = subparsers.add_parser("train-resnet-fast", help="Fast train a lightweight 3D ResNet")
    p_train.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p_train.add_argument("--batch-size", type=int, default=2, help="Batch size")
    p_train.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    p_train.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p_train.add_argument("--input-size", type=int, default=96, help="Resize all volumes to this cubic size")
    p_train.add_argument("--base-channels", type=int, default=16, help="Base channel width of ResNet")
    p_train.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    p_train.set_defaults(func=cmd_train_resnet_fast)

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
