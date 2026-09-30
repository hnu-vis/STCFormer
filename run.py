"""Train or test STCFormer and the included French weather baselines.

Examples:
    python run.py train --model STCFormer --dataset French_Temperature_dim1 --horizon 24 --gpu 0
    python run.py test --model STCFormer --dataset French_Temperature_dim1 --horizon 24 --gpu 0
"""

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="French weather forecasting")
    parser.add_argument("action", choices=("train", "test"))
    parser.add_argument("--model", choices=("STCFormer", "DLinear", "PatchTST", "TimeXer"), default="STCFormer")
    parser.add_argument("--dataset", choices=(
        "French_Temperature_dim1", "French_U_Wind_dim1", "French_V_Wind_dim1"
    ), default="French_Temperature_dim1")
    parser.add_argument("--horizon", type=int, choices=(24, 72), default=24)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint", default="", help="Test checkpoint; defaults to best validation MAE")
    args = parser.parse_args()

    os.chdir(ROOT)
    os.environ.update({
        "STC_MODEL": args.model,
        "STC_DATASET": args.dataset,
        "STC_HORIZON": str(args.horizon),
        "STC_SEED": str(args.seed),
        "STC_EPOCHS": str(args.epochs),
        "STC_BATCH_SIZE": str(args.batch_size),
    })
    import basicts

    config = "baselines/release_config.py"
    if args.action == "train":
        basicts.launch_training(config, args.gpu)
    else:
        basicts.launch_evaluation(
            config, args.checkpoint, device_type="gpu", gpus=args.gpu,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
