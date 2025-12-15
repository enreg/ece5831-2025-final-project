"""Helper to download GTSRB via torchvision."""
import argparse
from pathlib import Path

from data import DataConfig, get_datasets


def main():
    parser = argparse.ArgumentParser(description="Download GTSRB dataset")
    parser.add_argument("--data-root", type=str, default="data", help="Where to store the dataset")
    args = parser.parse_args()

    cfg = DataConfig(data_root=Path(args.data_root), download=True)
    train_set, val_set, test_set = get_datasets(cfg)
    print(f"Downloaded: train={len(train_set)+len(val_set)}, test={len(test_set)} into {cfg.data_root}")


if __name__ == "__main__":
    main()
