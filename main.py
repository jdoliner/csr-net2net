"""Main entry point: runs all three training protocols and compares results.

Usage:
    uv run python main.py
    uv run python main.py --epochs-pre 5 --epochs-total 15  # Quick test
"""

import argparse
import logging
import os
import time
from datetime import datetime

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from train import TrainConfig, train_protocol

logger = logging.getLogger(__name__)


def load_cifar10(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 using HuggingFace datasets, return train/val DataLoaders."""
    logger.info("Loading CIFAR-10 dataset...")
    ds = load_dataset("cifar10")

    def to_tensors(split) -> TensorDataset:
        import numpy as np

        # CIFAR-10 images are 32x32x3 PIL Images
        images_np = np.array([np.array(img) for img in split["img"]])
        # Normalize to [0, 1] and flatten to [N, 3072]
        images = torch.tensor(images_np, dtype=torch.float32) / 255.0
        labels = torch.tensor(split["label"], dtype=torch.long)
        return TensorDataset(images, labels)

    train_ds = to_tensors(ds["train"])
    val_ds = to_tensors(ds["test"])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="CSR-Net2Net Experiment")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs-pre", type=int, default=30)
    parser.add_argument("--epochs-total", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-dim", type=int, default=3072)
    parser.add_argument("--hidden-small", type=int, default=256)
    parser.add_argument("--hidden-large", type=int, default=512)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--log-dir", type=str, default="outputs/runs")
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["scratch", "net2net", "continuous"],
        choices=["scratch", "net2net", "continuous"],
        help="Which protocols to run",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs_pre=args.epochs_pre,
        epochs_total=args.epochs_total,
        seed=args.seed,
        input_dim=args.input_dim,
        hidden_small=args.hidden_small,
        hidden_large=args.hidden_large,
        device=args.device,
    )

    logger.info(f"Config: {config}")
    logger.info(f"Device: {config.device}")

    # Load data once, shared across all protocols
    train_loader, val_loader = load_cifar10(config.batch_size)

    # TensorBoard setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard log dir: {log_dir}")

    # Protocol mapping
    protocol_configs = {
        "scratch": ("scratch", "Scratch"),
        "net2net": ("net2net", "Net2Net"),
        "continuous": ("continuous", "CSR"),
    }

    all_results = {}

    for protocol_name in args.protocols:
        protocol, tag_prefix = protocol_configs[protocol_name]

        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        logger.info(f"\n{'='*60}")
        logger.info(f"Running protocol: {tag_prefix} ({protocol})")
        logger.info(f"{'='*60}")

        start_time = time.time()
        results = train_protocol(
            protocol=protocol,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            writer=writer,
            tag_prefix=tag_prefix,
        )
        elapsed = time.time() - start_time

        results["elapsed_time"] = elapsed
        all_results[protocol_name] = results

        logger.info(
            f"[{tag_prefix}] Completed in {elapsed:.1f}s | "
            f"Final val_acc={results['final_val_acc']:.4f} "
            f"val_loss={results['final_val_loss']:.4f}"
        )

    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")

    for name, results in all_results.items():
        tag = protocol_configs[name][1]
        logger.info(
            f"  {tag:>10s}: val_acc={results['final_val_acc']:.4f}  "
            f"val_loss={results['final_val_loss']:.4f}  "
            f"time={results['elapsed_time']:.1f}s"
        )
        if results["expansion_event"] is not None:
            ev = results["expansion_event"]
            logger.info(
                f"{'':>12s}  expansion_shock={ev.shock:+.4f}  "
                f"acc_delta={ev.acc_after_first_epoch - ev.acc_before:+.4f}"
            )

    writer.close()
    logger.info(f"\nTensorBoard logs: {log_dir}")
    logger.info(f"Run: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
