"""Main entry point: runs training protocols and compares results.

Supports two modes:
- Uniform expansion: all layers expand together (original experiment)
- Targeted expansion: CSR picks which layer to expand based on similarity

Usage:
    uv run python main.py                          # targeted mode (default)
    uv run python main.py --mode uniform           # uniform expansion mode
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

from models import MLP
from train import TrainConfig, train_protocol, train_targeted

logger = logging.getLogger(__name__)


def load_cifar10(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 using HuggingFace datasets, return train/val DataLoaders."""
    logger.info("Loading CIFAR-10 dataset...")
    ds = load_dataset("cifar10")

    def to_tensors(split) -> TensorDataset:
        import numpy as np

        images_np = np.array([np.array(img) for img in split["img"]])
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


def run_targeted(args, config: TrainConfig, train_loader, val_loader, writer):
    """Run targeted expansion experiment: CSR first, then Net2Net replay, then Scratch."""
    all_results = {}

    # --- 1. CSR Targeted (dynamic layer selection) ---
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    logger.info(f"\n{'='*60}")
    logger.info("Running protocol: CSR Targeted (dynamic layer selection)")
    logger.info(f"{'='*60}")

    start_time = time.time()
    csr_results = train_targeted(
        method="continuous",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        writer=writer,
        tag_prefix="CSR_Targeted",
    )
    csr_results["elapsed_time"] = time.time() - start_time
    all_results["csr_targeted"] = csr_results

    expansion_plan = csr_results["expansion_plan"]
    total_epochs_used = csr_results["total_epochs"]
    final_widths = csr_results["final_widths"]

    logger.info(
        f"[CSR_Targeted] Completed in {csr_results['elapsed_time']:.1f}s | "
        f"Final val_acc={csr_results['final_val_acc']:.4f} "
        f"val_loss={csr_results['final_val_loss']:.4f} | "
        f"widths={final_widths} | epochs={total_epochs_used}"
    )
    logger.info(f"[CSR_Targeted] Expansion plan ({len(expansion_plan)} steps):")
    for i, step in enumerate(expansion_plan):
        logger.info(f"  Step {i+1}: layer {step.layer_idx} {step.old_width}->{step.new_width}")

    # --- 2. Net2Net Targeted (replay same plan) ---
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    logger.info(f"\n{'='*60}")
    logger.info("Running protocol: Net2Net Targeted (replay CSR plan)")
    logger.info(f"{'='*60}")

    start_time = time.time()
    n2n_results = train_targeted(
        method="net2net",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        writer=writer,
        tag_prefix="Net2Net_Targeted",
        expansion_plan=expansion_plan,
    )
    n2n_results["elapsed_time"] = time.time() - start_time
    all_results["net2net_targeted"] = n2n_results

    logger.info(
        f"[Net2Net_Targeted] Completed in {n2n_results['elapsed_time']:.1f}s | "
        f"Final val_acc={n2n_results['final_val_acc']:.4f} "
        f"val_loss={n2n_results['final_val_loss']:.4f}"
    )

    # --- 3. Scratch (balanced architecture at final_width, same total epochs) ---
    # Use balanced architecture rather than CSR's discovered one, since CSR's
    # architecture is tuned to the expansion path and trains poorly from scratch.
    balanced_widths = [config.final_width] * config.num_hidden_layers
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    logger.info(f"\n{'='*60}")
    logger.info(f"Running protocol: Scratch (widths={balanced_widths}, epochs={total_epochs_used})")
    logger.info(f"{'='*60}")

    scratch_model = MLP(
        input_dim=config.input_dim,
        hidden_widths=balanced_widths,
        dropout=config.dropout,
    )
    device = config.device
    scratch_model = scratch_model.to(device)
    scratch_optimizer = torch.optim.AdamW(
        scratch_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scratch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        scratch_optimizer, T_max=total_epochs_used, eta_min=1e-6
    )
    criterion = torch.nn.CrossEntropyLoss()

    start_time = time.time()
    from train import _train_epochs, evaluate

    global_step, val_loss, val_acc = _train_epochs(
        scratch_model, scratch_optimizer, scratch_scheduler,
        train_loader, val_loader, criterion, device,
        writer, "Scratch_Targeted", 0, total_epochs_used, total_epochs_used, 0,
    )
    final_val_loss, final_val_acc = evaluate(scratch_model, val_loader, criterion, device)
    elapsed = time.time() - start_time

    scratch_results = {
        "protocol": "scratch",
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "expansion_events": [],
        "elapsed_time": elapsed,
        "total_epochs": total_epochs_used,
        "final_widths": tuple(balanced_widths),
    }
    all_results["scratch_targeted"] = scratch_results

    logger.info(
        f"[Scratch_Targeted] Completed in {elapsed:.1f}s | "
        f"Final val_acc={final_val_acc:.4f} val_loss={final_val_loss:.4f}"
    )

    return all_results


def run_uniform(args, config: TrainConfig, train_loader, val_loader, writer):
    """Run uniform expansion experiment (original mode)."""
    protocol_configs = {
        "scratch": ("scratch", "Scratch"),
        "net2net": ("net2net", "Net2Net"),
        "continuous": ("continuous", "CSR"),
    }

    all_results = {}

    for protocol_name in args.protocols:
        protocol, tag_prefix = protocol_configs[protocol_name]

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

    return all_results


def main():
    parser = argparse.ArgumentParser(description="CSR-Net2Net Experiment")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-dim", type=int, default=3072)
    parser.add_argument(
        "--width-schedule",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],
        help="Hidden widths for each stage (uniform mode)",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument(
        "--epochs-per-stage",
        type=int,
        nargs="+",
        default=[15, 15, 15, 15, 30],
        help="Epochs per stage (uniform mode)",
    )
    parser.add_argument("--epochs-per-targeted-stage", type=int, default=15)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--log-dir", type=str, default="outputs/runs")
    parser.add_argument(
        "--mode",
        choices=["targeted", "uniform"],
        default="targeted",
        help="Expansion mode: targeted (similarity-based) or uniform (all layers)",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["scratch", "net2net", "continuous"],
        choices=["scratch", "net2net", "continuous"],
        help="Which protocols to run (uniform mode only)",
    )
    args = parser.parse_args()

    if args.mode == "uniform":
        assert len(args.width_schedule) == len(args.epochs_per_stage)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        input_dim=args.input_dim,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
        width_schedule=args.width_schedule,
        epochs_per_stage=args.epochs_per_stage,
        epochs_per_targeted_stage=args.epochs_per_targeted_stage,
        device=args.device,
    )

    logger.info(f"Config: {config}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Param budget: {config.param_budget:,}")

    train_loader, val_loader = load_cifar10(config.batch_size)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard log dir: {log_dir}")

    if args.mode == "targeted":
        all_results = run_targeted(args, config, train_loader, val_loader, writer)
    else:
        all_results = run_uniform(args, config, train_loader, val_loader, writer)

    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")

    for name, results in all_results.items():
        widths_str = ""
        if "final_widths" in results:
            widths_str = f"  widths={results['final_widths']}"
        epochs_str = ""
        if "total_epochs" in results:
            epochs_str = f"  epochs={results['total_epochs']}"

        logger.info(
            f"  {name:>20s}: val_acc={results['final_val_acc']:.4f}  "
            f"val_loss={results['final_val_loss']:.4f}  "
            f"time={results['elapsed_time']:.1f}s{widths_str}{epochs_str}"
        )
        for ev in results.get("expansion_events", []):
            layer_str = f" layer={ev.layer_idx}" if ev.layer_idx is not None else ""
            logger.info(
                f"{'':>22s}  expand {ev.width_before}->{ev.width_after}{layer_str}: "
                f"shock={ev.shock:+.4f}  acc_delta={ev.acc_delta:+.4f}"
            )

    writer.close()
    logger.info(f"\nTensorBoard logs: {log_dir}")
    logger.info(f"Run: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
