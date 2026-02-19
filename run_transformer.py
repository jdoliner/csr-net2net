"""Entry point for the transformer experiment.

Usage:
    uv run python run_transformer.py
    uv run python run_transformer.py --patience 2 --max-epochs 50  # Quick test
"""

import argparse
import logging
import os
from datetime import datetime

import torch

from transformer_train import TransformerConfig, run_transformer_experiment


def main():
    parser = argparse.ArgumentParser(description="CSR Transformer Experiment")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff-init", type=int, default=1024)
    parser.add_argument("--d-ff-max", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-epochs-per-stage", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--min-improvement", type=float, default=0.001,
                        help="Relative improvement threshold for patience (0.001 = 0.1%%)")
    parser.add_argument("--max-train-seqs", type=int, default=0,
                        help="Cap training sequences (0=use all)")
    parser.add_argument("--steps-per-epoch", type=int, default=0,
                        help="Optimizer steps per epoch (0=full pass over data)")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--log-dir", type=str, default="outputs/transformer_runs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = TransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff_init=args.d_ff_init,
        d_ff_max=args.d_ff_max,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        seed=args.seed,
        patience=args.patience,
        min_epochs_per_stage=args.min_epochs_per_stage,
        max_epochs=args.max_epochs,
        min_improvement=args.min_improvement,
        max_train_seqs=args.max_train_seqs,
        steps_per_epoch=args.steps_per_epoch,
        device=args.device,
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Config: {config}")
    logger.info(f"Param budget: {config.param_budget:,}")
    logger.info(f"Base params: {config.base_params:,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)

    run_transformer_experiment(config, log_dir)


if __name__ == "__main__":
    main()
