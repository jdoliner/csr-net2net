"""Compute-budget experiment: does training small then scaling up beat training big from scratch?

Protocols (all get the same total optimizer steps):
- CSR:     Train at d_ff_init for `expand_at_step` steps, CSR-expand all layers to d_ff_max, train remaining
- Net2Net: Same schedule but using duplication
- Scratch: Train at d_ff_max for all steps

The key variable is `expand_at_step` — the fraction of total compute spent
at the small scale before expanding. Try different values (e.g., 10%, 25%, 50%)
to find the sweet spot.

Usage:
    uv run python run_scaleup.py --total-steps 5000 --expand-at-step 1000
    uv run python run_scaleup.py --total-steps 10000 --expand-at-frac 0.25
"""

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train import compute_scaled_lr
from transformer_model import TransformerLM
from transformer_ops import (
    expand_transformer_mlp_continuous,
    expand_transformer_mlp_net2net,
)
from transformer_train import evaluate_lm, load_wikitext103

logger = logging.getLogger(__name__)


@dataclass
class ScaleupConfig:
    """Configuration for the compute-budget scaleup experiment."""

    # Model
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff_init: int = 1024
    d_ff_max: int = 4096
    max_seq_len: int = 256
    dropout: float = 0.1

    # Training
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 32
    grad_accum_steps: int = 4
    seed: int = 42

    # Compute budget
    total_steps: int = 10000       # Total optimizer steps for all protocols
    expand_at_step: int = 2500     # When to expand (CSR/Net2Net)

    # Eval
    eval_every_steps: int = 200    # Evaluate every N optimizer steps

    # Data
    max_train_seqs: int = 0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_model(config: ScaleupConfig, d_ff: int) -> TransformerLM:
    """Create a TransformerLM with uniform d_ff across all layers."""
    return TransformerLM(
        vocab_size=50257,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff_list=[d_ff] * config.n_layers,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    )


def expand_all_layers(
    model: TransformerLM,
    new_d_ff: int,
    optimizer: torch.optim.Optimizer,
    method: str,
) -> TransformerLM:
    """Expand all MLP blocks from current d_ff to new_d_ff.

    Each single-layer expansion doubles d_ff. If the target requires multiple
    doublings (e.g., 1024 -> 4096 = 2 doublings), we repeat until all layers
    reach the target.
    """
    while True:
        all_at_target = True
        for layer_idx in range(model.n_layers):
            current_d_ff = model.d_ff_list[layer_idx]
            if current_d_ff < new_d_ff:
                all_at_target = False
                if method == "continuous":
                    model = expand_transformer_mlp_continuous(model, layer_idx, optimizer)
                else:
                    model = expand_transformer_mlp_net2net(model, layer_idx, optimizer)
        if all_at_target:
            break
    return model


class InfiniteDataIter:
    """Infinite reshuffling iterator over tokenized sequences."""

    def __init__(self, tokens: torch.Tensor, batch_size: int):
        self.tokens = tokens
        self.batch_size = batch_size
        self.n = len(tokens)
        self.pos = self.n  # Start exhausted to trigger shuffle
        self.indices = torch.arange(self.n)

    def next_batch(self) -> torch.Tensor:
        if self.pos + self.batch_size > self.n:
            self.indices = torch.randperm(self.n)
            self.pos = 0
        batch = self.tokens[self.indices[self.pos : self.pos + self.batch_size]]
        self.pos += self.batch_size
        return batch


def train_for_steps(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    data_iter: InfiniteDataIter,
    val_tokens: torch.Tensor,
    config: ScaleupConfig,
    writer: SummaryWriter,
    tag_prefix: str,
    num_steps: int,
    start_step: int = 0,
) -> tuple[int, float, float]:
    """Train for a fixed number of optimizer steps.

    Evaluates every eval_every_steps and logs to TensorBoard.

    Returns (steps_done, final_val_loss, final_ppl).
    """
    device = config.device
    global_step = start_step
    target_step = start_step + num_steps
    accum_count = 0

    model.train()
    optimizer.zero_grad()

    pbar = tqdm(
        total=num_steps,
        desc=f"[{tag_prefix}] Steps {start_step}->{target_step}",
        leave=False,
    )

    val_loss = float("inf")
    ppl = float("inf")

    while global_step < target_step:
        batch = data_iter.next_batch().to(device)
        if batch.shape[0] == 0:
            continue
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        loss = loss / config.grad_accum_steps
        loss.backward()

        accum_count += 1
        if accum_count >= config.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            accum_count = 0
            pbar.update(1)

            # Log train loss
            if global_step % 50 == 0:
                writer.add_scalar(
                    f"{tag_prefix}/Loss/Train",
                    loss.item() * config.grad_accum_steps,
                    global_step,
                )

            # Periodic evaluation
            if global_step % config.eval_every_steps == 0 or global_step >= target_step:
                val_loss, ppl = evaluate_lm(model, val_tokens, config.batch_size, device)
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar(f"{tag_prefix}/Loss/Val", val_loss, global_step)
                writer.add_scalar(f"{tag_prefix}/Perplexity/Val", ppl, global_step)
                writer.add_scalar(f"{tag_prefix}/LR", current_lr, global_step)

                logger.info(
                    f"[{tag_prefix}] Step {global_step}/{target_step} | "
                    f"val_loss={val_loss:.4f} ppl={ppl:.2f} | "
                    f"lr={current_lr:.6f}"
                )

    pbar.close()

    # Final eval if we didn't just do one
    if global_step % config.eval_every_steps != 0:
        val_loss, ppl = evaluate_lm(model, val_tokens, config.batch_size, device)
        writer.add_scalar(f"{tag_prefix}/Loss/Val", val_loss, global_step)
        writer.add_scalar(f"{tag_prefix}/Perplexity/Val", ppl, global_step)

    return global_step, val_loss, ppl


def run_protocol(
    name: str,
    method: str | None,
    config: ScaleupConfig,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    writer: SummaryWriter,
) -> dict:
    """Run a single protocol (CSR, Net2Net, or Scratch).

    Args:
        name: Display name for logging.
        method: 'continuous', 'net2net', or None for scratch.
        config: Experiment configuration.
        train_tokens: Training data.
        val_tokens: Validation data.
        writer: TensorBoard writer.

    Returns:
        Dict with results.
    """
    device = config.device
    is_scratch = method is None

    if is_scratch:
        d_ff = config.d_ff_max
        tag = "Scratch"
    else:
        d_ff = config.d_ff_init
        tag = name

    model = make_model(config, d_ff).to(device)
    params = sum(p.numel() for p in model.parameters())

    # Sqrt-scaled LR based on model size
    base_params = sum(p.numel() for p in make_model(config, config.d_ff_init).parameters())
    lr = compute_scaled_lr(config.lr, base_params, params)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config.weight_decay, betas=(0.9, 0.95),
    )

    data_iter = InfiniteDataIter(train_tokens, config.batch_size)

    logger.info(
        f"[{tag}] Starting | d_ff={d_ff} | params={params:,} | lr={lr:.6f} | "
        f"total_steps={config.total_steps}"
    )

    start_time = time.time()

    if is_scratch:
        # Train for all steps at full size
        final_step, val_loss, ppl = train_for_steps(
            model, optimizer, data_iter, val_tokens, config, writer, tag,
            num_steps=config.total_steps,
        )
    else:
        # Phase 1: Train small
        pre_steps = config.expand_at_step
        logger.info(f"[{tag}] Phase 1: Training small model for {pre_steps} steps")

        step, val_loss, ppl = train_for_steps(
            model, optimizer, data_iter, val_tokens, config, writer, tag,
            num_steps=pre_steps,
        )

        # Evaluate right before expansion
        val_loss_pre, ppl_pre = evaluate_lm(model, val_tokens, config.batch_size, device)
        logger.info(
            f"[{tag}] Pre-expansion: val_loss={val_loss_pre:.4f} ppl={ppl_pre:.2f} | "
            f"d_ff={model.d_ff_list}"
        )
        writer.add_scalar(f"{tag}/Val_Loss_PreExpand", val_loss_pre, step)
        writer.add_scalar(f"{tag}/Perplexity_PreExpand", ppl_pre, step)

        # Expand all layers
        logger.info(
            f"[{tag}] === EXPANDING all layers d_ff {config.d_ff_init} -> {config.d_ff_max} ==="
        )
        model = expand_all_layers(model, config.d_ff_max, optimizer, method)

        new_params = sum(p.numel() for p in model.parameters())

        # Update LR for new model size
        new_lr = compute_scaled_lr(config.lr, base_params, new_params)
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        val_loss_post, ppl_post = evaluate_lm(model, val_tokens, config.batch_size, device)
        logger.info(
            f"[{tag}] Post-expansion: val_loss={val_loss_post:.4f} ppl={ppl_post:.2f} | "
            f"d_ff={model.d_ff_list} | params={new_params:,} | lr={new_lr:.6f}"
        )
        writer.add_scalar(f"{tag}/Val_Loss_PostExpand", val_loss_post, step)
        writer.add_scalar(f"{tag}/Perplexity_PostExpand", ppl_post, step)
        writer.add_scalar(f"{tag}/Params", new_params, step)

        # Phase 2: Train large for remaining steps
        remaining = config.total_steps - pre_steps
        logger.info(f"[{tag}] Phase 2: Training expanded model for {remaining} steps")

        final_step, val_loss, ppl = train_for_steps(
            model, optimizer, data_iter, val_tokens, config, writer, tag,
            num_steps=remaining, start_step=step,
        )

    elapsed = time.time() - start_time
    final_loss, final_ppl = evaluate_lm(model, val_tokens, config.batch_size, device)

    logger.info(
        f"[{tag}] Final: val_loss={final_loss:.4f} ppl={final_ppl:.2f} | "
        f"d_ff={model.d_ff_list} | time={elapsed:.1f}s"
    )

    return {
        "protocol": name,
        "final_val_loss": final_loss,
        "final_ppl": final_ppl,
        "elapsed_time": elapsed,
        "final_d_ff": model.d_ff_list,
    }


def main():
    parser = argparse.ArgumentParser(description="CSR Scaleup Compute-Budget Experiment")
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
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--expand-at-step", type=int, default=0,
                        help="Step at which to expand (0 = use --expand-at-frac)")
    parser.add_argument("--expand-at-frac", type=float, default=0.25,
                        help="Fraction of total steps before expansion (if --expand-at-step=0)")
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--max-train-seqs", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--log-dir", type=str, default="outputs/scaleup_runs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    expand_at = args.expand_at_step if args.expand_at_step > 0 else int(args.total_steps * args.expand_at_frac)

    config = ScaleupConfig(
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
        total_steps=args.total_steps,
        expand_at_step=expand_at,
        eval_every_steps=args.eval_every_steps,
        max_train_seqs=args.max_train_seqs,
        device=args.device,
    )

    logger.info(f"Config: {config}")
    logger.info(
        f"Budget: {config.total_steps} total steps | "
        f"Expand at step {config.expand_at_step} "
        f"({config.expand_at_step / config.total_steps * 100:.0f}%)"
    )

    # Load data
    train_tokens, val_tokens = load_wikitext103(
        config.max_seq_len, config.batch_size, config.device, config.max_train_seqs,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    writer = SummaryWriter(log_dir=log_dir)

    all_results = {}

    for name, method in [("CSR", "continuous"), ("Net2Net", "net2net"), ("Scratch", None)]:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {name}")
        logger.info(f"{'='*60}")

        results = run_protocol(name, method, config, train_tokens, val_tokens, writer)
        all_results[name] = results

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(
        f"  Budget: {config.total_steps} steps | "
        f"Expand at step {config.expand_at_step} "
        f"({config.expand_at_step / config.total_steps * 100:.0f}%)"
    )
    logger.info(
        f"  Small: d_ff={config.d_ff_init} | Large: d_ff={config.d_ff_max}"
    )
    logger.info("")

    for name, results in all_results.items():
        logger.info(
            f"  {name:>10s}: ppl={results['final_ppl']:.2f}  "
            f"val_loss={results['final_val_loss']:.4f}  "
            f"time={results['elapsed_time']:.1f}s  "
            f"d_ff={results['final_d_ff']}"
        )

    writer.close()
    logger.info(f"\nTensorBoard: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
