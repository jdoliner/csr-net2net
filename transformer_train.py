"""Training and data loading for the transformer language model experiment.

Includes:
- WikiText-103 data loading with GPT-2 tokenization
- Patience-based training loop with perplexity metric
- Targeted expansion protocol for transformer MLP blocks
"""

import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import tiktoken
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train import ExpansionEvent, TargetedExpansionStep, compute_scaled_lr
from transformer_model import TransformerLM
from transformer_ops import (
    expand_transformer_mlp_continuous,
    expand_transformer_mlp_net2net,
    params_after_doubling_mlp,
    transformer_layer_similarity_scores,
)

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for the transformer experiment."""

    # Model
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff_init: int = 1024  # Initial d_ff (4 * d_model)
    d_ff_max: int = 4096   # Maximum d_ff for param budget
    max_seq_len: int = 256
    dropout: float = 0.1

    # Training
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 32
    grad_accum_steps: int = 4  # Effective batch = batch_size * grad_accum_steps
    seed: int = 42

    # Patience-based training
    patience: int = 3
    min_epochs_per_stage: int = 2
    max_epochs: int = 200

    # Eval
    eval_interval_steps: int = 200  # Evaluate every N optimizer steps

    # Data
    max_train_seqs: int = 0  # 0 = use all data, >0 = cap training sequences

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def param_budget(self) -> int:
        """Parameter count with all layers at d_ff_max."""
        ref = TransformerLM(
            vocab_size=50257,  # GPT-2 vocab size
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff_list=[self.d_ff_max] * self.n_layers,
            max_seq_len=self.max_seq_len,
        )
        return sum(p.numel() for p in ref.parameters())

    @property
    def base_params(self) -> int:
        ref = TransformerLM(
            vocab_size=50257,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff_list=[self.d_ff_init] * self.n_layers,
            max_seq_len=self.max_seq_len,
        )
        return sum(p.numel() for p in ref.parameters())


def load_wikitext103(
    seq_len: int, batch_size: int, device: str, max_train_seqs: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load WikiText-103 and tokenize with GPT-2 tokenizer.

    Tokenizes in chunks for efficiency and caches the result to disk.
    Returns pre-tokenized train and val tensors of shape [N, seq_len].
    """
    cache_dir = "outputs/data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"wikitext103_gpt2_seq{seq_len}.pt")

    if os.path.exists(cache_file):
        logger.info(f"Loading cached tokenized data from {cache_file}")
        data = torch.load(cache_file, weights_only=True)
        train_tokens = data["train"]
        val_tokens = data["val"]
    else:
        logger.info("Loading WikiText-103...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        enc = tiktoken.get_encoding("gpt2")

        def tokenize_split(split, split_name: str) -> torch.Tensor:
            logger.info(f"Tokenizing {split_name} ({len(split)} examples)...")
            all_tokens = []
            texts = split["text"]

            # Tokenize in batches of 10000 examples
            chunk_size = 10000
            for start in range(0, len(texts), chunk_size):
                chunk = texts[start : start + chunk_size]
                text = "\n".join(t for t in chunk if t.strip())
                if text:
                    tokens = enc.encode(text, allowed_special=set())
                    all_tokens.extend(tokens)
                if (start // chunk_size) % 10 == 0:
                    logger.info(
                        f"  {split_name}: {start:,}/{len(texts):,} examples, "
                        f"{len(all_tokens):,} tokens so far"
                    )

            tokens = torch.tensor(all_tokens, dtype=torch.long)
            n_seqs = len(tokens) // seq_len
            tokens = tokens[: n_seqs * seq_len].view(n_seqs, seq_len)
            return tokens

        train_tokens = tokenize_split(ds["train"], "train")
        val_tokens = tokenize_split(ds["validation"], "val")

        logger.info(f"Caching tokenized data to {cache_file}")
        torch.save({"train": train_tokens, "val": val_tokens}, cache_file)

    if max_train_seqs > 0 and train_tokens.shape[0] > max_train_seqs:
        logger.info(f"Capping training data from {train_tokens.shape[0]} to {max_train_seqs} sequences")
        train_tokens = train_tokens[:max_train_seqs]

    logger.info(
        f"Train: {train_tokens.shape[0]} sequences of length {seq_len} "
        f"({train_tokens.numel():,} tokens)"
    )
    logger.info(
        f"Val: {val_tokens.shape[0]} sequences of length {seq_len} "
        f"({val_tokens.numel():,} tokens)"
    )

    return train_tokens, val_tokens


def evaluate_lm(
    model: TransformerLM, val_tokens: torch.Tensor, batch_size: int, device: str
) -> tuple[float, float]:
    """Evaluate language model on validation set.

    Returns (val_loss, perplexity).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(val_tokens), batch_size):
            batch = val_tokens[i : i + batch_size].to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    return avg_loss, perplexity


def train_transformer_targeted(
    method: Literal["continuous", "net2net"],
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    config: TransformerConfig,
    writer: SummaryWriter,
    tag_prefix: str,
    expansion_plan: list[TargetedExpansionStep] | None = None,
) -> dict:
    """Train a transformer with targeted MLP expansion.

    Patience-based training: train until validation loss plateaus,
    then expand the MLP block with lowest similarity score.
    LR is reset to base_lr after each expansion.

    Args:
        method: 'continuous' for CSR, 'net2net' for Net2Net.
        train_tokens: [N_train, seq_len] tokenized training data.
        val_tokens: [N_val, seq_len] tokenized validation data.
        config: Experiment configuration.
        writer: TensorBoard writer.
        tag_prefix: Prefix for logging.
        expansion_plan: If provided, replay this plan.

    Returns:
        Dict with results.
    """
    device = config.device
    param_budget = config.param_budget
    base_lr = config.lr

    # Initialize model
    model = TransformerLM(
        vocab_size=50257,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff_list=[config.d_ff_init] * config.n_layers,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    ).to(device)

    current_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    dynamic = expansion_plan is None
    if dynamic:
        expansion_plan = []

    global_step = 0
    epoch_counter = 0
    expansion_events: list[ExpansionEvent] = []
    exp_idx = 0
    n_train = len(train_tokens)

    logger.info(
        f"[{tag_prefix}] Starting: {method} | d_ff={model.d_ff_list} | "
        f"params={current_params:,} | budget={param_budget:,} | lr={base_lr}"
    )

    def train_until_plateau() -> tuple[float, float]:
        """Train epochs until plateau. Returns (val_loss, perplexity)."""
        nonlocal global_step, epoch_counter

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        epochs_trained = 0
        val_loss = float("inf")
        ppl = float("inf")

        while epochs_trained < config.max_epochs - epoch_counter:
            # Shuffle training data
            perm = torch.randperm(n_train)
            shuffled = train_tokens[perm]

            model.train()
            epoch_loss = 0.0
            epoch_tokens = 0
            optimizer.zero_grad()

            pbar = tqdm(
                range(0, n_train, config.batch_size),
                desc=f"[{tag_prefix}] Epoch {epoch_counter + 1}",
                leave=False,
            )

            accum_count = 0
            for i in pbar:
                batch = shuffled[i : i + config.batch_size].to(device)
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

                epoch_loss += loss.item() * config.grad_accum_steps * targets.numel()
                epoch_tokens += targets.numel()

                accum_count += 1
                if accum_count >= config.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    accum_count = 0

                    # Log training loss periodically
                    if global_step % 50 == 0:
                        writer.add_scalar(
                            f"{tag_prefix}/Loss/Train",
                            loss.item() * config.grad_accum_steps,
                            global_step,
                        )

                pbar.set_postfix(loss=f"{loss.item() * config.grad_accum_steps:.4f}")

            # Handle remaining accumulated gradients
            if accum_count > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # End of epoch evaluation
            avg_train_loss = epoch_loss / max(epoch_tokens, 1)
            val_loss, ppl = evaluate_lm(model, val_tokens, config.batch_size, device)

            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar(f"{tag_prefix}/Loss/Val", val_loss, global_step)
            writer.add_scalar(f"{tag_prefix}/Perplexity/Val", ppl, global_step)
            writer.add_scalar(f"{tag_prefix}/LR", current_lr, global_step)

            epoch_counter += 1
            epochs_trained += 1

            logger.info(
                f"[{tag_prefix}] Epoch {epoch_counter} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} ppl={ppl:.2f} | "
                f"lr={current_lr:.6f} | "
                f"patience={epochs_without_improvement}/{config.patience}"
            )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                epochs_trained >= config.min_epochs_per_stage
                and epochs_without_improvement >= config.patience
            ):
                logger.info(
                    f"[{tag_prefix}] Plateau after {epochs_trained} epochs "
                    f"(best={best_val_loss:.4f})"
                )
                break

        return val_loss, ppl

    # --- Initial training ---
    val_loss, ppl = train_until_plateau()

    # --- Expansion loop ---
    while epoch_counter < config.max_epochs:
        if dynamic:
            scores = transformer_layer_similarity_scores(model)
            layer_order = sorted(range(len(scores)), key=lambda i: scores[i])

            chosen_layer = None
            for li in layer_order:
                new_params = params_after_doubling_mlp(model, li)
                if new_params <= param_budget:
                    chosen_layer = li
                    break

            if chosen_layer is None:
                logger.info(
                    f"[{tag_prefix}] No MLP can be doubled within budget. "
                    f"params={current_params:,}, budget={param_budget:,}"
                )
                break

            old_d_ff = model.d_ff_list[chosen_layer]
            new_d_ff = old_d_ff * 2
            step = TargetedExpansionStep(
                layer_idx=chosen_layer, old_width=old_d_ff, new_width=new_d_ff,
            )
            expansion_plan.append(step)

            logger.info(
                f"[{tag_prefix}] Similarity scores: "
                + ", ".join(f"L{i}={s:.4f}" for i, s in enumerate(scores))
            )
            logger.info(
                f"[{tag_prefix}] Selected layer {chosen_layer} "
                f"(sim={scores[chosen_layer]:.4f}, d_ff {old_d_ff}->{new_d_ff})"
            )
        else:
            if exp_idx >= len(expansion_plan):
                break
            step = expansion_plan[exp_idx]

        # Pre-expansion eval
        val_loss_before, ppl_before = evaluate_lm(
            model, val_tokens, config.batch_size, device
        )

        logger.info(
            f"[{tag_prefix}] === EXPANSION {exp_idx+1} at epoch {epoch_counter}: "
            f"layer {step.layer_idx} d_ff {step.old_width}->{step.new_width} ==="
        )
        logger.info(
            f"[{tag_prefix}] Pre: val_loss={val_loss_before:.4f} ppl={ppl_before:.2f} "
            f"| d_ff={model.d_ff_list}"
        )

        # Expand
        if method == "continuous":
            model = expand_transformer_mlp_continuous(model, step.layer_idx, optimizer)
        else:
            model = expand_transformer_mlp_net2net(model, step.layer_idx, optimizer)

        current_params = sum(p.numel() for p in model.parameters())

        # Reset LR to base
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr

        val_loss_after, ppl_after = evaluate_lm(
            model, val_tokens, config.batch_size, device
        )
        logger.info(
            f"[{tag_prefix}] Post: val_loss={val_loss_after:.4f} ppl={ppl_after:.2f} "
            f"| d_ff={model.d_ff_list} | params={current_params:,} | lr={base_lr}"
        )

        writer.add_scalar(f"{tag_prefix}/Val_Loss_PreExpand", val_loss_before, global_step)
        writer.add_scalar(f"{tag_prefix}/Val_Loss_PostExpand", val_loss_after, global_step)
        writer.add_scalar(f"{tag_prefix}/Perplexity_PreExpand", ppl_before, global_step)
        writer.add_scalar(f"{tag_prefix}/Params", current_params, global_step)

        # Train until plateau
        val_loss, ppl = train_until_plateau()

        event = ExpansionEvent(
            expansion_index=exp_idx,
            step=global_step,
            epoch=epoch_counter,
            width_before=step.old_width,
            width_after=step.new_width,
            loss_before=val_loss_before,
            loss_after_first_epoch=val_loss,
            acc_before=ppl_before,  # Reusing acc fields for perplexity
            acc_after_first_epoch=ppl,
            layer_idx=step.layer_idx,
        )
        expansion_events.append(event)

        logger.info(
            f"[{tag_prefix}] Expansion {exp_idx+1} result: "
            f"layer {step.layer_idx} d_ff {step.old_width}->{step.new_width} "
            f"loss_delta={event.shock:+.4f} ppl: {ppl_before:.2f}->{ppl:.2f}"
        )

        exp_idx += 1

    # Final eval
    final_loss, final_ppl = evaluate_lm(model, val_tokens, config.batch_size, device)

    logger.info(
        f"[{tag_prefix}] Final: d_ff={model.d_ff_list} | "
        f"params={sum(p.numel() for p in model.parameters()):,} | "
        f"epochs={epoch_counter} | val_loss={final_loss:.4f} ppl={final_ppl:.2f}"
    )

    return {
        "protocol": f"targeted_{method}",
        "final_val_loss": final_loss,
        "final_ppl": final_ppl,
        "expansion_events": expansion_events,
        "expansion_plan": expansion_plan,
        "total_epochs": epoch_counter,
        "final_d_ff": model.d_ff_list,
    }


def run_transformer_experiment(config: TransformerConfig, log_dir: str):
    """Run the full transformer experiment: CSR -> Net2Net replay -> Scratch."""
    writer = SummaryWriter(log_dir=log_dir)

    # Load data
    train_tokens, val_tokens = load_wikitext103(
        config.max_seq_len, config.batch_size, config.device, config.max_train_seqs,
    )

    all_results = {}

    # --- 1. CSR Targeted ---
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    logger.info(f"\n{'='*60}")
    logger.info("Running: CSR Targeted")
    logger.info(f"{'='*60}")

    start = time.time()
    csr_results = train_transformer_targeted(
        method="continuous",
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        config=config,
        writer=writer,
        tag_prefix="CSR",
    )
    csr_results["elapsed_time"] = time.time() - start
    all_results["csr"] = csr_results

    expansion_plan = csr_results["expansion_plan"]
    total_epochs_used = csr_results["total_epochs"]

    logger.info(f"[CSR] Plan ({len(expansion_plan)} steps):")
    for i, s in enumerate(expansion_plan):
        logger.info(f"  {i+1}: layer {s.layer_idx} d_ff {s.old_width}->{s.new_width}")

    # --- 2. Net2Net replay ---
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    logger.info(f"\n{'='*60}")
    logger.info("Running: Net2Net Targeted (replay)")
    logger.info(f"{'='*60}")

    start = time.time()
    n2n_results = train_transformer_targeted(
        method="net2net",
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        config=config,
        writer=writer,
        tag_prefix="Net2Net",
        expansion_plan=expansion_plan,
    )
    n2n_results["elapsed_time"] = time.time() - start
    all_results["net2net"] = n2n_results

    # --- 3. Scratch ---
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    logger.info(f"\n{'='*60}")
    logger.info("Running: Scratch")
    logger.info(f"{'='*60}")

    # Scratch: balanced d_ff at d_ff_max, sqrt-scaled LR, patience-based stopping
    scratch_model = TransformerLM(
        vocab_size=50257,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff_list=[config.d_ff_max] * config.n_layers,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    ).to(config.device)

    scratch_params = sum(p.numel() for p in scratch_model.parameters())
    scratch_lr = compute_scaled_lr(config.lr, config.base_params, scratch_params)

    logger.info(f"[Scratch] params={scratch_params:,} | lr={scratch_lr:.6f}")

    scratch_optimizer = torch.optim.AdamW(
        scratch_model.parameters(), lr=scratch_lr, weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    start = time.time()

    # Train with patience using a simple loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    epoch_counter = 0
    global_step = 0
    n_train = len(train_tokens)

    while epoch_counter < config.max_epochs:
        perm = torch.randperm(n_train)
        shuffled = train_tokens[perm]
        scratch_model.train()
        scratch_optimizer.zero_grad()
        accum_count = 0

        pbar = tqdm(
            range(0, n_train, config.batch_size),
            desc=f"[Scratch] Epoch {epoch_counter+1}",
            leave=False,
        )

        for i in pbar:
            batch = shuffled[i : i + config.batch_size].to(config.device)
            if batch.shape[0] == 0:
                continue
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = scratch_model(inputs)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            loss = loss / config.grad_accum_steps
            loss.backward()

            accum_count += 1
            if accum_count >= config.grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(scratch_model.parameters(), 1.0)
                scratch_optimizer.step()
                scratch_optimizer.zero_grad()
                global_step += 1
                accum_count = 0

                if global_step % 50 == 0:
                    writer.add_scalar(
                        "Scratch/Loss/Train",
                        loss.item() * config.grad_accum_steps,
                        global_step,
                    )

        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(scratch_model.parameters(), 1.0)
            scratch_optimizer.step()
            scratch_optimizer.zero_grad()
            global_step += 1

        val_loss, ppl = evaluate_lm(scratch_model, val_tokens, config.batch_size, config.device)
        writer.add_scalar("Scratch/Loss/Val", val_loss, global_step)
        writer.add_scalar("Scratch/Perplexity/Val", ppl, global_step)

        epoch_counter += 1
        logger.info(
            f"[Scratch] Epoch {epoch_counter} | val_loss={val_loss:.4f} ppl={ppl:.2f} | "
            f"lr={scratch_lr:.6f} | patience={epochs_without_improvement}/{config.patience}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch_counter >= config.min_epochs_per_stage
            and epochs_without_improvement >= config.patience
        ):
            logger.info(f"[Scratch] Plateau after {epoch_counter} epochs")
            break

    final_loss, final_ppl = evaluate_lm(
        scratch_model, val_tokens, config.batch_size, config.device
    )
    scratch_elapsed = time.time() - start

    scratch_results = {
        "protocol": "scratch",
        "final_val_loss": final_loss,
        "final_ppl": final_ppl,
        "expansion_events": [],
        "elapsed_time": scratch_elapsed,
        "total_epochs": epoch_counter,
        "final_d_ff": [config.d_ff_max] * config.n_layers,
    }
    all_results["scratch"] = scratch_results

    logger.info(
        f"[Scratch] Final: val_loss={final_loss:.4f} ppl={final_ppl:.2f} | "
        f"epochs={epoch_counter}"
    )

    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")

    for name, results in all_results.items():
        d_ff_str = ""
        if "final_d_ff" in results:
            d_ff_str = f"  d_ff={results['final_d_ff']}"
        logger.info(
            f"  {name:>12s}: ppl={results['final_ppl']:.2f}  "
            f"val_loss={results['final_val_loss']:.4f}  "
            f"time={results['elapsed_time']:.1f}s  "
            f"epochs={results['total_epochs']}{d_ff_str}"
        )
        for ev in results.get("expansion_events", []):
            logger.info(
                f"{'':>14s}  expand d_ff {ev.width_before}->{ev.width_after} "
                f"layer={ev.layer_idx}: loss_delta={ev.shock:+.4f} "
                f"ppl: {ev.acc_before:.2f}->{ev.acc_after_first_epoch:.2f}"
            )

    writer.close()
    logger.info(f"\nTensorBoard: tensorboard --logdir {os.path.dirname(log_dir)}")

    return all_results
