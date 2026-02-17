"""Training loop with TensorBoard logging and model expansion support."""

import logging
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import MLP
from ops import (
    expand_layer_continuous,
    expand_layer_net2net,
    expand_model_continuous,
    expand_model_net2net,
    layer_similarity_scores,
    params_after_doubling_layer,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration.

    The expansion schedule is defined by:
    - width_schedule: list of hidden widths, e.g., [128, 256, 512, 1024]
      The first entry is the initial width; each subsequent entry triggers an expansion.
    - epochs_per_stage: list of epoch counts for each stage, e.g., [20, 20, 20, 40]
      Must have the same length as width_schedule.
      The scratch baseline trains at the final width for sum(epochs_per_stage) epochs.
    """

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    seed: int = 42
    input_dim: int = 3072
    num_hidden_layers: int = 4
    dropout: float = 0.2
    width_schedule: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048])
    epochs_per_stage: list[int] = field(default_factory=lambda: [15, 15, 15, 15, 30])
    epochs_per_targeted_stage: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def total_epochs(self) -> int:
        return sum(self.epochs_per_stage)

    @property
    def initial_width(self) -> int:
        return self.width_schedule[0]

    @property
    def final_width(self) -> int:
        return self.width_schedule[-1]

    @property
    def expansion_epochs(self) -> list[int]:
        epochs = []
        cumulative = 0
        for stage_epochs in self.epochs_per_stage[:-1]:
            cumulative += stage_epochs
            epochs.append(cumulative)
        return epochs

    @property
    def expansion_targets(self) -> list[int]:
        return self.width_schedule[1:]

    @property
    def stage_boundaries(self) -> list[int]:
        boundaries = [0]
        cumulative = 0
        for stage_epochs in self.epochs_per_stage[:-1]:
            cumulative += stage_epochs
            boundaries.append(cumulative)
        return boundaries

    def stage_length_at(self, epoch: int) -> int:
        cumulative = 0
        for stage_epochs in self.epochs_per_stage:
            cumulative += stage_epochs
            if epoch < cumulative:
                return stage_epochs
        return self.epochs_per_stage[-1]

    @property
    def param_budget(self) -> int:
        """Parameter count of the fully expanded model (all layers at final_width)."""
        ref = MLP(
            input_dim=self.input_dim,
            hidden_widths=[self.final_width] * self.num_hidden_layers,
        )
        return sum(p.numel() for p in ref.parameters())


@dataclass
class ExpansionEvent:
    """Records what happened at a single expansion."""

    expansion_index: int
    step: int
    epoch: int
    width_before: int
    width_after: int
    loss_before: float
    loss_after_first_epoch: float
    acc_before: float
    acc_after_first_epoch: float
    layer_idx: int | None = None  # For targeted expansion

    @property
    def shock(self) -> float:
        return self.loss_after_first_epoch - self.loss_before

    @property
    def acc_delta(self) -> float:
        return self.acc_after_first_epoch - self.acc_before


@dataclass
class TargetedExpansionStep:
    """Records a single step in a targeted expansion plan for replay."""

    layer_idx: int
    old_width: int
    new_width: int


def evaluate(
    model: MLP, dataloader: DataLoader, criterion: nn.Module, device: str
) -> tuple[float, float]:
    """Compute validation loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def _train_epochs(
    model: MLP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    writer: SummaryWriter,
    tag_prefix: str,
    start_epoch: int,
    num_epochs: int,
    total_epochs: int,
    global_step: int,
) -> tuple[int, float, float]:
    """Train for a fixed number of epochs. Returns (global_step, val_loss, val_acc)."""
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(
            train_loader,
            desc=f"[{tag_prefix}] Epoch {epoch+1}/{total_epochs}",
            leave=False,
        )

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(labels).sum().item()
            batch_total = images.size(0)

            epoch_loss += batch_loss * batch_total
            epoch_correct += batch_correct
            epoch_total += batch_total

            writer.add_scalar(f"{tag_prefix}/Loss/Train", batch_loss, global_step)
            global_step += 1

            pbar.set_postfix(
                loss=f"{batch_loss:.4f}",
                acc=f"{batch_correct/batch_total:.4f}",
            )

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        writer.add_scalar(f"{tag_prefix}/Loss/Val", val_loss, global_step)
        writer.add_scalar(f"{tag_prefix}/Accuracy/Train", train_acc, global_step)
        writer.add_scalar(f"{tag_prefix}/Accuracy/Val", val_acc, global_step)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar(f"{tag_prefix}/LR", current_lr, global_step)

        logger.info(
            f"[{tag_prefix}] Epoch {epoch+1}/{total_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

    return global_step, val_loss, val_acc


def train_protocol(
    protocol: Literal["scratch", "net2net", "continuous"],
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    writer: SummaryWriter,
    tag_prefix: str,
) -> dict:
    """Run a complete training protocol with multiple uniform expansion stages."""
    device = config.device
    criterion = nn.CrossEntropyLoss()
    total_epochs = config.total_epochs

    # Initialize model
    if protocol == "scratch":
        model = MLP(
            input_dim=config.input_dim,
            hidden_widths=[config.final_width] * config.num_hidden_layers,
            dropout=config.dropout,
        ).to(device)
        expansion_schedule = {}
    else:
        model = MLP(
            input_dim=config.input_dim,
            hidden_widths=[config.initial_width] * config.num_hidden_layers,
            dropout=config.dropout,
        ).to(device)
        expansion_schedule = {
            epoch: (width, i)
            for i, (epoch, width) in enumerate(
                zip(config.expansion_epochs, config.expansion_targets)
            )
        }

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    if protocol == "scratch":
        scheduler_T = total_epochs
    else:
        scheduler_T = config.epochs_per_stage[0]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_T, eta_min=1e-6
    )

    global_step = 0
    expansion_events: list[ExpansionEvent] = []
    pending_shock: dict | None = None

    logger.info(
        f"[{tag_prefix}] Starting training: {protocol} | "
        f"Model widths: {model.hidden_widths} | "
        f"Total epochs: {total_epochs}"
    )
    if expansion_schedule:
        schedule_str = " -> ".join(str(w) for w in config.width_schedule)
        logger.info(f"[{tag_prefix}] Expansion schedule: {schedule_str}")
        logger.info(f"[{tag_prefix}] Expansion epochs: {config.expansion_epochs}")

    for epoch in range(total_epochs):
        if epoch in expansion_schedule:
            target_width, exp_idx = expansion_schedule[epoch]
            width_before = model.hidden_widths[0]

            logger.info(
                f"[{tag_prefix}] === EXPANSION {exp_idx+1} at epoch {epoch} "
                f"(step {global_step}): {width_before} -> {target_width} ==="
            )

            val_loss_before, val_acc_before = evaluate(model, val_loader, criterion, device)
            logger.info(
                f"[{tag_prefix}] Pre-expansion: val_loss={val_loss_before:.4f}, "
                f"val_acc={val_acc_before:.4f}"
            )

            if protocol == "net2net":
                model = expand_model_net2net(model, target_width, optimizer)
            elif protocol == "continuous":
                model = expand_model_continuous(model, target_width, optimizer)

            val_loss_immediate, val_acc_immediate = evaluate(model, val_loader, criterion, device)
            logger.info(
                f"[{tag_prefix}] Immediate post-expansion: val_loss={val_loss_immediate:.4f}, "
                f"val_acc={val_acc_immediate:.4f}"
            )

            writer.add_scalar(f"{tag_prefix}/Val_Loss_PreExpand", val_loss_before, global_step)
            writer.add_scalar(f"{tag_prefix}/Val_Loss_PostExpand_Immediate", val_loss_immediate, global_step)

            for pg in optimizer.param_groups:
                pg["lr"] = config.lr
            stage_epochs = config.stage_length_at(epoch)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage_epochs, eta_min=1e-6
            )
            logger.info(f"[{tag_prefix}] LR scheduler reset: T_max={stage_epochs}, lr={config.lr}")

            pending_shock = {
                "expansion_index": exp_idx,
                "step": global_step,
                "epoch": epoch,
                "width_before": width_before,
                "width_after": target_width,
                "loss_before": val_loss_before,
                "acc_before": val_acc_before,
            }

        # Training epoch
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(
            train_loader,
            desc=f"[{tag_prefix}] Epoch {epoch+1}/{total_epochs}",
            leave=False,
        )

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(labels).sum().item()
            batch_total = images.size(0)

            epoch_loss += batch_loss * batch_total
            epoch_correct += batch_correct
            epoch_total += batch_total

            writer.add_scalar(f"{tag_prefix}/Loss/Train", batch_loss, global_step)
            global_step += 1

            pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_correct/batch_total:.4f}")

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        writer.add_scalar(f"{tag_prefix}/Loss/Val", val_loss, global_step)
        writer.add_scalar(f"{tag_prefix}/Accuracy/Train", train_acc, global_step)
        writer.add_scalar(f"{tag_prefix}/Accuracy/Val", val_acc, global_step)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar(f"{tag_prefix}/LR", current_lr, global_step)

        logger.info(
            f"[{tag_prefix}] Epoch {epoch+1}/{total_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if pending_shock is not None:
            event = ExpansionEvent(
                expansion_index=pending_shock["expansion_index"],
                step=pending_shock["step"],
                epoch=pending_shock["epoch"],
                width_before=pending_shock["width_before"],
                width_after=pending_shock["width_after"],
                loss_before=pending_shock["loss_before"],
                loss_after_first_epoch=val_loss,
                acc_before=pending_shock["acc_before"],
                acc_after_first_epoch=val_acc,
            )
            expansion_events.append(event)

            writer.add_scalar(f"{tag_prefix}/Expansion_Shock", event.shock, pending_shock["step"])

            logger.info(
                f"[{tag_prefix}] Expansion {event.expansion_index+1} shock (after 1 epoch): "
                f"{event.width_before}->{event.width_after} "
                f"loss_delta={event.shock:+.4f}, acc_delta={event.acc_delta:+.4f}"
            )
            pending_shock = None

    final_val_loss, final_val_acc = evaluate(model, val_loader, criterion, device)

    return {
        "protocol": protocol,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "expansion_events": expansion_events,
    }


def train_targeted(
    method: Literal["continuous", "net2net"],
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    writer: SummaryWriter,
    tag_prefix: str,
    expansion_plan: list[TargetedExpansionStep] | None = None,
) -> dict:
    """Run a targeted expansion protocol.

    If expansion_plan is None (CSR mode), dynamically chooses which layer to
    expand based on similarity scores, recording the plan.

    If expansion_plan is provided (Net2Net replay mode), follows the plan exactly.

    Args:
        method: 'continuous' for CSR, 'net2net' for Net2Net replay.
        train_loader: Training data.
        val_loader: Validation data.
        config: Training configuration.
        writer: TensorBoard writer.
        tag_prefix: Prefix for logging tags.
        expansion_plan: If provided, replay this plan. If None, build plan dynamically.

    Returns:
        Dict with results including the expansion plan used.
    """
    device = config.device
    criterion = nn.CrossEntropyLoss()
    param_budget = config.param_budget
    epochs_per_stage = config.epochs_per_targeted_stage

    # Initialize small model
    model = MLP(
        input_dim=config.input_dim,
        hidden_widths=[config.initial_width] * config.num_hidden_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    dynamic = expansion_plan is None
    if dynamic:
        expansion_plan = []

    current_params = sum(p.numel() for p in model.parameters())
    global_step = 0
    epoch_counter = 0
    expansion_events: list[ExpansionEvent] = []
    exp_idx = 0

    # Determine total epochs: initial stage + (N expansions * epochs_per_stage) + final stage
    # We don't know N ahead of time for dynamic mode, so we compute total at the end.
    # For replay mode, we know the plan length.
    if not dynamic:
        num_expansions = len(expansion_plan)
    else:
        num_expansions = None  # unknown

    logger.info(
        f"[{tag_prefix}] Starting targeted training: {method} | "
        f"Model widths: {model.hidden_widths} | "
        f"Param budget: {param_budget:,} | Current params: {current_params:,}"
    )

    # --- Initial training stage ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_per_stage, eta_min=1e-6
    )
    total_epochs_estimate = epochs_per_stage * 20  # generous upper bound for display
    global_step, val_loss, val_acc = _train_epochs(
        model, optimizer, scheduler, train_loader, val_loader, criterion, device,
        writer, tag_prefix, epoch_counter, epochs_per_stage, total_epochs_estimate, global_step,
    )
    epoch_counter += epochs_per_stage

    # --- Expansion loop ---
    while True:
        if dynamic:
            # Score layers and pick the least similar one that fits budget
            scores = layer_similarity_scores(model)
            widths = list(model.hidden_widths)

            # Sort layers by similarity (ascending = least similar first)
            layer_order = sorted(range(len(scores)), key=lambda i: scores[i])

            chosen_layer = None
            for li in layer_order:
                new_params = params_after_doubling_layer(model, li)
                if new_params <= param_budget:
                    chosen_layer = li
                    break

            if chosen_layer is None:
                logger.info(
                    f"[{tag_prefix}] No layer can be doubled within budget. "
                    f"Current params: {current_params:,}, budget: {param_budget:,}"
                )
                break

            old_width = widths[chosen_layer]
            new_width = old_width * 2
            step = TargetedExpansionStep(
                layer_idx=chosen_layer, old_width=old_width, new_width=new_width
            )
            expansion_plan.append(step)

            logger.info(
                f"[{tag_prefix}] Layer similarity scores: "
                + ", ".join(f"L{i}={s:.4f}" for i, s in enumerate(scores))
            )
            logger.info(
                f"[{tag_prefix}] Selected layer {chosen_layer} "
                f"(sim={scores[chosen_layer]:.4f}, width {old_width}->{new_width})"
            )
        else:
            if exp_idx >= len(expansion_plan):
                break
            step = expansion_plan[exp_idx]

        # Evaluate before expansion
        val_loss_before, val_acc_before = evaluate(model, val_loader, criterion, device)
        width_before = step.old_width

        logger.info(
            f"[{tag_prefix}] === EXPANSION {exp_idx+1} at epoch {epoch_counter} "
            f"(step {global_step}): layer {step.layer_idx} "
            f"{step.old_width}->{step.new_width} ==="
        )
        logger.info(
            f"[{tag_prefix}] Pre-expansion: val_loss={val_loss_before:.4f}, "
            f"val_acc={val_acc_before:.4f} | widths={model.hidden_widths}"
        )

        # Expand
        if method == "continuous":
            model = expand_layer_continuous(model, step.layer_idx, optimizer)
        else:
            model = expand_layer_net2net(model, step.layer_idx, optimizer)

        current_params = sum(p.numel() for p in model.parameters())

        val_loss_immediate, val_acc_immediate = evaluate(model, val_loader, criterion, device)
        logger.info(
            f"[{tag_prefix}] Post-expansion: val_loss={val_loss_immediate:.4f}, "
            f"val_acc={val_acc_immediate:.4f} | widths={model.hidden_widths} | "
            f"params={current_params:,}"
        )

        writer.add_scalar(f"{tag_prefix}/Val_Loss_PreExpand", val_loss_before, global_step)
        writer.add_scalar(f"{tag_prefix}/Val_Loss_PostExpand_Immediate", val_loss_immediate, global_step)
        writer.add_scalar(f"{tag_prefix}/Params", current_params, global_step)

        # Reset LR and train next stage
        for pg in optimizer.param_groups:
            pg["lr"] = config.lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_stage, eta_min=1e-6
        )

        global_step, val_loss, val_acc = _train_epochs(
            model, optimizer, scheduler, train_loader, val_loader, criterion, device,
            writer, tag_prefix, epoch_counter, epochs_per_stage, total_epochs_estimate,
            global_step,
        )

        # Record shock (val_loss after first post-expansion epoch approximated by end-of-stage)
        # For consistency we use the val_loss after the full stage
        event = ExpansionEvent(
            expansion_index=exp_idx,
            step=global_step,
            epoch=epoch_counter,
            width_before=width_before,
            width_after=step.new_width,
            loss_before=val_loss_before,
            loss_after_first_epoch=val_loss,
            acc_before=val_acc_before,
            acc_after_first_epoch=val_acc,
            layer_idx=step.layer_idx,
        )
        expansion_events.append(event)

        logger.info(
            f"[{tag_prefix}] Expansion {exp_idx+1} result: "
            f"layer {step.layer_idx} {step.old_width}->{step.new_width} "
            f"loss_delta={event.shock:+.4f}, acc_delta={event.acc_delta:+.4f}"
        )

        epoch_counter += epochs_per_stage
        exp_idx += 1

    # --- Final training stage (double length, like the uniform protocol) ---
    final_stage_epochs = epochs_per_stage * 2
    for pg in optimizer.param_groups:
        pg["lr"] = config.lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=final_stage_epochs, eta_min=1e-6
    )
    global_step, val_loss, val_acc = _train_epochs(
        model, optimizer, scheduler, train_loader, val_loader, criterion, device,
        writer, tag_prefix, epoch_counter, final_stage_epochs, total_epochs_estimate,
        global_step,
    )
    epoch_counter += final_stage_epochs

    final_val_loss, final_val_acc = evaluate(model, val_loader, criterion, device)

    logger.info(
        f"[{tag_prefix}] Final model widths: {model.hidden_widths} | "
        f"params: {sum(p.numel() for p in model.parameters()):,} | "
        f"total epochs: {epoch_counter}"
    )

    return {
        "protocol": f"targeted_{method}",
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "expansion_events": expansion_events,
        "expansion_plan": expansion_plan,
        "total_epochs": epoch_counter,
        "final_widths": model.hidden_widths,
    }
