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
from ops import expand_model_continuous, expand_model_net2net

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
    dropout: float = 0.2
    width_schedule: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048])
    epochs_per_stage: list[int] = field(default_factory=lambda: [15, 15, 15, 15, 30])
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
        """Return the epoch numbers at which expansions occur.

        E.g., with epochs_per_stage=[20, 20, 20, 40], expansions happen at epochs 20, 40, 60.
        """
        epochs = []
        cumulative = 0
        for stage_epochs in self.epochs_per_stage[:-1]:
            cumulative += stage_epochs
            epochs.append(cumulative)
        return epochs

    @property
    def expansion_targets(self) -> list[int]:
        """Return the target widths for each expansion.

        E.g., with width_schedule=[128, 256, 512, 1024], targets are [256, 512, 1024].
        """
        return self.width_schedule[1:]

    @property
    def stage_boundaries(self) -> list[int]:
        """Return the starting epoch of each stage (including 0 for the first).

        E.g., with epochs_per_stage=[30, 30, 30, 60], returns [0, 30, 60, 90].
        """
        boundaries = [0]
        cumulative = 0
        for stage_epochs in self.epochs_per_stage[:-1]:
            cumulative += stage_epochs
            boundaries.append(cumulative)
        return boundaries

    def stage_length_at(self, epoch: int) -> int:
        """Return the number of epochs in the stage that contains the given epoch.

        Used to set T_max for cosine annealing within each stage.
        """
        cumulative = 0
        for stage_epochs in self.epochs_per_stage:
            cumulative += stage_epochs
            if epoch < cumulative:
                return stage_epochs
        return self.epochs_per_stage[-1]


@dataclass
class ExpansionEvent:
    """Records what happened at a single expansion.

    Shock is measured as the difference between the pre-expansion val loss
    and the val loss after the first post-expansion training epoch.
    """

    expansion_index: int
    step: int
    epoch: int
    width_before: int
    width_after: int
    loss_before: float
    loss_after_first_epoch: float
    acc_before: float
    acc_after_first_epoch: float

    @property
    def shock(self) -> float:
        return self.loss_after_first_epoch - self.loss_before

    @property
    def acc_delta(self) -> float:
        return self.acc_after_first_epoch - self.acc_before


def evaluate(
    model: MLP, dataloader: DataLoader, criterion: nn.Module, device: str
) -> tuple[float, float]:
    """Compute validation loss and accuracy.

    Returns:
        (loss, accuracy) tuple.
    """
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


def train_protocol(
    protocol: Literal["scratch", "net2net", "continuous"],
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    writer: SummaryWriter,
    tag_prefix: str,
) -> dict:
    """Run a complete training protocol with multiple expansion stages.

    Args:
        protocol: One of 'scratch', 'net2net', 'continuous'.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        writer: TensorBoard writer.
        tag_prefix: Prefix for TensorBoard tags.

    Returns:
        Dict with final metrics and list of expansion events.
    """
    device = config.device
    criterion = nn.CrossEntropyLoss()
    total_epochs = config.total_epochs

    # Initialize model
    if protocol == "scratch":
        model = MLP(
            input_dim=config.input_dim,
            hidden1=config.final_width,
            hidden2=config.final_width,
            dropout=config.dropout,
        ).to(device)
        expansion_schedule = {}  # No expansions
    else:
        model = MLP(
            input_dim=config.input_dim,
            hidden1=config.initial_width,
            hidden2=config.initial_width,
            dropout=config.dropout,
        ).to(device)
        # Map epoch -> (target_width, expansion_index)
        expansion_schedule = {
            epoch: (width, i)
            for i, (epoch, width) in enumerate(
                zip(config.expansion_epochs, config.expansion_targets)
            )
        }

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Cosine annealing LR scheduler — reset at each expansion.
    # For scratch: anneal over all epochs. For expansion protocols: anneal per stage.
    if protocol == "scratch":
        scheduler_T = total_epochs
    else:
        scheduler_T = config.epochs_per_stage[0]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_T, eta_min=1e-6
    )

    global_step = 0
    expansion_events: list[ExpansionEvent] = []

    # State for tracking shock across the post-expansion epoch
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
        # --- Check for expansion ---
        if epoch in expansion_schedule:
            target_width, exp_idx = expansion_schedule[epoch]
            width_before = model.fc1.out_features

            logger.info(
                f"[{tag_prefix}] === EXPANSION {exp_idx+1} at epoch {epoch} "
                f"(step {global_step}): {width_before} -> {target_width} ==="
            )

            # Evaluate before expansion
            val_loss_before, val_acc_before = evaluate(model, val_loader, criterion, device)
            logger.info(
                f"[{tag_prefix}] Pre-expansion: val_loss={val_loss_before:.4f}, "
                f"val_acc={val_acc_before:.4f}"
            )

            # Expand
            if protocol == "net2net":
                model = expand_model_net2net(
                    model, target_width, target_width, optimizer
                )
            elif protocol == "continuous":
                model = expand_model_continuous(
                    model, target_width, target_width, optimizer
                )

            # Evaluate immediately after expansion (for TensorBoard)
            val_loss_immediate, val_acc_immediate = evaluate(model, val_loader, criterion, device)
            logger.info(
                f"[{tag_prefix}] Immediate post-expansion: val_loss={val_loss_immediate:.4f}, "
                f"val_acc={val_acc_immediate:.4f}"
            )

            writer.add_scalar(f"{tag_prefix}/Val_Loss_PreExpand", val_loss_before, global_step)
            writer.add_scalar(f"{tag_prefix}/Val_Loss_PostExpand_Immediate", val_loss_immediate, global_step)

            # Reset LR and scheduler for the new stage
            for pg in optimizer.param_groups:
                pg["lr"] = config.lr
            stage_epochs = config.stage_length_at(epoch)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage_epochs, eta_min=1e-6
            )
            logger.info(
                f"[{tag_prefix}] LR scheduler reset: T_max={stage_epochs}, lr={config.lr}"
            )

            # Remember that we need to measure shock after the next epoch
            pending_shock = {
                "expansion_index": exp_idx,
                "step": global_step,
                "epoch": epoch,
                "width_before": width_before,
                "width_after": target_width,
                "loss_before": val_loss_before,
                "acc_before": val_acc_before,
            }

        # --- Training epoch ---
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

        # --- End of epoch validation ---
        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        writer.add_scalar(f"{tag_prefix}/Loss/Val", val_loss, global_step)
        writer.add_scalar(f"{tag_prefix}/Accuracy/Train", train_acc, global_step)
        writer.add_scalar(f"{tag_prefix}/Accuracy/Val", val_acc, global_step)

        # Step LR scheduler and log
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar(f"{tag_prefix}/LR", current_lr, global_step)

        logger.info(
            f"[{tag_prefix}] Epoch {epoch+1}/{total_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

        # --- Measure shock after first post-expansion epoch ---
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

            writer.add_scalar(
                f"{tag_prefix}/Expansion_Shock",
                event.shock,
                pending_shock["step"],
            )

            logger.info(
                f"[{tag_prefix}] Expansion {event.expansion_index+1} shock (after 1 epoch): "
                f"{event.width_before}->{event.width_after} "
                f"loss_delta={event.shock:+.4f}, acc_delta={event.acc_delta:+.4f}"
            )
            pending_shock = None

    # Final evaluation
    final_val_loss, final_val_acc = evaluate(model, val_loader, criterion, device)

    results = {
        "protocol": protocol,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "expansion_events": expansion_events,
    }

    return results
