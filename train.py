"""Training loop with TensorBoard logging and model expansion support."""

import logging
import time
from dataclasses import dataclass
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
    """Training configuration."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs_pre: int = 10
    epochs_total: int = 30
    seed: int = 42
    hidden_small: int = 128
    hidden_large: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExpansionEvent:
    """Records what happened at expansion time."""

    step: int
    epoch: int
    loss_before: float
    loss_after: float
    acc_before: float
    acc_after: float

    @property
    def shock(self) -> float:
        return self.loss_after - self.loss_before


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
    """Run a complete training protocol.

    Args:
        protocol: One of 'scratch', 'net2net', 'continuous'.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        writer: TensorBoard writer.
        tag_prefix: Prefix for TensorBoard tags (e.g., 'Scratch', 'Net2Net', 'CSR').

    Returns:
        Dict with final metrics and expansion event info.
    """
    device = config.device
    criterion = nn.CrossEntropyLoss()

    # Initialize model
    if protocol == "scratch":
        model = MLP(hidden1=config.hidden_large, hidden2=config.hidden_large).to(device)
        total_epochs = config.epochs_total
        expand_at_epoch = None
    else:
        model = MLP(hidden1=config.hidden_small, hidden2=config.hidden_small).to(device)
        total_epochs = config.epochs_total
        expand_at_epoch = config.epochs_pre

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    global_step = 0
    expansion_event = None

    logger.info(
        f"[{tag_prefix}] Starting training: {protocol} | "
        f"Model widths: {model.hidden_widths} | "
        f"Total epochs: {total_epochs}"
    )

    for epoch in range(total_epochs):
        # --- Check for expansion ---
        if expand_at_epoch is not None and epoch == expand_at_epoch:
            logger.info(f"[{tag_prefix}] === EXPANSION at epoch {epoch} (step {global_step}) ===")

            # Evaluate before expansion
            val_loss_before, val_acc_before = evaluate(model, val_loader, criterion, device)
            logger.info(
                f"[{tag_prefix}] Pre-expansion: val_loss={val_loss_before:.4f}, "
                f"val_acc={val_acc_before:.4f}"
            )

            # Expand
            if protocol == "net2net":
                model = expand_model_net2net(
                    model, config.hidden_large, config.hidden_large, optimizer
                )
            elif protocol == "continuous":
                model = expand_model_continuous(
                    model, config.hidden_large, config.hidden_large, optimizer
                )

            # Evaluate after expansion
            val_loss_after, val_acc_after = evaluate(model, val_loader, criterion, device)
            logger.info(
                f"[{tag_prefix}] Post-expansion: val_loss={val_loss_after:.4f}, "
                f"val_acc={val_acc_after:.4f}"
            )

            expansion_event = ExpansionEvent(
                step=global_step,
                epoch=epoch,
                loss_before=val_loss_before,
                loss_after=val_loss_after,
                acc_before=val_acc_before,
                acc_after=val_acc_after,
            )

            writer.add_scalar(f"{tag_prefix}/Expansion_Shock", expansion_event.shock, global_step)
            writer.add_scalar(f"{tag_prefix}/Val_Loss_PreExpand", val_loss_before, global_step)
            writer.add_scalar(f"{tag_prefix}/Val_Loss_PostExpand", val_loss_after, global_step)
            writer.add_scalar(f"{tag_prefix}/Val_Acc_PreExpand", val_acc_before, global_step)
            writer.add_scalar(f"{tag_prefix}/Val_Acc_PostExpand", val_acc_after, global_step)

            logger.info(
                f"[{tag_prefix}] Expansion shock: {expansion_event.shock:+.4f} "
                f"(loss delta), acc delta: {val_acc_after - val_acc_before:+.4f}"
            )

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

            # Track batch stats
            batch_loss = loss.item()
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(labels).sum().item()
            batch_total = images.size(0)

            epoch_loss += batch_loss * batch_total
            epoch_correct += batch_correct
            epoch_total += batch_total

            # Log training loss every step
            writer.add_scalar(f"{tag_prefix}/Loss/Train", batch_loss, global_step)
            global_step += 1

            # Update progress bar
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

        logger.info(
            f"[{tag_prefix}] Epoch {epoch+1}/{total_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # Final evaluation
    final_val_loss, final_val_acc = evaluate(model, val_loader, criterion, device)

    results = {
        "protocol": protocol,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "expansion_event": expansion_event,
    }

    return results
