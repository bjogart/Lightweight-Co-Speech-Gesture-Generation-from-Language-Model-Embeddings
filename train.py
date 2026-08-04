import util
import json
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import (
  SequentialLR,
  LRScheduler,
  LinearLR,
  CosineAnnealingLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Optional
import os
import torch
import torch.nn as nn


def to_device(v, device: str):
  if type(v) is torch.Tensor:
    return v.to(device)
  elif type(v) is dict:
    return {k: to_device(v, device) for k, v in v.items()}
  else:
    raise Exception(f"cannot move value of type {type(v)} to {device}: {v}")


def loss_batch(
  batch,
  model: nn.Module,
  loss_func: Callable[
    [dict[str, torch.Tensor], dict[str, torch.Tensor]],
    torch.Tensor,
  ],
  optim_and_scheduler: Optional[tuple[AdamW, LRScheduler]],
) -> int | float:
  batch = to_device(batch, util.DEVICE)
  targets = batch.pop("targets")
  outputs = model(**batch)
  loss = loss_func(outputs, targets)
  if optim_and_scheduler is not None:
    optim, scheduler = optim_and_scheduler
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    optim.zero_grad()
    scheduler.step()
  return loss.item()


def train_epoch(
  progress_epoch: tqdm,
  model: nn.Module,
  loss_func: Callable[
    [dict[str, torch.Tensor], dict[str, torch.Tensor]],
    torch.Tensor,
  ],
  optim_and_scheduler: tuple[AdamW, LRScheduler],
  train_data: DataLoader,
  val_data: DataLoader,
) -> tuple[
  list[tuple[int, int | float | torch.Tensor, int | float | torch.Tensor]],
  list[tuple[int, int | float | torch.Tensor, int | float | torch.Tensor]],
]:
  model.train()
  segment_len = len(train_data) // 10
  train_losses = []
  val_losses = []
  progress_train = tqdm(
    train_data, position=1, desc="train", leave=False, smoothing=0
  )
  val_loss = val_epoch(
    progress_epoch, progress_train, model, loss_func, val_data
  )
  _, scheduler = optim_and_scheduler
  lr = scheduler.get_last_lr()[0]
  val_losses.append((0, val_loss, lr))
  for idx, batch in enumerate(progress_train):
    progress_epoch.refresh()
    train_loss = loss_batch(batch, model, loss_func, optim_and_scheduler)
    _, scheduler = optim_and_scheduler
    lr = scheduler.get_last_lr()[0]
    train_losses.append((idx + 1, train_loss, lr))
    if (idx % segment_len) == 0:
      val_loss = val_epoch(
        progress_epoch, progress_train, model, loss_func, val_data
      )
      val_losses.append((idx + 1, val_loss, lr))
  return train_losses, val_losses


def val_epoch(
  progress_epoch: Optional[tqdm],
  progress_train: Optional[tqdm],
  model: nn.Module,
  loss_func: Callable[
    [dict[str, torch.Tensor], dict[str, torch.Tensor]],
    torch.Tensor,
  ],
  val_data: DataLoader,
) -> float | torch.Tensor:
  model.eval()
  val_loss = 0.0
  val_len = 0
  with torch.no_grad():
    for batch in tqdm(
      val_data, position=2, desc="validate", leave=False, smoothing=0
    ):
      if progress_epoch:
        progress_epoch.refresh()
      if progress_train:
        progress_train.refresh()
      loss = loss_batch(batch, model, loss_func, None)
      val_loss += loss
      val_len += 1
    return val_loss / val_len


def fit_epoch(
  progress_epoch: tqdm,
  model: nn.Module,
  loss_func: Callable[
    [dict[str, torch.Tensor], dict[str, torch.Tensor]],
    torch.Tensor,
  ],
  optim_and_scheduler: tuple[AdamW, LRScheduler],
  train_data: DataLoader,
  val_data: DataLoader,
) -> tuple[
  list[tuple[int, int | float | torch.Tensor, int | float | torch.Tensor]],
  list[tuple[int, int | float | torch.Tensor, int | float | torch.Tensor]],
]:
  train_losses, val_losses = train_epoch(
    progress_epoch, model, loss_func, optim_and_scheduler, train_data, val_data
  )
  val_loss = val_epoch(progress_epoch, None, model, loss_func, val_data)
  _, scheduler = optim_and_scheduler
  lr = scheduler.get_last_lr()[0]
  val_losses.append((len(train_data), val_loss, lr))
  return train_losses, val_losses


def report_epoch(
  model: nn.Module,
  model_dir: str,
  epoch_idx: int,
  train_losses: list[
    tuple[int, int | float | torch.Tensor, int | float | torch.Tensor]
  ],
  val_losses: list[
    tuple[int, int | float | torch.Tensor, int | float | torch.Tensor]
  ],
  optim_and_scheduler: tuple[AdamW, LRScheduler],
):
  _, scheduler = optim_and_scheduler
  lr = scheduler.get_last_lr()[0]
  epoch = epoch_idx + 1
  avg_train_loss = (
    sum(loss for _, loss, _ in train_losses) / len(train_losses)
    if len(train_losses) != 0
    else float("inf")
  )
  avg_val_loss = (
    sum(loss for _, loss, _ in val_losses) / len(val_losses)
    if len(val_losses) != 0
    else float("inf")
  )
  tqdm.write(
    f"epoch={epoch:02} train_loss={avg_train_loss:07.4f} val_loss={avg_val_loss:07.4f} lr={lr:012.10f}"
  )

  with open(os.path.join(model_dir, f"epoch{epoch:02}.json"), "w") as f:
    json.dump(
      {
        "epoch": epoch,
        "train_losses": [
          {"batch": batch, "loss": loss, "lr": lr}
          for (batch, loss, lr) in train_losses
        ],
        "val_losses": [
          {"batch": batch, "loss": loss, "lr": lr}
          for (batch, loss, lr) in val_losses
        ],
      },
      f,
    )
  torch.save(
    model.state_dict(), os.path.join(model_dir, f"epoch{epoch:02}.pth")
  )


def fit(
  model_dir: str,
  n_epochs: int,
  model: nn.Module,
  loss_func: Callable[
    [dict[str, torch.Tensor], dict[str, torch.Tensor]],
    torch.Tensor,
  ],
  optim_and_scheduler: tuple[AdamW, LRScheduler],
  train_data: DataLoader,
  val_data: DataLoader,
):
  os.makedirs(model_dir, exist_ok=True)
  with open(os.path.join(model_dir, ".gitignore"), "w") as f:
    f.write("/*")
  progress_epoch = tqdm(range(n_epochs), desc="epoch", position=0)

  val_loss = val_epoch(progress_epoch, None, model, loss_func, val_data)
  _, scheduler = optim_and_scheduler
  lr = scheduler.get_last_lr()[0]
  report_epoch(
    model,
    model_dir,
    -1,
    [],
    [(0, val_loss, lr)],
    optim_and_scheduler,
  )

  for epoch_idx in progress_epoch:
    train_loss, steps = fit_epoch(
      progress_epoch,
      model,
      loss_func,
      optim_and_scheduler,
      train_data,
      val_data,
    )
    report_epoch(
      model,
      model_dir,
      epoch_idx,
      train_loss,
      steps,
      optim_and_scheduler,
    )


def multi_headed_loss(
  loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], torch.Tensor]:
  def inner(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    total_loss = 0.0
    for part in outputs.keys():
      total_loss += loss_func(outputs[part], targets[part])
    return total_loss / len(outputs)  # ty: ignore

  return inner


def cross_entropy_sequence_loss():
  loss_fn = nn.CrossEntropyLoss()

  def inner(outputs, targets):
    return loss_fn(outputs.transpose(1, 2), targets)

  return inner


def optimizer_with_schedule(
  warmup_steps: int,
  n_epochs: int,
  lr: int | float,
  model: nn.Module,
  train_data: DataLoader,
) -> tuple[AdamW, LRScheduler]:
  optim = AdamW(model.parameters(), lr=lr)
  total_steps = len(train_data) * n_epochs
  if warmup_steps == 0:
    scheduler = CosineAnnealingLR(optim, T_max=total_steps)
  else:
    warmup = LinearLR(
      optim, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay = CosineAnnealingLR(optim, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(
      optim, schedulers=[warmup, decay], milestones=[warmup_steps]
    )
  return optim, scheduler


def train_model(
  model: nn.Module,
  loss_func: Callable[
    [torch.Tensor, torch.Tensor],
    torch.Tensor,
  ],
  train_data: DataLoader,
  val_data: DataLoader,
  lr: int | float,
  n_epochs: int,
  warmup_steps: int,
  model_dir_stem: str,
):
  model.to(util.DEVICE)
  optim_and_scheduler = optimizer_with_schedule(
    warmup_steps, n_epochs, lr, model, train_data
  )

  fit(
    f"{model_dir_stem}--{util.iso_timestamp()}",
    n_epochs,
    model,
    multi_headed_loss(loss_func),
    optim_and_scheduler,
    train_data,
    val_data,
  )
