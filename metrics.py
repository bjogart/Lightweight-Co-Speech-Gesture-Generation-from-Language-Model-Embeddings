import dataclasses
import gc
import json
import os
from collections.abc import Callable

import librosa
import numpy as np
import pandas as pd
import torch
import transformers.utils.logging
from scipy import linalg
from tqdm import tqdm

import PantoMatrix.emage_utils.rotation_conversions as rc
import pipeline as pl
import prepare
from PantoMatrix.emage_utils.motion_rep_transfer import get_motion_rep_numpy
from PantoMatrix.models.emage_audio.modeling_emage_audio import EmageVQModel
from scripts import util
from scripts.metrics_genea import Bc, Fgd, L1Div, Srgr


def update_fgd(
  fgd: Fgd,
  poses_gt: np.ndarray,
  poses_sampled_pred: list[np.ndarray],
  device: str = util.DEVICE,
):
  """Update FGD evaluator with one ground truth sequence and one or more predictions.

  Args:
    fgd: FGD evaluator instance
    poses_gt: Ground truth axis-angle poses, shape (T, 165).
    poses_sampled_pred: List of predicted axis-angle poses, each shape (T, 165).
      Pass a single-element list for argmax/deterministic predictions.
  """
  t = min(poses_gt.shape[0], *[p.shape[0] for p in poses_sampled_pred])
  gt = torch.from_numpy(poses_gt[:t]).to(device).unsqueeze(0)
  gt = rc.axis_angle_to_rotation_6d(gt.reshape(1, t, 55, 3)).reshape(
    1, t, 55 * 6
  )
  fgd.update_target(gt.float())
  for pred in poses_sampled_pred:
    pred = torch.from_numpy(pred[:t]).to(device).unsqueeze(0)
    pred = rc.axis_angle_to_rotation_6d(pred.reshape(1, t, 55, 3)).reshape(
      1, t, 55 * 6
    )
    fgd.update_pred(pred.float())


def update_bc(
  bc: Bc,
  poses: np.ndarray,
  betas: np.ndarray,
  audio: np.ndarray,
  sr: float,
  device: str = util.DEVICE,
):
  """Update BC evaluator with one sequence.

  Trims 2 seconds from each end of the sequence before computing, following
  the BEAT2 evaluation convention used on the GENEA leaderboard.

  Args:
    bc: BC evaluator instance from metric.
    poses: Predicted axis-angle poses, shape (T, 165).
    betas: SMPLX body shape parameters for joint position conversion.
    audio: audio data.
    sr: audio sampling rate.
    device: Device for get_motion_rep_numpy.
  """
  t = poses.shape[0]
  motion_position = get_motion_rep_numpy(poses, device=device, betas=betas)[
    "position"
  ]
  motion_position = motion_position.reshape(t, -1)
  audio_beat = bc.load_audio(
    audio,
    t_start=2 * sr,
    t_end=int((t - 60) / 30 * sr),
    without_file=True,
  )
  motion_beat = bc.load_motion(
    motion_position,
    t_start=60,
    t_end=t - 60,
    pose_fps=30,
    without_file=True,
  )
  bc.compute(audio_beat, motion_beat, length=t - 120, pose_fps=30)


def update_l1div(
  l1div: L1Div,
  poses: np.ndarray,
  betas: np.ndarray,
  device: str = util.DEVICE,
):
  """Update L1Div evaluator with one sequence.

  Args:
    l1div: L1Div evaluator instance from metric.
    poses: Predicted axis-angle poses, shape (T, 165).
    betas: SMPLX body shape parameters for joint position conversion.
    device: Device for get_motion_rep_numpy.
  """
  t = poses.shape[0]
  motion_position = get_motion_rep_numpy(poses, device=device, betas=betas)[
    "position"
  ]
  motion_position = motion_position.reshape(t, -1)
  l1div.compute(motion_position)


def update_srgr(
  srgr: Srgr,
  poses_gt: np.ndarray,
  poses_pred: np.ndarray,
  betas: np.ndarray,
  sem_path: str,
  device: str = util.DEVICE,
):
  """Update SRGR evaluator with one sequence.

  Args:
    srgr: SRGR evaluator instance from metric_modified.
    poses_pred: Predicted axis-angle poses, shape (T, 165).
    poses_gt: Ground truth axis-angle poses, shape (T, 165).
    sem_path: Path to the BEAT2 semantic annotation .txt file.
    betas: SMPLX body shape parameters for joint position conversion.
    device: Device for get_motion_rep_numpy.
  """
  t = min(poses_pred.shape[0], poses_gt.shape[0])
  motion_position_pred = get_motion_rep_numpy(
    poses_pred[:t], device=device, betas=betas
  )["position"].reshape(t, -1)
  motion_position_gt = get_motion_rep_numpy(
    poses_gt[:t], device=device, betas=betas
  )["position"].reshape(t, -1)
  df = pd.read_csv(
    sem_path,
    sep="\t",
    header=None,
    names=["label", "start", "end", "duration", "weight", "comment"],
  )
  srgr.run(motion_position_pred, motion_position_gt, df)


def calculate_activation_statistics(
  activations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  mu = np.mean(activations, axis=0)
  cov = np.cov(activations, rowvar=False)
  return mu, cov


def calculate_frechet_distance(
  mu1: np.ndarray,
  sigma1: np.ndarray,
  mu2: np.ndarray,
  sigma2: np.ndarray,
  eps: float = 1e-6,
) -> float:
  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)

  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)

  assert mu1.shape == mu2.shape, (
    "Training and test mean vectors have different lengths"
  )
  assert sigma1.shape == sigma2.shape, (
    "Training and test covariances have different dimensions"
  )

  diff = mu1 - mu2

  # Product might be almost singular
  covmean = linalg.sqrtm(sigma1.dot(sigma2))
  if not np.isfinite(covmean).all():
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # Numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError(f"Imaginary component {m}")
    covmean = covmean.real

  tr_covmean = np.trace(covmean)

  return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class Fd:
  """Accumulates pose features and computes Fréchet Distance.

  Used for both FD_g (static poses) and FD_k (frame differences).
  """

  def __init__(self):
    self.gt_features = []
    self.pred_features = []

  def update_target(self, frames: np.ndarray):
    """Add ground truth frames, shape (T, D)."""
    self.gt_features.append(frames)

  def update_pred(self, frames: np.ndarray):
    """Add predicted frames, shape (T, D)."""
    self.pred_features.append(frames)

  def compute(self) -> float:
    gt = np.concatenate(self.gt_features, axis=0)
    pred = np.concatenate(self.pred_features, axis=0)
    mu_gt, cov_gt = calculate_activation_statistics(gt)
    mu_pred, cov_pred = calculate_activation_statistics(pred)
    return calculate_frechet_distance(mu_gt, cov_gt, mu_pred, cov_pred)


def update_fd(
  fd_g: Fd,
  fd_k: Fd,
  poses_gt: np.ndarray,
  poses_pred_list: list[np.ndarray],
):
  """Update FD_g and FD_k evaluators with one sequence.

  Args:
    fd_g: Evaluator for static pose Fréchet Distance.
    fd_k: Evaluator for kinematic Fréchet Distance.
    poses_gt: Ground truth axis-angle poses, shape (T, 165).
    poses_pred_list: List of predicted poses, each shape (T, 165).
  """
  gt_static = poses_gt.reshape(poses_gt.shape[0], -1)
  fd_g.update_target(gt_static)
  fd_k.update_target(gt_static[1:] - gt_static[:-1])
  for poses_pred in poses_pred_list:
    t = min(poses_pred.shape[0], poses_gt.shape[0])
    pred_static = poses_pred[:t].reshape(t, -1)
    fd_g.update_pred(pred_static)
    fd_k.update_pred(pred_static[1:] - pred_static[:-1])


class SampleDiv:
  """Accumulates per-sequence sample diversity scores."""

  def __init__(self):
    self.diversity_vals = []

  def update(self, mean_seq_div: float):
    self.diversity_vals.append(mean_seq_div)

  def avg(self) -> float:
    if not self.diversity_vals:
      return 0.0
    return np.mean(self.diversity_vals)


def update_sample_div(
  sample_div: SampleDiv,
  poses_pred_list: list[np.ndarray],
):
  """Update sample diversity evaluator with multiple samples for one sequence.

  Args:
    sample_div: SampleDiv instance.
    poses_pred_list: List of predicted pose arrays for the same
      sequence, each shape (T, 165). Requires at least 2 samples.
  """
  if len(poses_pred_list) < 2:
    return
  min_len = min(p.shape[0] for p in poses_pred_list)
  pose_array = np.stack(
    [p[:min_len].reshape(min_len, -1) for p in poses_pred_list], axis=0
  )  # (n_samples, T, 165)
  cross_sample_var = np.var(
    pose_array.reshape(len(poses_pred_list), -1), axis=0
  )
  sample_div.update(cross_sample_var.mean())


@dataclasses.dataclass
class GestureEvalFile:
  poses_gt: np.ndarray
  poses_pred: list[np.ndarray]
  betas: np.ndarray
  audio: np.ndarray
  sr: int | float
  sem_path: str


class MetricEvaluators:
  def __init__(self):
    self.fgd = Fgd(
      os.path.join(prepare.EVALTOOLS_DIR, "AESKConv_240_100.bin"),
      os.path.join(prepare.EVALTOOLS_DIR),
    )
    self.bc = Bc(
      os.path.join(
        "data",
        "beat2",
        "beat_english_v2.0.0",
        "weights",
        "mean_vel_smplxflame_30.npy",
      )
    )
    self.l1div = L1Div()
    self.srgr = Srgr()
    self.fd_g = Fd()
    self.fd_k = Fd()
    self.sample_div = SampleDiv()

  def finish(self) -> dict[str, int | float]:
    bc = self.bc.avg()
    return {
      "FGD": float(self.fgd.compute()),
      "BC_a2m": float(bc["a2m"]),
      "BC_m2a": float(bc["m2a"]),
      "L1Div": float(self.l1div.avg()),
      "SRGR": float(self.srgr.avg()),
      "FD_g": float(self.fd_g.compute()),
      "FD_k": float(self.fd_k.compute()),
      "SampleDiv": float(self.sample_div.avg()),
    }


def update_evaluators(
  evaluators: MetricEvaluators,
  file: GestureEvalFile,
  device: str = util.DEVICE,
):
  update_fgd(evaluators.fgd, file.poses_gt, file.poses_pred, device=device)
  update_fd(evaluators.fd_g, evaluators.fd_k, file.poses_gt, file.poses_pred)
  update_sample_div(evaluators.sample_div, file.poses_pred)
  for poses_pred in file.poses_pred:
    update_bc(
      evaluators.bc,
      poses_pred,
      file.betas,
      file.audio,
      file.sr,
      device=device,
    )
    update_l1div(evaluators.l1div, poses_pred, file.betas, device=device)
    update_srgr(
      evaluators.srgr,
      file.poses_gt,
      poses_pred,
      file.betas,
      file.sem_path,
      device=device,
    )


def decode_outputs(
  motion_prior: EmageVQModel,
  sampler: Callable[[torch.Tensor], torch.Tensor],
  outputs: dict[str, np.ndarray],
  device: str = util.DEVICE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  decode_kwargs = {}
  for k, v in outputs.items():
    value_pt = torch.from_numpy(v).to(device)
    value_pt = sampler(value_pt) if k.endswith("_index") else value_pt
    decode_kwargs[k] = value_pt.unsqueeze(0)
  motion_prior.to(device)
  with torch.no_grad():
    result = motion_prior.decode(
      **decode_kwargs,
      ref_trans=torch.zeros(1, 1, 3, dtype=torch.float32).to(device),
      get_global_motion=True,
    )
  return (
    result["motion_axis_angle"].squeeze(0).cpu().numpy(),
    result["expression"].squeeze(0).cpu().numpy(),
    result["trans"].squeeze(0).cpu().numpy(),
  )


def load_poses_gt(file_id):
  with np.load(dataset.file_path(prepare.BEAT2_MOTION_DIR, file_id)) as f:
    return f["poses"]


if __name__ == "__main__":
  import argparse

  from scripts import dataset, models
  from scripts import eval_poses as ep

  parser = argparse.ArgumentParser(
    description="Evaluate a latent pose predictor on the test split."
  )
  parser.add_argument("weights", help="Path to predictor weights (.pth file)")
  parser.add_argument(
    "--output",
    "-o",
    default=None,
    metavar="PATH",
    help="Output JSON path (default: weights stem + --objmetrics.json)",
  )
  parser.add_argument("--context-size", type=int, default=150, metavar="N")
  parser.add_argument("--n-layers", type=int, default=2, metavar="N")
  parser.add_argument("--model-dim", type=int, default=768, metavar="N")
  parser.add_argument("--n-heads", type=int, default=12, metavar="N")
  parser.add_argument("--batch-size", type=int, default=32, metavar="N")
  parser.add_argument("--stride", type=int, default=1, metavar="N")
  parser.add_argument(
    "--window-size",
    type=int,
    default=None,
    metavar="N",
    help="Sliding-window alignment size in frames (default: repeat alignment)",
  )
  args = parser.parse_args()

  stem, _ = os.path.splitext(args.weights)
  out = args.output or "objmetrics.json"

  transformers.utils.logging.disable_progress_bar()

  align = (
    dataset.align_inputs_mean_sliding_window(args.window_size)
    if args.window_size
    else dataset.align_inputs_repeat
  )

  predictor = models.TransformerSequenceNoBiasPredictor(
    args.context_size, args.n_layers, args.model_dim, args.n_heads
  )
  predictor.load_state_dict(torch.load(args.weights, weights_only=True))
  predictor.eval()
  predictor.to(util.DEVICE)

  sampler = pl.sampler_nucleus(0.2, 0.3)
  n_samples = 5

  motion_prior = util.load_motion_prior(True)

  evaluators = MetricEvaluators()
  for file_id, aligned_embeds in tqdm(
    dataset.predictor_inputs(False, align),
    desc="file",
    total=len(dataset.TEST_SPLIT_FILES),
    smoothing=0,
  ):
    logits_raw = ep.make_transformer_seq_logits_with_stride(
      predictor, args.context_size, args.stride, args.batch_size, aligned_embeds
    )
    logits = {f"{part}_index": logits_raw[part] for part in ep.BODY_PARTS}
    pred_poses = [
      decode_outputs(motion_prior, sampler, logits)[0] for _ in range(n_samples)
    ]
    poses_gt = load_poses_gt(file_id)
    audio, sr = librosa.load(
      os.path.join(
        "data", "beat2", "beat_english_v2.0.0", "wave16k", f"{file_id}.wav"
      ),
      sr=16000,
    )
    betas = np.load(
      os.path.join(
        "data",
        "beat2",
        "beat_english_v2.0.0",
        "smplxflame_30",
        f"{file_id}.npz",
      )
    )["betas"]
    sem_path = os.path.join(
      "data", "beat2", "beat_english_v2.0.0", "sem", f"{file_id}.txt"
    )
    update_evaluators(
      evaluators,
      GestureEvalFile(poses_gt, pred_poses, betas, audio, sr, sem_path),
    )
    gc.collect()
    torch.cuda.empty_cache()

  with open(out, "w") as f:
    json.dump([{"sampler": "nucleus_p0.2_t0.3", **evaluators.finish()}], f)
