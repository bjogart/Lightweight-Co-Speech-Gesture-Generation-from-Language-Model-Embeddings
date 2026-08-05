import random
import re
from collections.abc import Callable

import numpy as np
import torch

import pipeline as pl
from PantoMatrix.models.emage_audio.modeling_emage_audio import (
  EmageAudioModel,
  EmageVQModel,
)
from scripts import models, util
from scripts.dataset import TEST_SPLIT_FILES

BODY_PARTS = ["face", "upper", "hands", "lower"]

TEST_SPLIT_SUBSET = random.Random(12345).sample(TEST_SPLIT_FILES, 25)


def slice_eval_snippet(
  text_embeds: np.ndarray,
  text_durations: np.ndarray,
  start_seconds: float,
  end_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
  token_starts = np.roll(np.cumsum(text_durations), 1)
  token_starts[0] = 0.0
  token_ends = token_starts + text_durations
  mask = (token_ends >= start_seconds) & (token_starts < end_seconds)
  sliced_embeds = text_embeds[mask]
  clipped_durations = text_durations[mask].copy()
  clipped_starts = token_starts[mask]
  clipped_ends = token_ends[mask]
  clipped_durations[0] -= max(0.0, start_seconds - clipped_starts[0])
  clipped_durations[-1] -= max(0.0, clipped_ends[-1] - end_seconds)
  return sliced_embeds, clipped_durations


def make_transformer_seq_logits(
  pipeline: pl.GesturePipeline,
  align_inputs: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
  text_embeds: np.ndarray,
  token_durations: np.ndarray,
  device: str = util.DEVICE,
) -> dict[str, np.ndarray]:
  pipeline.model.to(device)
  n_frames = int(np.sum(token_durations) * pl.FPS)
  aligned_embeds = align_inputs(text_embeds, token_durations, n_frames)
  acc = {}
  with torch.no_grad():
    for batch in pipeline.preprocess(aligned_embeds):
      batch_preds = pipeline.model(batch.to(device))
      for part, logits in batch_preds.items():
        acc.setdefault(part, []).append(logits[:, -1, :].cpu())
  pipeline.model.cpu()
  return {k: torch.cat(vs, dim=0).numpy() for k, vs in acc.items()}


def make_transformer_seq_logits_with_stride(
  predictor: models.TransformerSequenceNoBiasPredictor,
  context_size: int,
  stride: int,
  batch_size: int,
  aligned_embeds: np.ndarray,
) -> dict[str, np.ndarray]:
  preprocess = pl.context_preprocess(context_size, batch_size, stride)
  acc = {}
  with torch.no_grad():
    for batch in preprocess(aligned_embeds):
      batch_preds = predictor(batch.to(util.DEVICE))
      for part, logits in batch_preds.items():
        acc.setdefault(part, []).append(logits[:, -stride:, :].cpu())
  return {
    part: torch.cat([t.flatten(0, 1) for t in part_acc])[
      : aligned_embeds.shape[0]
    ].numpy()
    for part, part_acc in acc.items()
  }


def make_emage_logits(
  motion_prior: EmageVQModel,
  model: EmageAudioModel,
  audio: np.ndarray,
  device: str = util.DEVICE,
) -> np.ndarray:
  motion_prior.to(device)
  model.to(device)
  audio_pt = torch.from_numpy(audio).to(device).unsqueeze(0)
  speaker_id = torch.zeros(1, 1).long().to(device)
  with torch.no_grad():
    latent_dict = model.inference(
      audio_pt, speaker_id, motion_prior, masked_motion=None, mask=None
    )
    face_latent = latent_dict["rec_face"]
    upper_index = latent_dict["cls_upper"]
    hands_index = latent_dict["cls_hands"]
    lower_index = latent_dict["cls_lower"]
  return np.stack(
    (
      face_latent.squeeze(0).cpu().numpy(),
      upper_index.squeeze(0).cpu().numpy(),
      hands_index.squeeze(0).cpu().numpy(),
      lower_index.squeeze(0).cpu().numpy(),
    )
  )


def parse_align_from_model_dir(model_dir: str) -> int | None:
  window_match = re.search(r"_window(\d+)", model_dir)
  if window_match:
    window_size = int(window_match.group(1))
    return window_size
  return None
