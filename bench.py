import gc
import json
import os
import random
import time
from collections.abc import Callable, Generator

import librosa
import numpy as np
import torch
from tqdm import tqdm

import pipeline as pl
from PantoMatrix.models.emage_audio.modeling_emage_audio import (
  EmageAudioModel,
  EmageVQModel,
)
from scripts import dataset, models, util

N_WARMUP_FILES = 10
PREDICTOR_CONTEXT_SIZE = 150


class Stopwatch:
  def __init__(self):
    self.elapsed_sec = []
    self._t_start = 0.0

  def start(self):
    torch.cuda.synchronize(util.DEVICE)
    self._t_start = time.perf_counter()

  def end(self):
    torch.cuda.synchronize(util.DEVICE)
    self.elapsed_sec.append(time.perf_counter() - self._t_start)
    self._t_start = 0.0


def predictor_forward(
  predictor: models.TransformerSequenceNoBiasPredictor,
  stride: int,
  batch_size: int,
  aligned_embeds: np.ndarray,
  latency_stopwatch: Stopwatch,
  context_size: int = PREDICTOR_CONTEXT_SIZE,
) -> torch.Tensor:
  preprocess = pl.context_preprocess(context_size, batch_size, stride)
  acc = {}
  with torch.no_grad():
    for batch in preprocess(aligned_embeds):
      latency_stopwatch.start()
      batch_preds = predictor(batch.to(util.DEVICE))
      latency_stopwatch.end()
      for part, logits in batch_preds.items():
        acc.setdefault(part, []).append(logits[:, -stride:, :].cpu())
  res = torch.cat([t.flatten(0, 1) for t in acc["upper"]])
  return res[: aligned_embeds.shape[0]]


def emage_inputs(
  shuffle: bool,
) -> Generator[tuple[str, np.ndarray], None, None]:
  files = dataset.TEST_SPLIT_FILES.copy()
  if shuffle:
    random.shuffle(files)
  for file_id in files:
    audio_path = os.path.join(
      "data", "beat2", "beat_english_v2.0.0", "wave16k", f"{file_id}.wav"
    )
    audio, _ = librosa.load(audio_path, sr=16000)
    yield (file_id, audio)


def emage_forward(
  model: EmageAudioModel,
  prior: EmageVQModel,
  audio: np.ndarray,
  latency_stopwatch: Stopwatch,
) -> torch.Tensor:
  orig_forward = model.forward

  def timed_forward(*args, **kwargs):
    latency_stopwatch.start()
    res = orig_forward(*args, **kwargs)
    latency_stopwatch.end()
    return res

  model.forward = timed_forward  # ty:ignore[invalid-assignment]
  try:
    audio_pt = torch.from_numpy(audio).to(util.DEVICE).unsqueeze(0)
    speaker_id = torch.zeros(1, 1).long().to(util.DEVICE)
    with torch.no_grad():
      res = model.inference(
        audio_pt, speaker_id, prior, masked_motion=None, mask=None
      )
  finally:
    model.forward = orig_forward  # ty:ignore[invalid-assignment]
  return res["cls_upper"].squeeze(0)


def bench(
  n_reps: int,
  make_inputs: Callable[[bool], Generator[tuple[str, np.ndarray], None, None]],
  forward: Callable[[np.ndarray, Stopwatch], torch.Tensor],
) -> list[list[dict[str, str | int | float | list[int | float]]]]:
  stopwatch = Stopwatch()
  for _, (_, inputs) in zip(range(N_WARMUP_FILES), make_inputs(False)):
    forward(inputs, stopwatch)
    gc.collect()
    torch.cuda.empty_cache()

  results: list[list[dict[str, str | int | float | list[int | float]]]] = []
  for _ in tqdm(range(n_reps), position=0):
    results.append([])
    for file_id, inputs in tqdm(
      make_inputs(True),
      position=1,
      leave=False,
      total=sum(1 for _ in make_inputs(False)),
    ):
      latency_stopwatch = Stopwatch()
      torch.cuda.reset_peak_memory_stats(util.DEVICE)
      torch.cuda.synchronize(util.DEVICE)
      t_start = time.perf_counter()
      res = forward(inputs, latency_stopwatch)
      torch.cuda.synchronize(util.DEVICE)
      peak_vram = torch.cuda.max_memory_allocated(util.DEVICE)
      t_end = time.perf_counter()
      n_frames = res.size()[0]
      elapsed = t_end - t_start
      results[-1].append(
        {
          "file_id": file_id,
          "n_frames_generated": n_frames,
          "elapsed_sec": elapsed,
          "peak_vram_bytes": peak_vram,
          "latencies": latency_stopwatch.elapsed_sec,
        }
      )
      gc.collect()
      torch.cuda.empty_cache()
  return results


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
    description="Benchmark gesture generation throughput, latency, and peak VRAM."
  )
  parser.add_argument(
    "--output",
    "-o",
    default="bench_results.json",
    metavar="PATH",
    help="Output JSON path (default: bench_results.json)",
  )
  parser.add_argument(
    "--n-reps",
    type=int,
    default=10,
    metavar="N",
    help="Number of benchmark repetitions (default: 10)",
  )
  subparsers = parser.add_subparsers(dest="model", required=True)

  predictor_parser = subparsers.add_parser(
    "predictor", help="Benchmark the predictor"
  )
  predictor_parser.add_argument(
    "weights", help="Path to predictor weights (.pth file)"
  )
  predictor_parser.add_argument("--stride", type=int, default=1, metavar="N")
  predictor_parser.add_argument(
    "--context-size", type=int, default=PREDICTOR_CONTEXT_SIZE, metavar="N"
  )
  predictor_parser.add_argument("--n-layers", type=int, default=2, metavar="N")
  predictor_parser.add_argument(
    "--model-dim", type=int, default=768, metavar="N"
  )
  predictor_parser.add_argument("--n-heads", type=int, default=12, metavar="N")
  predictor_parser.add_argument(
    "--batch-size", type=int, default=32, metavar="N"
  )
  predictor_parser.add_argument(
    "--window-size",
    type=int,
    default=None,
    metavar="N",
    help="Sliding-window alignment size in frames (default: repeat alignment)",
  )

  subparsers.add_parser("emage", help="Benchmark the EMAGE baseline")

  args = parser.parse_args()

  if args.model == "predictor":
    predictor = models.TransformerSequenceNoBiasPredictor(
      args.context_size, args.n_layers, args.model_dim, args.n_heads
    )
    predictor.load_state_dict(torch.load(args.weights, weights_only=True))
    predictor.eval()
    predictor.to(util.DEVICE)

    align = (
      dataset.align_inputs_mean_sliding_window(args.window_size)
      if args.window_size
      else dataset.align_inputs_repeat
    )

    results = bench(
      n_reps=args.n_reps,
      make_inputs=lambda shuffle: dataset.predictor_inputs(shuffle, align),
      forward=lambda inputs, timer: predictor_forward(
        predictor,
        args.stride,
        args.batch_size,
        inputs,
        timer,
        args.context_size,
      ),
    )
  else:
    prior = util.load_motion_prior(local_files_only=True).to(util.DEVICE).eval()
    emage = util.load_emage(local_files_only=True).to(util.DEVICE).eval()
    results = bench(
      n_reps=args.n_reps,
      make_inputs=lambda shuffle: emage_inputs(shuffle),
      forward=lambda inputs, timer: emage_forward(emage, prior, inputs, timer),
    )

  with open(args.output, mode="w") as f:
    json.dump({"model": args.model, "per_file": results}, f)
