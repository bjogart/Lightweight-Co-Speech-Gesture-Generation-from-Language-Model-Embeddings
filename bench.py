import librosa
from typing import Generator, Callable
import gc
import json
import os
import time
import numpy as np
import torch
import dataset
import models
import util
import pipeline as pl
import random
from related.emage.models.audio.modeling_emage_audio import (
  EmageAudioModel,
  EmageVQModel,
)

N_WARMUP_FILES = 10
ADAPTER_CONTEXT_SIZE = 150


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


def adapter_forward(
  adapter: models.TransformerSequenceNoBiasAdapter,
  stride: int,
  batch_size: int,
  aligned_embeds: np.ndarray,
  latency_stopwatch: Stopwatch,
  context_size: int = ADAPTER_CONTEXT_SIZE,
) -> torch.Tensor:
  preprocess = pl.context_preprocess(context_size, batch_size, stride)
  acc = {}
  with torch.no_grad():
    for batch in preprocess(aligned_embeds):
      latency_stopwatch.start()
      batch_preds = adapter(batch.to(util.DEVICE))
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
  for _ in range(n_reps):
    results.append([])
    for file_id, inputs in make_inputs(True):
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
    "--output", "-o", default="bench_results.json", metavar="PATH",
    help="Output JSON path (default: bench_results.json)",
  )
  parser.add_argument(
    "--n-reps", type=int, default=10, metavar="N",
    help="Number of benchmark repetitions (default: 10)",
  )
  subparsers = parser.add_subparsers(dest="model", required=True)

  adapter_parser = subparsers.add_parser("adapter", help="Benchmark the adapter")
  adapter_parser.add_argument("weights", help="Path to adapter weights (.pth file)")
  adapter_parser.add_argument("--stride", type=int, default=1, metavar="N")
  adapter_parser.add_argument("--context-size", type=int, default=ADAPTER_CONTEXT_SIZE, metavar="N")
  adapter_parser.add_argument("--n-layers", type=int, default=2, metavar="N")
  adapter_parser.add_argument("--model-dim", type=int, default=768, metavar="N")
  adapter_parser.add_argument("--n-heads", type=int, default=12, metavar="N")
  adapter_parser.add_argument("--batch-size", type=int, default=32, metavar="N")

  subparsers.add_parser("emage", help="Benchmark the EMAGE baseline")

  args = parser.parse_args()

  if args.model == "adapter":
    adapter = models.TransformerSequenceNoBiasAdapter(
      args.context_size, args.n_layers, args.model_dim, args.n_heads
    )
    adapter.load_state_dict(torch.load(args.weights, weights_only=True))
    adapter.eval()
    adapter.to(util.DEVICE)
    results = bench(
      n_reps=args.n_reps,
      make_inputs=lambda shuffle: dataset.adapter_inputs(shuffle),
      forward=lambda inputs, timer: adapter_forward(
        adapter, args.stride, args.batch_size, inputs, timer, args.context_size
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
