import math
import os
import shutil
import subprocess
from collections.abc import Callable, Generator
from dataclasses import dataclass

import numpy as np
import torch
import transformers.utils.logging
from torch import nn
from tqdm import tqdm
from transformers import (
  Mistral3ForConditionalGeneration,
  SentencePieceBackend,
  TokenizersBackend,
)

from PantoMatrix.models.emage_audio.modeling_emage_audio import EmageVQModel
from scripts import dataset, models, preprocess, render, util

# Output frame rate.
FPS = 30
# Average token rate derived from BEAT2 dataset. (165 tokens/min, approx. 124 words/min).
TOKENS_PER_SEC = 2.75


def sampler_argmax() -> Callable[[torch.Tensor], torch.Tensor]:
  def sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)

  return sample


def sampler_stochastic(
  temperature: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
  def sample(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(
      probs.view(-1, probs.shape[-1]), num_samples=1
    ).view(probs.shape[:-1])

  return sample


def sampler_top_k(
  k: int, temperature: float
) -> Callable[[torch.Tensor], torch.Tensor]:
  def sample(logits: torch.Tensor) -> torch.Tensor:
    top_k_logits, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_logits[..., -1, None]
    filtered = logits.masked_fill(logits < threshold, float("-inf"))
    probs = torch.softmax(filtered / temperature, dim=-1)
    return torch.multinomial(
      probs.view(-1, probs.shape[-1]), num_samples=1
    ).view(probs.shape[:-1])

  return sample


def sampler_nucleus(
  p: float, temperature: float
) -> Callable[[torch.Tensor], torch.Tensor]:
  def sample(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative - sorted_probs > p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    flat = sorted_probs.view(-1, sorted_probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1).view(
      sorted_probs.shape[:-1]
    )
    return sorted_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

  return sample


@dataclass
class GesturePipeline:
  """Bundles a latent pose predictor with model-specific pre- and postprocessing.

  Different predictor architectures require different input shapes and output
  decoding strategies. This dataclass groups them together so that
  generate_gestures() can operate on any predictor through a uniform interface.

  Attributes:
    model: The latent pose predictor module.
    preprocess: Takes aligned text embeddings of shape (n_frames, embed_dim)
      and yields batches of model inputs as tensors, one batch at a time.
      Responsible for chunking, padding, and any reshaping the predictor needs.
    postprocess: Takes one batch of raw predictor outputs and converts them to
      a dict of keyword arguments for motion_prior.decode(). Keys should match
      the parameter names of EmageVQModel.decode(), e.g. "face_index",
      "upper_index", etc.
    batch_size: Number of poses to predict per batch.
  """

  model: nn.Module
  preprocess: Callable[[np.ndarray], Generator[torch.Tensor, None, None]]
  postprocess: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
  batch_size: int


def context_preprocess(
  context_size: int, batch_size: int, stride: int = 1
) -> Callable[[np.ndarray], Generator[torch.Tensor, None, None]]:
  """Return a preprocess function for transformer-based predictors.

  Pads the input sequence so that the first frame has a full context window,
  then yields overlapping chunks of shape (batch_size, context_size, embed_dim)
  one batch at a time. Only one batch of chunks is held in memory at once.

  Args:
    context_size: Number of frames in each input chunk (the predictor's context window).
    batch_size: Number of chunks to yield per batch.
    stride: Step size between consecutive output frames. Defaults to 1 (every
      frame). With stride > 1 each batch covers batch_size * stride input frames.
  """

  def preprocess(
    text_embeds: np.ndarray,
  ) -> Generator[torch.Tensor, None, None]:
    n_frames = text_embeds.shape[0]
    # Pad by repeating the first frame so every output frame has a full context.
    padded = np.pad(text_embeds, ((context_size - 1, 0), (0, 0)), mode="edge")
    for start_idx in range(0, n_frames, batch_size * stride):
      end_idx = min(start_idx + batch_size * stride, n_frames)
      batch = np.stack(
        [
          padded[i : i + context_size]
          for i in range(start_idx, end_idx, stride)
        ]
      )
      yield torch.from_numpy(batch)

  return preprocess


def last_idx_postprocessor(
  sampler: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
  """Return a postprocess function for sequence-predicting index predictors.

  For each body part, selects logits for the last frame in the context window
  and applies the sampler to obtain a codebook index.

  Args:
    sampler: Function mapping logits of shape (..., n_codebook_entries) to
      index tensor of the same shape minus the last dimension. Use
      sampler_stochastic(), sampler_top_k(), or sampler_nucleus() for
      stochastic sampling.

  Returns:
    A postprocess function mapping predictor outputs to a dict of
    "{part}_index" tensors of shape (1, batch) suitable for motion_prior.decode().
  """

  def postprocess(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
      f"{part}_index": sampler(part_outputs[:, -1, :]).unsqueeze(0)
      for part, part_outputs in outputs.items()
    }

  return postprocess


def last_idx_postprocessor_no_lower(
  sampler: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
  """Return a postprocess function for sequence-predicting index predictors.

  For each body part, selects logits for the last frame in the context window
  and applies the sampler to obtain a codebook index.
  The "lower" part is ignored.

  Args:
    sampler: Function mapping logits of shape (..., n_codebook_entries) to
      index tensor of the same shape minus the last dimension. Use
      sampler_stochastic(), sampler_top_k(), or sampler_nucleus() for
      stochastic sampling.

  Returns:
    A postprocess function mapping predictor outputs to a dict of
    "{part}_index" tensors of shape (1, batch) suitable for motion_prior.decode().
  """

  def postprocess(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
      f"{part}_index": sampler(part_outputs[:, -1, :]).unsqueeze(0)
      for part, part_outputs in outputs.items()
      if part != "lower"
    }

  return postprocess


def idx_postprocessor(
  sampler: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
  """Return a postprocess function for a last-pose predicting index predictors.

  For each body part, selects logits for the given frame of the context window
  and apply the sampler to obtain a codebook index.

  Args:
    sampler: Function mapping logits of shape (..., n_codebook_entries) to
      index tensor of the same shape minus the last dimension. Use
      sampler_stochastic(), sampler_top_k(), or sampler_nucleus() for
      stochastic sampling.

  Returns:
    A postprocess function mapping predictor outputs to a dict of
    "{part}_index" tensors of shape (1, batch) suitable for motion_prior.decode().
  """

  def postprocess(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
      f"{part}_index": sampler(part_outputs).unsqueeze(0)
      for part, part_outputs in outputs.items()
    }

  return postprocess


def make_transformer_pipeline(
  make_predictor: Callable[[int, int, int, int], nn.Module],
  weights_path: str,
  context_size: int,
  n_layers: int,
  model_dim: int,
  n_heads: int,
  postprocess: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
  batch_size: int,
) -> Callable[[], GesturePipeline]:
  """Return a factory function that constructs a transformer GesturePipeline.

  The predictor is not loaded until the returned factory is called, so this
  function is cheap to call and suitable for passing to render_gestures_from_text().

  Args:
    make_predictor: Constructor for the predictor module, called with
      (context_size, n_layers, model_dim, n_heads).
    weights_path: Path to the saved predictor weights (.pth file).
    context_size: Context window size in frames.
    n_layers: Number of transformer decoder layers.
    model_dim: Hidden dimension of the transformer.
    n_heads: Number of attention heads.
    postprocess: Postprocessing function for this predictor's output format.
    batch_size: Number of chunks to process per forward pass.
  """

  def make():
    predictor = make_predictor(context_size, n_layers, model_dim, n_heads)
    predictor.load_state_dict(torch.load(weights_path, weights_only=True))
    predictor.eval()
    return GesturePipeline(
      predictor,
      context_preprocess(context_size, batch_size),
      postprocess,
      batch_size,
    )

  return make


def make_transformer_nobias_custom_parts_pipeline(
  weights_path: str,
  parts: list[str],
  context_size: int,
  n_layers: int,
  model_dim: int,
  n_heads: int,
  sampler: Callable[[torch.Tensor], torch.Tensor],
  batch_size: int,
) -> Callable[[], GesturePipeline]:
  """Factory for TransformerSequenceNoBiasCustomHeadsPredictor pipelines.

  This predictor predicts a codebook index for every frame in the context window;
  only the prediction for the last frame is used.
  """
  return make_transformer_pipeline(
    lambda context_size, n_layers, model_dim, n_heads: (
      models.TransformerSequenceNoBiasCustomPartsPredictor(
        parts, context_size, n_layers, model_dim, n_heads
      )
    ),
    weights_path,
    context_size,
    n_layers,
    model_dim,
    n_heads,
    last_idx_postprocessor(sampler),
    batch_size,
  )


def make_transformer_nobias_pipeline(
  weights_path: str,
  context_size: int,
  n_layers: int,
  model_dim: int,
  n_heads: int,
  sampler: Callable[[torch.Tensor], torch.Tensor],
  batch_size: int,
) -> Callable[[], GesturePipeline]:
  """Factory for TransformerSequenceNoBiasPredictor pipelines.

  This predictor predicts a codebook index for every frame in the context window;
  only the prediction for the last frame is used.
  """
  return make_transformer_pipeline(
    models.TransformerSequenceNoBiasPredictor,
    weights_path,
    context_size,
    n_layers,
    model_dim,
    n_heads,
    last_idx_postprocessor(sampler),
    batch_size,
  )


def make_transformer_idx_pipeline(
  weights_path: str,
  context_size: int,
  n_layers: int,
  model_dim: int,
  n_heads: int,
  sampler: Callable[[torch.Tensor], torch.Tensor],
  batch_size: int,
) -> Callable[[], GesturePipeline]:
  """Factory for TransformerLastIndexPredictor pipelines.

  This predictor predicts a single codebook index per input frame directly.
  """
  return make_transformer_pipeline(
    models.TransformerLastIndexPredictor,
    weights_path,
    context_size,
    n_layers,
    model_dim,
    n_heads,
    idx_postprocessor(sampler),
    batch_size,
  )


def make_transformer_seq_pipeline(
  weights_path: str,
  context_size: int,
  n_layers: int,
  model_dim: int,
  n_heads: int,
  sampler: Callable[[torch.Tensor], torch.Tensor],
  batch_size: int,
) -> Callable[[], GesturePipeline]:
  """Factory for TransformerSequenceIndexPredictor pipelines.

  This predictor predicts a codebook index for every frame in the context window;
  only the prediction for the last frame is used.
  """
  return make_transformer_pipeline(
    models.TransformerSequenceIndexPredictor,
    weights_path,
    context_size,
    n_layers,
    model_dim,
    n_heads,
    last_idx_postprocessor(sampler),
    batch_size,
  )


def generate_gestures(
  motion_prior: EmageVQModel,
  pipeline: GesturePipeline,
  text_embeds: np.ndarray,
  device: str = util.DEVICE,
  n_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Run the latent pose predictor and decode the result to poses and expressions.

  Streams input through the predictor one batch at a time to limit memory usage.
  Postprocessed outputs (codebook indices or latents) are accumulated across
  batches and concatenated before a single call to the motion prior decoder,
  which needs the full sequence to apply its convolutional smoothing.

  Args:
    motion_prior: Pretrained EMAGE VQ motion prior used for decoding.
    pipeline: GesturePipeline containing the predictor and its pre/postprocessing.
    text_embeds: Aligned text embeddings of shape (n_frames, embed_dim).
    device: Device to run the predictor on.
    n_batches: Optional total batch count, used only for progress display.

  Returns:
    Tuple of (poses, expressions, translation) as float32 numpy arrays.
  """
  acc = {}
  with torch.no_grad():
    pipeline.model.to(device)
    batches = pipeline.preprocess(text_embeds)
    progress = (
      tqdm(batches, desc="Gestures", total=n_batches, smoothing=0)
      if n_batches
      else batches
    )
    for batch in progress:
      batch_preds = pipeline.model(batch.to(device))
      batch_encs = pipeline.postprocess(
        {k: v.cpu() for k, v in batch_preds.items()}
      )
      for k, v in batch_encs.items():
        acc.setdefault(k, []).append(v)
    pipeline.model.cpu()
    # Concatenate across batches along the sequence dimension before decoding.
    # The motion prior is a convolutional model and must see the full sequence.
    concats = {k: torch.cat(vs, dim=1).to(device) for k, vs in acc.items()}
    motion_prior.to(device)
    res = motion_prior.decode(
      **concats,
      ref_trans=torch.zeros(1, 3, dtype=torch.float32).to(device),
      get_global_motion=True,
    )
    motion_prior.cpu()
  poses = res["motion_axis_angle"].cpu().squeeze(0).numpy()
  expressions = res["expression"].cpu().squeeze(0).numpy()
  trans = res["trans"].cpu().squeeze(0).numpy()
  return poses, expressions, trans


def tokenize_text(
  tokenizer: TokenizersBackend | SentencePieceBackend,
  text: str,
  char_durations: np.ndarray | None = None,
) -> preprocess.AlignedTokens:
  """Tokenize text and compute per-token timing information.

  If char_durations is not provided, durations are estimated from the average
  token rate in BEAT2 (TOKENS_PER_SEC). Token durations are derived from
  character durations: longer tokens (in characters) receive proportionally
  longer durations.

  Because tokenizing individual words produces different results than
  tokenizing the full text, the full text is always tokenized at once.
  A two-pass approach is used when estimating durations: the first pass with
  dummy character durations determines token count, which is then used to
  compute a uniform character duration for the second pass.

  Args:
    tokenizer: Mistral tokenizer.
    text: Input text string.
    char_durations: Optional array of per-character durations in seconds,
      of length len(text). If None, durations are estimated uniformly.

  Returns:
    AlignedTokens with token ids, per-token durations, and character offsets.
  """
  if char_durations is None:
    # First pass: dummy durations to count tokens.
    dummy_token_data = preprocess.aligned_tokens(
      tokenizer, text, np.ones(len(text), dtype=np.float32)
    )
    total_duration = dummy_token_data.tokens.shape[0] / TOKENS_PER_SEC
    char_durations = np.full(
      len(text), total_duration / len(text), dtype=np.float32
    )
  return preprocess.aligned_tokens(tokenizer, text, char_durations)


def compute_text_embeds(
  llm: Mistral3ForConditionalGeneration,
  tokens: np.ndarray,
  device: str = util.DEVICE,
) -> np.ndarray:
  """Extract last hidden layer embeddings from the LLM for a token sequence.

  Moves the model to device for inference, then back to CPU to free VRAM.

  Args:
    llm: Pretrained Ministral-3B model.
    tokens: Token id array of shape (n_tokens,).
    device: Device to run inference on.

  Returns:
    Float32 numpy array of shape (n_tokens, 3072).
  """
  llm.to(device)  # ty: ignore
  tokens_pt = torch.from_numpy(tokens).to(device)
  text_embeds = preprocess.compute_text_embeds(llm, tokens_pt)
  llm.cpu()
  return text_embeds.float().cpu().numpy()


@dataclass
class TokenData:
  """Token ids, durations, and character offsets for a text sequence.

  Holds everything needed for subtitle generation and frame alignment
  after tokenization, so it can be passed between pipeline stages.

  Attributes:
    token_durations: Per-token durations in seconds, shape (n_tokens,).
    token_offsets: Character offsets into the source string, shape (n_tokens,).
    text: The original text string, used for subtitle generation.
  """

  token_durations: np.ndarray
  token_offsets: np.ndarray
  text: str


def render_gestures_from_embeds(
  make_pipeline: Callable[[], GesturePipeline],
  aligned_embeds: np.ndarray,
  token_data: TokenData,
  dest: str,
  add_subs: bool = False,
  audio_path: str | None = None,
  device: str = util.DEVICE,
  log_actions: bool = True,
):
  """Render gestures from pre-computed aligned text embeddings.

  Picks up the pipeline after embedding extraction, so the LLM does not
  need to be loaded. Useful for visualization loops that reuse embeddings
  from pack files across multiple predictor checkpoints.

  Args:
    make_pipeline: Zero-argument factory that constructs a GesturePipeline.
    aligned_embeds: Frame-aligned text embeddings, shape (n_frames, embed_dim).
    token_data: Token durations, offsets, and source text for subtitle generation.
    dest: Output path for the final video.
    add_subs: If True, burn karaoke-style subtitles into the video.
    audio_path: Optional path to an audio file to mix into the video.
    device: Device for predictor and motion prior inference.
    log_actions: If True, print each pipeline stage to stdout.
  """

  def do_render(logger):
    n_frames = aligned_embeds.shape[0]

    logger("Load pipeline")
    pipeline = make_pipeline()
    logger("Generate gestures")
    motion_prior = util.load_motion_prior(True)
    poses, expressions, _ = generate_gestures(
      motion_prior,
      pipeline,
      aligned_embeds,
      device=device,
      n_batches=math.ceil(n_frames / pipeline.batch_size)
      if log_actions
      else None,
    )
    del motion_prior
    del pipeline
    torch.cuda.empty_cache()

    logger("Render gestures")
    render_path = f"_render--{util.iso_timestamp()}.mp4"
    vid_path = render_path
    render.render_to_file(
      render_path, poses, expressions, show_progress=log_actions
    )

    subs_path = f"_subs--{util.iso_timestamp()}.srt"
    subbed_path = f"_subbed--{util.iso_timestamp()}.mp4"
    if add_subs:
      logger("Add subtitles")
      subs = render.subs(
        token_data.text,
        token_data.token_offsets,
        token_data.token_durations,
      )
      with open(subs_path, "w", encoding="utf-8") as f:
        f.write(subs)
      subprocess.run(
        [
          "ffmpeg",
          "-loglevel",
          "quiet",
          "-i",
          vid_path,
          "-filter_complex",
          f"subtitles={subs_path}",
          subbed_path,
        ],
        check=True,
      )
      vid_path = subbed_path

    dubbed_path = f"_dubbed--{util.iso_timestamp()}.mp4"
    if audio_path:
      logger("Add audio track")
      subprocess.run(
        [
          "ffmpeg",
          "-loglevel",
          "quiet",
          "-i",
          vid_path,
          "-i",
          audio_path,
          dubbed_path,
        ],
        check=True,
      )
      vid_path = dubbed_path

    if not dest.lower().endswith(".mp4"):
      dest_path = f"{dest}.mp4"
    else:
      dest_path = dest
    logger(f"Move MP4 to {dest_path}")
    shutil.move(vid_path, dest_path)

    for path in [render_path, subs_path, subbed_path, dubbed_path]:
      if os.path.exists(path):
        logger(f"Clean up {path}")
        os.remove(path)

  if not log_actions:
    transformers.utils.logging.disable_progress_bar()
  try:
    logger = print if log_actions else lambda _: None
    do_render(logger)
  finally:
    if not log_actions:
      transformers.utils.logging.enable_progress_bar()


def render_gestures_from_text(
  make_pipeline: Callable[[], GesturePipeline],
  align_embeds: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
  text: str,
  dest: str,
  add_subs: bool = False,
  audio_path: str | None = None,
  device: str = util.DEVICE,
  log_actions: bool = True,
):
  """Run the full text-to-gesture pipeline and render the result to a video file.

  Orchestrates the full pipeline in sequence:
    1. Tokenize text and estimate timing.
    2. Extract LLM embeddings (LLM is deleted after to free VRAM).
    3. Align embeddings to the motion frame rate.
    4. Run the latent pose predictor and decode via the motion prior.
    5. Render to video, optionally with subtitles and audio.

  Args:
    make_pipeline: Zero-argument factory that constructs a GesturePipeline.
    align_embeds: `dataset.align_inputs_repeat` or `dataset.align_inputs_mean_sliding_window(window_size_frames)`
    text: Input text to generate gestures for.
    dest: Output path for the final video.
    add_subs: If True, burn karaoke-style subtitles into the video.
    audio_path: Optional path to an audio file to mix into the video.
    device: Device for inference.
    log_actions: If True, print each pipeline stage to stdout.
  """
  if not log_actions:
    transformers.utils.logging.disable_progress_bar()
  try:
    logger = print if log_actions else lambda _: None

    logger("Tokenize text")
    tokenizer = util.load_llm_tokenizer(True)
    token_data_raw = tokenize_text(tokenizer, text)

    logger("Load language model")
    llm = util.load_llm(True)
    logger("Compute text embeddings")
    text_embeds = compute_text_embeds(llm, token_data_raw.tokens, device)
    del llm
    torch.cuda.empty_cache()

    logger("Align text embeddings")
    n_frames = int(np.sum(token_data_raw.token_durations) * FPS)
    aligned_embeds = align_embeds(
      text_embeds,
      token_data_raw.token_durations,
      n_frames,
    )

    token_data = TokenData(
      token_durations=token_data_raw.token_durations,
      token_offsets=token_data_raw.token_offsets,
      text=text,
    )

    render_gestures_from_embeds(
      make_pipeline,
      aligned_embeds,
      token_data,
      dest,
      add_subs=add_subs,
      audio_path=audio_path,
      device=device,
      log_actions=log_actions,
    )
  finally:
    if not log_actions:
      transformers.utils.logging.enable_progress_bar()


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
    description="Generate and render co-speech gestures from text."
  )
  parser.add_argument("weights", help="Path to predictor weights (.pth file)")
  parser.add_argument("text", help="Input text to generate gestures for")
  parser.add_argument(
    "--output",
    "-o",
    default="render.mp4",
    metavar="PATH",
    help="Output video path (default: render.mp4)",
  )
  parser.add_argument(
    "--audio", metavar="PATH", help="Audio file to mix into the video"
  )
  parser.add_argument(
    "--subs", action="store_true", help="Burn karaoke subtitles into the video"
  )
  parser.add_argument("--context-size", type=int, default=150, metavar="N")
  parser.add_argument("--n-layers", type=int, default=2, metavar="N")
  parser.add_argument("--model-dim", type=int, default=768, metavar="N")
  parser.add_argument("--n-heads", type=int, default=12, metavar="N")
  parser.add_argument("--batch-size", type=int, default=32, metavar="N")
  parser.add_argument(
    "--window-size",
    type=int,
    default=None,
    metavar="N",
    help="Sliding-window alignment size in frames (default: repeat alignment)",
  )
  parser.add_argument(
    "--no-lower",
    action="store_true",
    help="Ignore lower body predictions",
  )
  args = parser.parse_args()

  align = (
    dataset.align_inputs_mean_sliding_window(args.window_size)
    if args.window_size
    else dataset.align_inputs_repeat
  )
  postprocess_fn = (
    last_idx_postprocessor_no_lower if args.no_lower else last_idx_postprocessor
  )
  make_pipeline = make_transformer_pipeline(
    models.TransformerSequenceNoBiasPredictor,
    args.weights,
    args.context_size,
    args.n_layers,
    args.model_dim,
    args.n_heads,
    postprocess_fn(sampler_nucleus(0.2, 0.3)),
    batch_size=args.batch_size,
  )
  render_gestures_from_text(
    make_pipeline,
    align,
    args.text,
    args.output,
    add_subs=args.subs,
    audio_path=args.audio,
  )
