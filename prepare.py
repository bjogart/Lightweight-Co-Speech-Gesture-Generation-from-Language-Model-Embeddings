import dataset
import csv
from related.emage.models.audio.modeling_emage_audio import EmageVQModel
from transformers import (
  Mistral3ForConditionalGeneration,
  SentencePieceBackend,
  TokenizersBackend,
)
import torch
import util
import numpy as np
from typing import Callable, Any, Optional
import preprocess
import os
from tqdm import tqdm
from glob import glob
import smplx

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVALTOOLS_DIR = os.path.join("data", "emage_evaltools")
BEAT2_DIR = os.path.join("data", "beat2")
BEAT2_EN_DIR = os.path.join(BEAT2_DIR, "beat_english_v2.0.0")
BEAT2_MOTION_DIR = os.path.join(BEAT2_EN_DIR, "smplxflame_30")
BEAT2_TEXTGRID_DIR = os.path.join(BEAT2_EN_DIR, "textgrid")
FOOT_CONTACT_DIR = os.path.join("data", "foot_contact")
TEXT_EMBEDS_DIR = os.path.join("data", "text_embeds")
POSE_IDXS_DIR = os.path.join("data", "pose_idxs")
PACK_DIR = os.path.join("data", "pack")

BEAT2_SPLITS_CSV = os.path.join(
  BEAT2_DIR, "beat_english_v2.0.0", "train_test_split.csv"
)


def make_glob_pat(dir: str, ext: str) -> str:
  return os.path.join(dir, f"*.{ext}")


def get_file_id(file: str) -> str:
  base = os.path.basename(file)
  file_id, ext = os.path.splitext(base)
  return file_id


def iter_files(
  desc: str,
  glob_pats: list[str],
  make_dest_path: Callable[[str], str],
  read_file: Callable[[list[str]], Optional[dict[str, Any]]],
  process: Callable[[dict[str, Any]], Optional[dict[str, Any]]],
  write_file: Callable[[str, dict[str, Any]], None],
):
  files = {}
  for glob_pat in glob_pats:
    for file in glob(glob_pat):
      file_id = get_file_id(file)
      if file_id not in files:
        files[file_id] = []
      files[file_id].append(file)
  files = {
    file_id: file_paths
    for file_id, file_paths in files.items()
    if len(file_paths) == len(glob_pats)
    and not os.path.exists(make_dest_path(file_id))
  }

  for file_id, files in (
    progress := tqdm(files.items(), desc=desc, smoothing=0)
  ):
    progress.set_postfix_str(file_id)
    dest = make_dest_path(file_id)
    if os.path.exists(dest):
      continue
    data = read_file(files)
    if not data:
      continue
    processed_data = process(data)
    if not processed_data:
      continue
    write_file(dest, processed_data)


def read_npz(file: str) -> dict[str, np.ndarray]:
  return dict(**np.load(file))


def npz_writer(compress_npz: bool) -> Callable[[str, dict[str, Any]], None]:
  def inner(dest: str, data: dict[str, Any]):
    dest_dir, _ = os.path.split(dest)
    os.makedirs(dest_dir, exist_ok=True)
    if compress_npz:
      np.savez_compressed(dest, **data)
    else:
      np.savez(dest, **data)

  return inner


def foot_contact_fn(
  smplx_model: smplx.SMPLX,
) -> Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]:
  def inner(motion: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    foot_contact = preprocess.compute_foot_contact(
      smplx_model,
      torch.from_numpy(motion["betas"]).to(DEVICE),
      torch.from_numpy(motion["poses"]).to(DEVICE),
      torch.from_numpy(motion["expressions"]).float().to(DEVICE),
      torch.from_numpy(motion["trans"]).to(DEVICE),
    )
    return {"foot_contact": foot_contact.cpu().numpy()}

  return inner


def read_textgrid(file: str) -> Optional[dict[str, Any]]:
  # Filter out faulty transcriptions in BEAT2.
  if get_file_id(file) in [
    "3_solomon_0_25_25",  # Transcription is empty
    "3_solomon_0_26_26",  # Transcription is empty
    "2_scott_0_67_67",  # Partial transcription
    "2_scott_0_68_68",  # Partial transcription
  ]:
    return None
  words, word_durations = preprocess.textgrid_words(file)
  return {
    "words": words,
    "word_durations": word_durations,
  }


def text_embeds_fn(
  model: Mistral3ForConditionalGeneration,
  tokenizer: TokenizersBackend | SentencePieceBackend,
) -> Callable[[dict[str, Any]], Optional[dict[str, np.ndarray]]]:
  def inner(data: dict[str, Any]) -> Optional[dict[str, np.ndarray]]:
    text = preprocess.concat_words(data["words"])
    char_durations = preprocess.char_durations_from_words(
      data["words"], data["word_durations"]
    )
    token_data = preprocess.aligned_tokens(tokenizer, text, char_durations)
    text_embeds = preprocess.compute_text_embeds(
      model, torch.from_numpy(token_data.tokens).to(DEVICE)
    )
    return {"text_embeds": text_embeds.float().cpu().numpy()}

  return inner


def read_pose_motion(files: list[str]) -> dict[str, np.ndarray]:
  motion_file, foot_contact_file = files
  motion = np.load(motion_file)
  foot_contact = np.load(foot_contact_file)
  return {
    "poses": motion["poses"],
    "expressions": motion["expressions"],
    "trans": motion["trans"],
    "foot_contact": foot_contact["foot_contact"],
  }


def pose_idxs_fn(
  motion_prior: EmageVQModel,
) -> Callable[[dict[str, Any]], dict[str, np.ndarray]]:
  def inner(motion: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    idxs = preprocess.compute_pose_idxs(
      motion_prior,
      torch.from_numpy(motion["poses"]).to(DEVICE),
      torch.from_numpy(motion["expressions"]).float().to(DEVICE),
      torch.from_numpy(motion["trans"]).float().to(DEVICE),
      torch.from_numpy(motion["foot_contact"]).to(DEVICE),
    )
    return {part: idx.cpu().numpy() for part, idx in idxs.items()}

  return inner


def beat2_splits() -> dict[str, int]:
  splits = {}
  with open(BEAT2_SPLITS_CSV, newline="") as f:
    reader = iter(csv.reader(f))
    _header = next(reader)
    for row in reader:
      file_id, split_str = row
      split_idx = dataset.SPLIT_IDXS[split_str]
      splits[file_id] = split_idx
  return splits


def read_pack_data(files: list[str]) -> dict[str, Any]:
  textgrid_file, text_embeds_file, pose_idxs_file = files
  words, word_durations = preprocess.textgrid_words(textgrid_file)
  text_embeds = read_npz(text_embeds_file)
  pose_idxs = read_npz(pose_idxs_file)
  return dict(
    file_id=get_file_id(textgrid_file),
    words=words,
    word_durations=word_durations,
    **text_embeds,
    pose_idxs=pose_idxs,
  )


def pack_data_fn(
  splits: dict[str, int], tokenizer: TokenizersBackend | SentencePieceBackend
) -> Callable[[dict[str, Any]], dict[str, str | int | np.ndarray]]:
  def inner(data: dict[str, Any]) -> dict[str, str | int | np.ndarray]:
    text = preprocess.concat_words(data["words"])
    char_durations = preprocess.char_durations_from_words(
      data["words"], data["word_durations"]
    )
    token_data = preprocess.aligned_tokens(tokenizer, text, char_durations)
    return {
      "text": text,
      "token_offsets": token_data.token_offsets,
      "split": splits[data["file_id"]],
      "text_durations": token_data.token_durations,
      "text_embeds": data["text_embeds"],
      "face_idx": data["pose_idxs"]["face"],
      "upper_idx": data["pose_idxs"]["upper"],
      "hands_idx": data["pose_idxs"]["hands"],
      "lower_idx": data["pose_idxs"]["lower"],
    }

  return inner


def train_setup():
  if not os.path.exists(EVALTOOLS_DIR):
    preprocess.download_evaltools(EVALTOOLS_DIR)
  if not os.path.exists(BEAT2_DIR):
    preprocess.download_beat2(BEAT2_DIR)
  iter_files(
    "foot contact positions",
    [make_glob_pat(BEAT2_MOTION_DIR, "npz")],
    lambda file_id: os.path.join(FOOT_CONTACT_DIR, f"{file_id}.npz"),
    lambda paths: read_npz(paths[0]),
    foot_contact_fn(util.load_smplx_model().to(DEVICE)),
    npz_writer(True),
  )
  iter_files(
    "text embeddings",
    [make_glob_pat(BEAT2_TEXTGRID_DIR, "TextGrid")],
    lambda file_id: os.path.join(TEXT_EMBEDS_DIR, f"{file_id}.npz"),
    lambda paths: read_textgrid(paths[0]),
    text_embeds_fn(
      util.load_llm(False).to(DEVICE),  # ty:ignore
      util.load_llm_tokenizer(False),
    ),
    npz_writer(True),
  )
  iter_files(
    "pose indices",
    [
      make_glob_pat(BEAT2_MOTION_DIR, "npz"),
      make_glob_pat(FOOT_CONTACT_DIR, "npz"),
    ],
    lambda file_id: os.path.join(POSE_IDXS_DIR, f"{file_id}.npz"),
    read_pose_motion,
    pose_idxs_fn(util.load_motion_prior(False).to(DEVICE)),
    npz_writer(True),
  )
  iter_files(
    "pack data",
    [
      make_glob_pat(BEAT2_TEXTGRID_DIR, "TextGrid"),
      make_glob_pat(TEXT_EMBEDS_DIR, "npz"),
      make_glob_pat(POSE_IDXS_DIR, "npz"),
    ],
    lambda file_id: os.path.join(PACK_DIR, f"{file_id}.npz"),
    read_pack_data,
    pack_data_fn(beat2_splits(), util.load_llm_tokenizer(False)),
    npz_writer(False),
  )


if __name__ == "__main__":
  train_setup()
