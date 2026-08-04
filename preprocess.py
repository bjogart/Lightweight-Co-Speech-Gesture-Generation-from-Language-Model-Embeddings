from dataclasses import dataclass
import related.emage.utils.rotation_conversions as rc
from related.emage.models.audio.modeling_emage_audio import EmageVQModel
from tgt import read_textgrid
import numpy as np
from transformers import (
  Mistral3ForConditionalGeneration,
  TokenizersBackend,
  SentencePieceBackend,
)
from huggingface_hub import snapshot_download
import torch
import smplx


def download_beat2(dest_dir: str):
  snapshot_download("H-Liu1997/BEAT2", repo_type="dataset", local_dir=dest_dir)


def download_evaltools(dest_dir: str):
  snapshot_download(
    "H-Liu1997/emage_evaltools", repo_type="model", local_dir=dest_dir
  )


# Adapted from https://github.com/PantoMatrix/PantoMatrix/blob/3b19fb86cc8830d847f78102118cb3e9c7c99df6/datasets/foot_contact.py
def compute_foot_contact(
  smplx_model: smplx.SMPLX,
  betas: torch.Tensor,
  poses: torch.Tensor,
  expressions: torch.Tensor,
  trans: torch.Tensor,
) -> torch.Tensor:
  CHUNK_SIZE = 128
  n, c = poses.shape
  betas = torch.tile(betas.reshape(1, 300), (n, 1))

  all_joints = []
  for start in range(0, n, CHUNK_SIZE):
    with torch.no_grad():
      end = min(start + CHUNK_SIZE, n)
      sl = slice(start, end)
      joints = smplx_model(
        betas=betas[sl],
        transl=trans[sl],
        expression=expressions[sl],
        jaw_pose=poses[sl, 66:69],
        global_orient=poses[sl, :3],
        body_pose=poses[sl, 3:66],
        left_hand_pose=poses[sl, 75:120],
        right_hand_pose=poses[sl, 120:165],
        leye_pose=poses[sl, 69:72],
        reye_pose=poses[sl, 72:75],
        return_joints=True,
      )["joints"][:, (7, 8, 10, 11), :]
    all_joints.append(joints)

  joints = torch.cat(all_joints, dim=0).permute(1, 0, 2)  # (4, n, 3)
  feetv = torch.zeros(4, n)
  feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
  contacts = (feetv < 0.01).float()  # (n, 4)
  return contacts.transpose(1, 0)


def textgrid_words(path: str) -> tuple[list[str], np.ndarray]:
  words_tier = read_textgrid(path).get_tier_by_name("words")
  words = [w for w in words_tier if w.text.strip()]
  durations = np.zeros(len(words), dtype=np.float32)
  for idx, word in enumerate(words):
    gap_before = (
      word.start_time - words[idx - 1].end_time
      if idx != 0
      else 2 * word.start_time
    )
    gap_after = (
      words[idx + 1].start_time - word.end_time
      if idx != len(words) - 1
      else 2 * (words_tier.end_time - word.end_time)
    )
    duration = (
      word.end_time - word.start_time + (gap_before / 2) + (gap_after / 2)
    )
    durations[idx] = duration
  words = [w.text.strip() for w in words]
  return (words, durations)


def concat_words(words: list[str]) -> str:
  return " ".join(words)


def char_durations_from_words(words: list[str], word_durations: np.ndarray):
  # Uniform durations by word, plus 0.0 for the space character inserted by `concat_words`.
  ch_durations_by_word = [
    [duration / len(w)] * len(w) + [0.0]
    for w, duration in zip(words, word_durations)
  ]
  return np.array(
    [ch for w in ch_durations_by_word for ch in w][:-1], dtype=np.float32
  )


@dataclass
class AlignedTokens:
  tokens: np.ndarray
  token_durations: np.ndarray
  token_offsets: np.ndarray


def aligned_tokens(
  tokenizer: TokenizersBackend | SentencePieceBackend,
  text: str,
  char_durations: np.ndarray,
) -> AlignedTokens:
  enc = tokenizer(
    text,
    return_special_tokens_mask=True,
    return_offsets_mapping=True,
    return_tensors="pt",
  )
  last_offset = 0
  tokens = enc["input_ids"].squeeze(0).cpu().numpy()
  n = tokens.shape[0]

  token_durations = np.zeros(n, dtype=np.float32)
  token_offsets = np.zeros(n, dtype=np.int32)
  offsets = enc["offset_mapping"].cpu().numpy().squeeze(0)
  for idx, (start_idx, end_idx) in enumerate(offsets):
    (start_idx, end_idx) = (int(start_idx), int(end_idx))
    start = min(last_offset, start_idx)
    token_offsets[idx] = start
    token_durations[idx] = sum(char_durations[start:end_idx])
    last_offset = end_idx
  token_durations *= (
    enc["special_tokens_mask"].cpu().numpy().squeeze(0) != 1
  ).astype(np.float32)
  return AlignedTokens(tokens, token_durations, token_offsets)


def compute_text_embeds(
  model: Mistral3ForConditionalGeneration,
  tokens: torch.Tensor,
) -> torch.Tensor:
  with torch.no_grad():
    outputs = model(tokens.unsqueeze(0), output_hidden_states=True)
    return outputs.hidden_states[-1].squeeze(0)


def compute_pose_idxs(
  motion_prior: EmageVQModel,
  poses: torch.Tensor,
  expressions: torch.Tensor,
  trans: torch.Tensor,
  foot_contact: torch.Tensor,
) -> dict[str, torch.Tensor]:
  t, _ = poses.size()
  motion_6d = rc.axis_angle_to_rotation_6d(poses.reshape(t, 55, 3)).reshape(
    t, 330
  )
  with torch.no_grad():
    idxs = motion_prior.map2index(
      motion_6d.unsqueeze(0),
      expressions.unsqueeze(0),
      tar_contact=foot_contact.unsqueeze(0),
      tar_trans=trans.unsqueeze(0),
    )
  return {part: idx.squeeze(0) for part, idx in idxs.items()}
