from transformers import (
  AutoTokenizer,
  BitsAndBytesConfig,
  Mistral3ForConditionalGeneration,
  SentencePieceBackend,
  TokenizersBackend,
)
from typing import Any
import torch
import smplx
import numpy as np
import os
from datetime import datetime
from related.emage.models.audio.modeling_emage_audio import (
  EmageVAEConv,
  EmageVQVAEConv,
  EmageVQModel,
  EmageAudioModel,
)

LLM_ID = "mistralai/Ministral-3-3B-Base-2512"
SMPLX_MODEL_DIR = os.path.join("data", "emage_evaltools", "smplx_models")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_smplx_model() -> smplx.SMPLX:
  return smplx.create(
    SMPLX_MODEL_DIR,
    model_type="smplx",
    gender="NEUTRAL_2020",
    use_face_contour=False,
    num_betas=300,
    num_expression_coeffs=100,
    ext="npz",
    use_pca=False,
  ).eval()  # ty:ignore


def load_smplx_faces() -> np.ndarray:
  return np.load(
    os.path.join(SMPLX_MODEL_DIR, "smplx", "SMPLX_NEUTRAL_2020.npz"),
    allow_pickle=True,
  )["f"]


def load_llm_tokenizer(
  local_files_only: bool,
) -> TokenizersBackend | SentencePieceBackend:
  return AutoTokenizer.from_pretrained(
    LLM_ID, local_files_only=local_files_only
  )  # ty: ignore


def load_llm(local_files_only: bool) -> Mistral3ForConditionalGeneration:
  return Mistral3ForConditionalGeneration.from_pretrained(
    LLM_ID, local_files_only=local_files_only
  ).eval()


def load_llm_quantized(
  local_files_only: bool,
) -> Mistral3ForConditionalGeneration:
  return Mistral3ForConditionalGeneration.from_pretrained(
    LLM_ID,
    local_files_only=local_files_only,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
  ).eval()


def load_motion_prior(local_files_only: bool) -> EmageVQModel:
  face_motion_vq = EmageVQVAEConv.from_pretrained(
    "H-Liu1997/emage_audio",
    subfolder="emage_vq/face",
    local_files_only=local_files_only,
  )
  upper_motion_vq = EmageVQVAEConv.from_pretrained(
    "H-Liu1997/emage_audio",
    subfolder="emage_vq/upper",
    local_files_only=local_files_only,
  )
  lower_motion_vq = EmageVQVAEConv.from_pretrained(
    "H-Liu1997/emage_audio",
    subfolder="emage_vq/lower",
    local_files_only=local_files_only,
  )
  hands_motion_vq = EmageVQVAEConv.from_pretrained(
    "H-Liu1997/emage_audio",
    subfolder="emage_vq/hands",
    local_files_only=local_files_only,
  )
  global_motion_ae = EmageVAEConv.from_pretrained(
    "H-Liu1997/emage_audio",
    subfolder="emage_vq/global",
    local_files_only=local_files_only,
  )
  motion_vq = EmageVQModel(
    face_model=face_motion_vq,
    upper_model=upper_motion_vq,
    lower_model=lower_motion_vq,
    hands_model=hands_motion_vq,
    global_model=global_motion_ae,
  )
  motion_vq.eval()
  return motion_vq


def load_emage(local_files_only: bool) -> EmageAudioModel:
  return EmageAudioModel.from_pretrained("H-Liu1997/emage_audio").eval()


def iso_timestamp():
  now = datetime.now()
  return f"{str(now.year)[-2:]}{now.month:02}{now.day:02}T{now.hour:02}{now.minute:02}{now.second:02}"


def entry_info(
  entries: dict[str, Any],
) -> dict[str, tuple[torch.Size | Any, torch.dtype]]:
  sizes = {}
  for k, v in entries.items():
    if type(v) is torch.Tensor:
      sizes[k] = (v.size(), v.dtype)
    elif type(v) is np.ndarray:
      sizes[k] = (v.shape, v.dtype)
    elif type(v) is dict:
      for inner_k, inner_v in entry_info(v).items():
        sizes[f"{k}_{inner_k}"] = inner_v
    else:
      raise Exception(f"cannot compute size of value of type ({type(v)}): {v}")
  return sizes
