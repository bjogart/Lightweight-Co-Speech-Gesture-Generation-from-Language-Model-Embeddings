from numpy.lib.npyio import NpzFile
import random
import torch
from glob import glob
import os
from typing import Callable, Generator, Optional
import scipy.ndimage as ndimage
import numpy as np
from torch.utils.data import Dataset
from related.emage.models.audio.modeling_emage_audio import (
  EmageVQModel,
  EmageVQVAEConv,
)


SPLIT_IDXS: dict[str, int] = {
  "train": 0,
  "val": 1,
  "test": 2,
  "additional": 3,
}


def align_inputs_repeat(
  inputs: np.ndarray, inputs_durations: np.ndarray, n_frames: int
) -> np.ndarray:
  inputs_starts = np.roll(np.cumsum(inputs_durations), 1)
  inputs_starts[0] = 0.0
  total_duration = inputs_durations.sum()
  target_times = np.linspace(
    0.0, total_duration, num=n_frames, dtype=np.float32
  )
  active_inputs = np.searchsorted(inputs_starts, target_times, side="right") - 1
  n_inputs = inputs.shape[0]
  active_inputs = np.clip(active_inputs, 0, n_inputs - 1)
  return inputs[active_inputs]


def resample_inputs_repeat(
  inputs: np.ndarray,
  inputs_durations: np.ndarray,
  targets: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  for part_targets in targets.values():
    n_targets = part_targets.shape[0]
    break
  return align_inputs_repeat(inputs, inputs_durations, n_targets), targets


def align_inputs_mean_sliding_window(
  window_size_frames: int,
) -> Callable[
  [
    np.ndarray,
    np.ndarray,
    int,
  ],
  np.ndarray,
]:
  def resample(
    inputs: np.ndarray, inputs_durations: np.ndarray, n_frames: int
  ) -> np.ndarray:
    repeated_inputs = align_inputs_repeat(inputs, inputs_durations, n_frames)
    mean_inputs = ndimage.uniform_filter1d(
      repeated_inputs, window_size_frames, axis=0, mode="nearest"
    )
    return mean_inputs

  return resample


def resample_inputs_mean_sliding_window(
  window_size_frames: int,
) -> Callable[
  [
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
  ],
  tuple[np.ndarray, dict[str, np.ndarray]],
]:
  def resample(
    inputs: np.ndarray,
    inputs_durations: np.ndarray,
    targets: dict[str, np.ndarray],
  ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    repeated_inputs, targets = resample_inputs_repeat(
      inputs, inputs_durations, targets
    )
    mean_inputs = ndimage.uniform_filter1d(
      repeated_inputs, window_size_frames, axis=0, mode="nearest"
    )
    return mean_inputs, targets

  return resample


def align_latents(
  motion_prior: EmageVQModel,
  align: Callable[
    [
      np.ndarray,
      np.ndarray,
      dict[str, np.ndarray],
    ],
    tuple[np.ndarray, dict[str, np.ndarray]],
  ],
) -> Callable[
  [
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
  ],
  tuple[np.ndarray, dict[str, np.ndarray]],
]:
  def inner(
    inputs: np.ndarray,
    inputs_durations: np.ndarray,
    idxs: dict[str, np.ndarray],
  ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    def convert(
      part_prior: EmageVQVAEConv, part_idxs: np.ndarray
    ) -> np.ndarray:
      idxs_pt = torch.from_numpy(part_idxs)
      latents = part_prior.quantizer.get_codebook_entry(
        idxs_pt.unsqueeze(0)
      ).squeeze(0)
      return latents.cpu().numpy()

    with torch.no_grad():
      latents = {
        "face": convert(motion_prior.vq_model_face, idxs["face"]),
        "upper": convert(motion_prior.vq_model_upper, idxs["upper"]),
        "hands": convert(motion_prior.vq_model_hands, idxs["hands"]),
        "lower": convert(motion_prior.vq_model_lower, idxs["lower"]),
      }
      return align(inputs, inputs_durations, latents)

  return inner


def squeeze_chunk(
  _chunk_size: int,
  inputs: np.ndarray,
  targets: dict[str, np.ndarray],
) -> Optional[dict[str, np.ndarray | dict[str, np.ndarray]]]:
  return {
    "inputs": inputs.squeeze(0),
    "targets": {
      part: part_targets.squeeze(0) for part, part_targets in targets.items()
    },
  }


def select_last_frame_as_target(
  chunk_size: int,
  inputs: np.ndarray,
  targets: dict[str, np.ndarray],
) -> Optional[dict[str, np.ndarray | dict[str, np.ndarray]]]:
  if chunk_size != inputs.shape[0]:
    return None
  return {
    "inputs": inputs,
    "targets": {
      part: part_targets[-1:].squeeze(0)
      for part, part_targets in targets.items()
    },
  }


def filter_short_chunks(
  chunk_size: int,
  inputs: np.ndarray,
  targets: dict[str, np.ndarray],
) -> Optional[dict[str, np.ndarray | dict[str, np.ndarray]]]:
  if chunk_size != inputs.shape[0]:
    return None
  return {"inputs": inputs, "targets": targets}


def get_file_id(file_path: str) -> str:
  name = os.path.basename(file_path)
  id, _ = os.path.splitext(name)
  return id


def dataset_files(data_dir: str) -> list[str]:
  return glob(os.path.join(data_dir, "**.npz"))


def sorted_file_ids(data_dir: str) -> list[str]:
  return sorted(list(get_file_id(f) for f in dataset_files(data_dir)))


def file_path(data_dir: str, file_id: str) -> str:
  return os.path.join(data_dir, f"{file_id}.npz")


def get_split_idxs(split: Optional[str] | list[str]) -> list[int]:
  if split:
    if type(split) is str:
      return [SPLIT_IDXS[split.lower()]]
    elif type(split) is list:
      return [SPLIT_IDXS[s.lower()] for s in split]
    else:
      raise Exception(f"unknown split type: {split}")
  else:
    return []


def get_data_file(data_dir: str, file_id: str) -> NpzFile:
  file_ids = {get_file_id(path): path for path in dataset_files(data_dir)}
  path = file_ids[file_id]
  return np.load(path)


def slice_chunk(
  file_inputs: np.ndarray,
  file_targets: dict[str, np.ndarray],
  chunk_offset: int,
  chunk_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  start = chunk_offset
  end = min(start + chunk_size, file_inputs.shape[0])
  chunk_inputs = file_inputs[start:end]
  chunk_targets = {
    part: part_targets[start:end] for part, part_targets in file_targets.items()
  }
  return chunk_inputs, chunk_targets


def to_torch(v):
  if type(v) is np.ndarray:
    return torch.from_numpy(v)
  elif type(v) is dict:
    return {key: to_torch(value) for key, value in v.items()}
  else:
    raise Exception(
      f"cannot convert value of type {type(v)} to PyTorch tensor: {v}"
    )


class GestureDataset(Dataset):
  def __init__(
    self,
    data_dir: str,
    align: Callable[
      [np.ndarray, np.ndarray, dict[str, np.ndarray]],
      tuple[np.ndarray, dict[str, np.ndarray]],
    ],
    filter_map_chunk: Callable[
      [int, np.ndarray, dict[str, np.ndarray]],
      Optional[dict[str, np.ndarray | dict[str, np.ndarray]]],
    ],
    chunk_size: int,
    chunk_stride: int,
    split: Optional[str] | list[str],
    shuffle_files: bool,
  ):
    """
    Initialize the dataset by scanning all files in data_dir, filtering by
    split, and precomputing chunk offsets.

    This dataset contains a LRU cache of size 1 to amortize alignment costs.
    This cache only functions if items are accessed in sequence. (When
    DataLoader(shuffle=False).) Initialize GestureDataset with
    shuffle_files=True to shuffle in a cache-friendly way.

    During initialization, each file is loaded and aligned using dummy
    inputs of size (N,), rather than full text embeddings of size (N, *embeds_size)
    to determine the number of output frames without incurring the cost of
    embedding interpolation.

    Args:
      data_dir: Directory containing .npz dataset files.
      align: Function mapping (inputs, durations, targets) -> (aligned_inputs, aligned_targets).
        Must preserve the time dimension as axis 0. Called
        twice per file: once with token arrays during init (for chunk layout), and
        once with full text embeddings during __getitem__ (for retrieval).
      chunk_size: Number of frames per chunk.
      chunk_stride: Step size between
      chunk start positions. Setting chunk_stride < chunk_size produces
        overlapping chunks.
      split: One or more split names ("train", "val",
        "test", "additional"), or None to include all files.
    """
    self.data_dir = data_dir
    self.align = align
    self.filter_map_chunk = filter_map_chunk
    self.chunk_size = chunk_size

    self.file_ids = sorted_file_ids(self.data_dir)
    if shuffle_files:
      random.shuffle(self.file_ids)
    self.cache = None

    split_idxs = get_split_idxs(split)
    chunks = []
    for file_id_idx, file_id in enumerate(self.file_ids):
      data = np.load(file_path(self.data_dir, file_id))
      if split_idxs and data["split"] not in split_idxs:
        continue
      inputs, targets = self.align(
        data["text_durations"],
        data["text_durations"],
        {
          "face": data["face_idx"],
          "upper": data["upper_idx"],
          "hands": data["hands_idx"],
          "lower": data["lower_idx"],
        },
      )
      n_inputs, n_targets = inputs.shape[0], targets["face"].shape[0]
      assert n_inputs == n_targets, f"{n_inputs} != {n_targets}"
      for chunk_offset in range(0, n_inputs, chunk_stride):
        chunk_inputs, chunk_targets = slice_chunk(
          inputs, targets, chunk_offset, self.chunk_size
        )
        if self.filter_map_chunk(self.chunk_size, chunk_inputs, chunk_targets):
          chunks.append((file_id_idx, chunk_offset))
    self.chunks = np.array(chunks)

  def __len__(self):
    return self.chunks.shape[0]

  def __getitem__(self, idx: int):  # ty: ignore[invalid-method-override]
    """
    Load and return one chunk of aligned, padded data.

    Returns a dict with keys:
      'inputs': float32 tensor of shape (chunk_size, embed_dim) 'targets': dict
      mapping body part name to int64 tensor of shape (chunk_size,) containing
      codebook indices 'attention_mask': int64 tensor of shape (chunk_size,),
      with 1 for real frames and 0 for padding
    """
    file_id_idx, chunk_offset = self.chunks[idx]
    if self.cache and file_id_idx == self.cache[0]:
      _, inputs, targets = self.cache
    else:
      file_id = self.file_ids[file_id_idx]
      data = np.load(file_path(self.data_dir, file_id))
      inputs, targets = self.align(
        data["text_embeds"],
        data["text_durations"],
        {
          "face": data["face_idx"],
          "upper": data["upper_idx"],
          "hands": data["hands_idx"],
          "lower": data["lower_idx"],
        },
      )
      self.cache = (file_id_idx, inputs, targets)
    chunk_inputs, chunk_targets = slice_chunk(
      inputs, targets, chunk_offset, self.chunk_size
    )
    # self.filter_map_chunk() cannot be None here, because invalid chunk offsets were
    # already filtered out in __init__().
    data = self.filter_map_chunk(self.chunk_size, chunk_inputs, chunk_targets)

    return to_torch(data)


TEST_SPLIT_FILES: list[str] = [
  "1_wayne_0_1_1",
  "1_wayne_0_2_2",
  "1_wayne_0_3_3",
  "1_wayne_0_4_4",
  "1_wayne_0_5_5",
  "1_wayne_0_6_6",
  "1_wayne_0_7_7",
  "1_wayne_0_8_8",
  "1_wayne_0_65_65",
  "1_wayne_0_73_73",
  "1_wayne_0_81_81",
  "1_wayne_0_87_87",
  "1_wayne_0_95_95",
  "1_wayne_0_103_103",
  "1_wayne_0_111_111",
  "2_scott_0_1_1",
  "2_scott_0_2_2",
  "2_scott_0_3_3",
  "2_scott_0_4_4",
  "2_scott_0_5_5",
  "2_scott_0_6_6",
  "2_scott_0_7_7",
  "2_scott_0_8_8",
  "2_scott_0_65_65",
  "2_scott_0_73_73",
  "2_scott_0_81_81",
  "2_scott_0_87_87",
  "2_scott_0_95_95",
  "2_scott_0_103_103",
  "2_scott_0_111_111",
  "3_solomon_0_1_1",
  "3_solomon_0_2_2",
  "3_solomon_0_3_3",
  "3_solomon_0_4_4",
  "3_solomon_0_5_5",
  "3_solomon_0_6_6",
  "3_solomon_0_7_7",
  "3_solomon_0_8_8",
  "3_solomon_0_65_65",
  "3_solomon_0_73_73",
  "3_solomon_0_81_81",
  "3_solomon_0_87_87",
  "3_solomon_0_95_95",
  "3_solomon_0_103_103",
  "3_solomon_0_111_111",
  "4_lawrence_0_1_1",
  "4_lawrence_0_2_2",
  "4_lawrence_0_3_3",
  "4_lawrence_0_4_4",
  "4_lawrence_0_5_5",
  "4_lawrence_0_6_6",
  "4_lawrence_0_7_7",
  "4_lawrence_0_8_8",
  "4_lawrence_0_65_65",
  "4_lawrence_0_73_73",
  "4_lawrence_0_81_81",
  "4_lawrence_0_87_87",
  "4_lawrence_0_95_95",
  "4_lawrence_0_103_103",
  "4_lawrence_0_111_111",
  "5_stewart_0_1_1",
  "5_stewart_0_2_2",
  "5_stewart_0_3_3",
  "5_stewart_0_4_4",
  "5_stewart_0_5_5",
  "5_stewart_0_6_6",
  "5_stewart_0_7_7",
  "5_stewart_0_8_8",
  "5_stewart_0_65_65",
  "5_stewart_0_73_73",
  "5_stewart_0_81_81",
  "5_stewart_0_87_87",
  "5_stewart_0_95_95",
  "5_stewart_0_103_103",
  "5_stewart_0_111_111",
  "6_carla_0_65_65",
  "6_carla_0_73_73",
  "6_carla_0_81_81",
  "6_carla_0_87_87",
  "6_carla_0_95_95",
  "6_carla_0_103_103",
  "6_carla_0_111_111",
  "7_sophie_0_1_1",
  "7_sophie_0_2_2",
  "7_sophie_0_3_3",
  "7_sophie_0_4_4",
  "7_sophie_0_5_5",
  "7_sophie_0_6_6",
  "7_sophie_0_7_7",
  "7_sophie_0_8_8",
  "7_sophie_0_65_65",
  "7_sophie_0_73_73",
  "7_sophie_0_81_81",
  "7_sophie_0_87_87",
  "7_sophie_0_95_95",
  "7_sophie_0_103_103",
  "7_sophie_0_111_111",
  "9_miranda_0_65_65",
  "9_miranda_0_73_73",
  "9_miranda_0_81_81",
  "9_miranda_0_87_87",
  "9_miranda_0_95_95",
  "9_miranda_0_103_103",
  "9_miranda_0_111_111",
  "10_kieks_0_1_1",
  "10_kieks_0_2_2",
  "10_kieks_0_3_3",
  "10_kieks_0_4_4",
  "10_kieks_0_5_5",
  "10_kieks_0_6_6",
  "10_kieks_0_7_7",
  "10_kieks_0_8_8",
  "10_kieks_0_65_65",
  "10_kieks_0_73_73",
  "10_kieks_0_81_81",
  "10_kieks_0_87_87",
  "10_kieks_0_95_95",
  "10_kieks_0_103_103",
  "10_kieks_0_111_111",
  "11_nidal_0_1_1",
  "11_nidal_0_2_2",
  "11_nidal_0_3_3",
  "11_nidal_0_4_4",
  "11_nidal_0_5_5",
  "11_nidal_0_6_6",
  "11_nidal_0_7_7",
  "11_nidal_0_8_8",
  "11_nidal_0_65_65",
  "11_nidal_0_73_73",
  "11_nidal_0_81_81",
  "11_nidal_0_87_87",
  "11_nidal_0_95_95",
  "11_nidal_0_103_103",
  "11_nidal_0_111_111",
  "12_zhao_0_1_1",
  "12_zhao_0_2_2",
  "12_zhao_0_65_65",
  "12_zhao_0_73_73",
  "12_zhao_0_81_81",
  "12_zhao_0_87_87",
  "12_zhao_0_95_95",
  "12_zhao_0_103_103",
  "12_zhao_0_111_111",
  "13_lu_0_1_1",
  "13_lu_0_2_2",
  "13_lu_0_65_65",
  "13_lu_0_73_73",
  "13_lu_0_81_81",
  "13_lu_0_87_87",
  "13_lu_0_95_95",
  "13_lu_0_103_103",
  "13_lu_0_111_111",
  "15_carlos_0_1_1",
  "15_carlos_0_2_2",
  "15_carlos_0_65_65",
  "15_carlos_0_73_73",
  "15_carlos_0_81_81",
  "15_carlos_0_87_87",
  "15_carlos_0_95_95",
  "15_carlos_0_103_103",
  "15_carlos_0_111_111",
  "16_jorge_0_1_1",
  "16_jorge_0_2_2",
  "16_jorge_0_65_65",
  "16_jorge_0_73_73",
  "16_jorge_0_81_81",
  "16_jorge_0_87_87",
  "16_jorge_0_95_95",
  "16_jorge_0_103_103",
  "16_jorge_0_111_111",
  "17_itoi_0_1_1",
  "17_itoi_0_2_2",
  "17_itoi_0_65_65",
  "17_itoi_0_73_73",
  "17_itoi_0_81_81",
  "17_itoi_0_87_87",
  "17_itoi_0_95_95",
  "17_itoi_0_103_103",
  "17_itoi_0_111_111",
  "18_daiki_0_1_1",
  "18_daiki_0_2_2",
  "18_daiki_0_65_65",
  "18_daiki_0_73_73",
  "18_daiki_0_81_81",
  "18_daiki_0_87_87",
  "18_daiki_0_95_95",
  "18_daiki_0_103_103",
  "18_daiki_0_111_111",
  "20_li_0_1_1",
  "20_li_0_2_2",
  "20_li_0_65_65",
  "20_li_0_73_73",
  "20_li_0_81_81",
  "20_li_0_87_87",
  "20_li_0_95_95",
  "20_li_0_103_103",
  "20_li_0_111_111",
  "21_ayana_0_65_65",
  "21_ayana_0_73_73",
  "21_ayana_0_81_81",
  "21_ayana_0_87_87",
  "21_ayana_0_95_95",
  "21_ayana_0_103_103",
  "21_ayana_0_111_111",
  "22_luqi_0_1_1",
  "22_luqi_0_2_2",
  "22_luqi_0_65_65",
  "22_luqi_0_73_73",
  "22_luqi_0_81_81",
  "22_luqi_0_87_87",
  "22_luqi_0_95_95",
  "22_luqi_0_103_103",
  "22_luqi_0_111_111",
  "23_hailing_0_1_1",
  "23_hailing_0_2_2",
  "23_hailing_0_65_65",
  "23_hailing_0_81_81",
  "23_hailing_0_87_87",
  "23_hailing_0_95_95",
  "23_hailing_0_103_103",
  "23_hailing_0_111_111",
  "24_kexin_0_1_1",
  "24_kexin_0_2_2",
  "24_kexin_0_65_65",
  "24_kexin_0_73_73",
  "24_kexin_0_81_81",
  "24_kexin_0_87_87",
  "24_kexin_0_95_95",
  "24_kexin_0_103_103",
  "24_kexin_0_111_111",
  "25_goto_0_2_2",
  "25_goto_0_65_65",
  "25_goto_0_73_73",
  "25_goto_0_81_81",
  "25_goto_0_87_87",
  "25_goto_0_95_95",
  "25_goto_0_103_103",
  "25_goto_0_111_111",
  "27_yingqing_0_1_1",
  "27_yingqing_0_2_2",
  "27_yingqing_0_65_65",
  "27_yingqing_0_73_73",
  "27_yingqing_0_81_81",
  "27_yingqing_0_87_87",
  "27_yingqing_0_95_95",
  "27_yingqing_0_103_103",
  "27_yingqing_0_111_111",
  "28_tiffnay_0_1_1",
  "28_tiffnay_0_2_2",
  "28_tiffnay_0_65_65",
  "28_tiffnay_0_73_73",
  "28_tiffnay_0_81_81",
  "28_tiffnay_0_87_87",
  "28_tiffnay_0_95_95",
  "28_tiffnay_0_103_103",
  "28_tiffnay_0_111_111",
  "30_katya_0_1_1",
  "30_katya_0_2_2",
  "30_katya_0_65_65",
  "30_katya_0_73_73",
  "30_katya_0_81_81",
  "30_katya_0_87_87",
  "30_katya_0_95_95",
  "30_katya_0_103_103",
  "30_katya_0_111_111",
]


def adapter_inputs(
  shuffle: bool,
  align: Callable[[np.ndarray, np.ndarray, int], np.ndarray] = align_inputs_repeat,
) -> Generator[tuple[str, np.ndarray], None, None]:
  files = TEST_SPLIT_FILES.copy()
  if shuffle:
    random.shuffle(files)
  for file_id in files:
    pack_path = os.path.join("data", "pack", f"{file_id}.npz")
    data = np.load(pack_path)
    n_frames = data["face_idx"].shape[0]
    embeds = align(
      data["text_embeds"],
      data["text_durations"],
      n_frames,
    )
    yield (file_id, embeds)
