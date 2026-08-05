# Adapted from https://github.com/GENEALeaderboard/objective_metric/blob/02ba223acd8df61c33ab0acacab1331d6dbbe0cd/exp_eval_metrics.py
import math

import librosa
import numpy as np
import torch
from scipy import linalg
from scipy.signal import argrelextrema

from emage_evaltools.motion_encoder import VAESKConv
from scripts import util


class L1Div(object):
  def __init__(self):
    self.counter = 0
    self.sum = 0

  def compute(self, results):
    self.counter += results.shape[0]
    mean = np.mean(results, axis=0)
    sum_l1 = np.sum(np.abs(results - mean), axis=None)
    self.sum += sum_l1

  def avg(self):
    if self.counter == 0:
      return 0
    return self.sum / self.counter


class Srgr(object):
  def __init__(self, threshold=0.1, joints=55, joint_dim=3, fps=30):
    self.threshold = threshold
    self.pose_dimes = joints
    self.joint_dim = joint_dim
    self.fps = fps
    self.counter = 0
    self.sum = 0

  def parse_semantic(self, tsv_data, n):
    fps = self.fps

    col_map = {}
    for key in ("start", "end", "weight"):
      col_map[key] = tsv_data[key].astype(float).values
    starts, ends, weights = (
      col_map["start"],
      col_map["end"],
      col_map["weight"],
    )

    semantic = np.zeros(n, dtype=np.float32)

    for s, e, w in zip(starts, ends, weights):
      if e <= s:
        continue
      s_f = int(np.floor(s * fps))
      e_f = int(np.ceil(e * fps))
      semantic[s_f:e_f] = np.maximum(semantic[s_f:e_f], w)

    return semantic

  def run(self, results, targets, semantic_raw_data=None, verbose=False):
    if semantic_raw_data is None:
      semantic = np.ones(results.shape[0])
      avg_weight = 1.0
    else:
      semantic = self.parse_semantic(semantic_raw_data, targets.shape[0])
      avg_weight = 0.1672  # scale range to [0, 1] when all success
    results = results.reshape(-1, self.pose_dimes, self.joint_dim)
    targets = targets.reshape(-1, self.pose_dimes, self.joint_dim)
    semantic = semantic.reshape(-1)
    diff = np.linalg.norm(results - targets, axis=2)  # T, J
    if verbose:
      print(diff)
    success = np.where(diff < self.threshold, 1.0, 0.0)
    for i in range(success.shape[0]):
      success[i, :] *= semantic[i] * (1 / avg_weight)
    rate = np.sum(success) / (success.shape[0] * success.shape[1])
    self.counter += success.shape[0]
    self.sum += rate * success.shape[0]
    return rate

  def avg(self):
    return self.sum / self.counter


class Bc(object):
  def __init__(
    self,
    mmae_path: str,
    sigma=0.3,
    order=7,
    upper_body=[3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
  ):
    self.sigma = sigma
    self.order = order
    self.upper_body = upper_body
    self.pose_data = []
    self.mmae = np.load(mmae_path)
    self.threshold = 0.10
    self.counter = 0
    self.sum_a2m = 0
    self.sum_m2a = 0

  def load_audio(
    self, wave, t_start=None, t_end=None, without_file=False, sr_audio=16000
  ):
    hop_length = 512
    if without_file:
      y = wave
    else:
      y, sr = librosa.load(wave, sr=sr_audio)

    short_y = y[t_start:t_end] if t_start is not None else y
    short_y = short_y.astype(np.float32)
    onset_t = librosa.onset.onset_detect(
      y=short_y, sr=sr_audio, hop_length=hop_length, units="time"
    )
    return onset_t

  def load_motion(self, pose, t_start, t_end, pose_fps, without_file=False):
    data_each_file = []
    if without_file:
      data_each_file = pose
    else:
      with open(pose, "r") as f:
        for i, line_data in enumerate(f.readlines()):
          if i < 432:
            continue
          line_data_np = np.fromstring(line_data, sep=" ")
          if pose_fps == 15 and i % 2 == 0:
            continue
          data_each_file.append(
            np.concatenate([line_data_np[30:39], line_data_np[112:121]], 0)
          )
      data_each_file = np.array(data_each_file)  # T*165
    joints = data_each_file.transpose(1, 0)
    dt = 1 / pose_fps
    init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
    middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
    final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
    vel = (
      np.concatenate([init_vel, middle_vel, final_vel], 1)
      .transpose(1, 0)
      .reshape(data_each_file.shape[0], -1, 3)
    )

    if self.mmae is not None:
      vel = np.linalg.norm(vel, axis=2) / self.mmae
    else:
      print("Warning: mmae is not provided, using max value of vel as mmae")
      self.mmae = np.linalg.norm(vel, axis=2).max()
      vel = np.linalg.norm(vel, axis=2) / self.mmae

    beat_vel_all = []
    for i in range(vel.shape[1]):
      vel_mask = np.where(vel[:, i] > self.threshold)
      beat_vel = argrelextrema(vel[t_start:t_end, i], np.less, order=self.order)
      beat_vel_list = [j for j in beat_vel[0] if j in vel_mask[0]]
      beat_vel_all.append(np.array(beat_vel_list))
    return beat_vel_all

  @staticmethod
  def motion_frames2time(vel, offset, pose_fps):
    return vel / pose_fps + offset

  @staticmethod
  def GAHR(a, b, sigma):
    dis_all_b2a = 0
    for b_each in b:
      l2_min = min(abs(a_each - b_each) for a_each in a)
      dis_all_b2a += math.exp(-(l2_min**2) / (2 * sigma**2))
    return dis_all_b2a / len(b)

  def compute(self, onset_bt_rms, beat_vel, length=1, pose_fps=30):
    avg_dis_all_a2m_list = []
    avg_dis_all_m2a_list = []
    for its, beat_vel_each in enumerate(beat_vel):
      if its not in self.upper_body:
        continue
      if beat_vel_each.size == 0:
        avg_dis_all_a2m_list.append(0)
        avg_dis_all_m2a_list.append(0)
        continue
      pose_bt = self.motion_frames2time(beat_vel_each, 0, pose_fps)
      avg_dis_all_a2m_list.append(self.GAHR(pose_bt, onset_bt_rms, self.sigma))
      avg_dis_all_m2a_list.append(self.GAHR(onset_bt_rms, pose_bt, self.sigma))
    self.sum_a2m += (sum(avg_dis_all_a2m_list) / len(self.upper_body)) * length
    self.sum_m2a += (sum(avg_dis_all_m2a_list) / len(self.upper_body)) * length
    self.counter += length

  def avg(self):
    return {
      "a2m": self.sum_a2m / self.counter,
      "m2a": self.sum_m2a / self.counter,
    }


class FgdEvalModelArgs(object):
  def __init__(self):
    self.vae_length = 240
    self.vae_test_dim = 330
    self.vae_test_len = 32
    self.vae_layer = 4
    self.vae_test_stride = 20
    self.vae_grow = [1, 1, 2, 1]
    self.variational = False


class Fgd(object):
  def __init__(
    self,
    eval_model_path: str,
    smplx_dir: str,
    device=util.DEVICE,
  ):
    smplx_dir += "/"
    self.eval_model = VAESKConv(
      FgdEvalModelArgs(), smplx_dir
    )  # Assumes LocalEncoder is defined elsewhere
    old_stat = torch.load(eval_model_path)["model_state"]
    new_stat = {}
    for k, v in old_stat.items():
      new_key = k.replace("module.", "") if "module." in k else k
      new_stat[new_key] = v
    self.eval_model.load_state_dict(new_stat)
    self.eval_model.eval()
    self.eval_model.to(device)

    self.pred_features = []
    self.target_features = []
    self.device = device

  def get_feature(self, data):
    assert len(data.shape) == 3
    if data.shape[1] % 32 != 0:
      drop_len = data.shape[1] % 32
      data = data[:, :-drop_len]
    with torch.no_grad():
      feature = self.eval_model.map2latent(data.to(self.device)).cpu().numpy()
    return feature

  def update_pred(self, pred):
    self.pred_features.append(self.get_feature(pred))

  def update_target(self, target):
    self.target_features.append(self.get_feature(target))

  def compute(self):
    pred_features = np.concatenate(
      [x.reshape(-1, x.shape[-1]) for x in self.pred_features], axis=0
    )
    target_features = np.concatenate(
      [x.reshape(-1, x.shape[-1]) for x in self.target_features], axis=0
    )
    return self.frechet_distance(pred_features, target_features)

  @staticmethod
  def frechet_distance(samples_A, samples_B, eps=1e-6):
    mu1 = np.mean(samples_A, axis=0)
    sigma1 = np.cov(samples_A, rowvar=False)
    mu2 = np.mean(samples_B, axis=0)
    sigma2 = np.cov(samples_B, rowvar=False)
    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
      covmean = covmean.real
    return (
      diff.dot(diff)
      + np.trace(sigma1)
      + np.trace(sigma2)
      - 2 * np.trace(covmean)
    )
