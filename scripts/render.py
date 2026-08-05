# Visualization code adapted from https://github.com/PantoMatrix/PantoMatrix/blob/c7356f35f8e39e469e510ccd1bf37e44adf8ec0e/emage_utils/fast_render.py
import subprocess

import numpy as np
import pyrender
import torch
import trimesh
from tqdm import tqdm

from scripts import util

VIDEO_WIDTH = 480
VIDEO_HEIGHT = 720


_DEFAULT_TRANS = None


def default_trans(n: int) -> np.ndarray:
  global _DEFAULT_TRANS
  if _DEFAULT_TRANS is None:
    smplx = util.load_smplx_model()
    smplx.eval()
    with torch.no_grad():
      joints = smplx(
        betas=torch.zeros(1, 300),
        transl=torch.zeros(1, 3),
        expression=torch.zeros(1, 100),
        jaw_pose=torch.zeros(1, 3),
        global_orient=torch.zeros(1, 3),
        body_pose=torch.zeros(1, 63),
        left_hand_pose=torch.zeros(1, 45),
        right_hand_pose=torch.zeros(1, 45),
        leye_pose=torch.zeros(1, 3),
        reye_pose=torch.zeros(1, 3),
        return_joints=True,
      )["joints"].numpy()
    _DEFAULT_TRANS = -(joints[0, 10, :] + joints[0, 11, :]) / 2
  return np.tile(_DEFAULT_TRANS, (n, 1))


def deg_to_rad(degrees: float) -> float:
  return degrees * np.pi / 180


def create_pose_camera(angle_deg: float) -> np.ndarray:
  angle_rad = deg_to_rad(angle_deg)
  return np.array(
    [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, np.cos(angle_rad), -np.sin(angle_rad), 1.0],
      [0.0, np.sin(angle_rad), np.cos(angle_rad), 5.0],
      [0.0, 0.0, 0.0, 1.0],
    ]
  )


def create_pose_light(angle_deg: float) -> np.ndarray:
  angle_rad = deg_to_rad(angle_deg)
  return np.array(
    [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, np.cos(angle_rad), -np.sin(angle_rad), 0.0],
      [0.0, np.sin(angle_rad), np.cos(angle_rad), 3.0],
      [0.0, 0.0, 0.0, 1.0],
    ]
  )


def create_scene_with_mesh(
  vertices: np.ndarray,
  faces: np.ndarray,
  vertex_colors: list[int],
  pose_camera: np.ndarray,
  pose_light: np.ndarray,
) -> pyrender.Scene:
  trimesh_mesh = trimesh.Trimesh(
    vertices=vertices, faces=faces, vertex_colors=vertex_colors
  )
  mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=True)
  scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
  scene.add(mesh)
  camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
  scene.add(camera, pose=pose_camera)
  light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
  scene.add(light, pose=pose_light)
  return scene


def gesture_vertices(
  poses: np.ndarray,
  expressions: np.ndarray,
) -> np.ndarray:
  model = util.load_smplx_model()
  n = poses.shape[0]
  poses_pt = torch.from_numpy(poses)
  return (
    model(
      betas=torch.zeros(n, 300),
      transl=torch.from_numpy(default_trans(n)),
      expression=torch.from_numpy(expressions[:n]).float(),
      jaw_pose=poses_pt[:, 66:69],
      global_orient=poses_pt[:, :3],
      body_pose=poses_pt[:, 3 : 21 * 3 + 3],
      left_hand_pose=poses_pt[:, 25 * 3 : 40 * 3],
      right_hand_pose=poses_pt[:, 40 * 3 : 55 * 3],
      leye_pose=poses_pt[:, 69:72],
      reye_pose=poses_pt[:, 72:75],
      return_verts=True,
    )["vertices"]
    .cpu()
    .numpy()
  )


def render_vertices(faces: np.ndarray, vertices: np.ndarray):
  renderer = pyrender.OffscreenRenderer(VIDEO_WIDTH, VIDEO_HEIGHT)
  pose_camera = create_pose_camera(angle_deg=-2)
  pose_light = create_pose_light(angle_deg=-30)
  vertex_colors = [220, 220, 220, 255]
  for vert in vertices:
    scene = create_scene_with_mesh(
      vert, faces, vertex_colors, pose_camera, pose_light
    )
    frame_buf, _ = renderer.render(scene)
    yield frame_buf
  renderer.delete()


def format_timestamp(t: float) -> str:
  hs = int(t // 3600)
  mins = int((t % 3600) // 60)
  secs = int(t % 60)
  ms = int((t % 1) * 1000)
  return f"{hs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def subs(
  text: str, token_offsets: np.ndarray, token_durations: np.ndarray
) -> str:
  CONTEXT_DURATION = 5

  end_offsets = np.roll(token_offsets, -1)
  end_offsets[-1] = len(text)
  start_times = np.roll(np.cumsum(token_durations), 1)
  start_times[0] = 0.0
  end_times = start_times + token_durations

  chunk_cats = end_times // CONTEXT_DURATION
  chunk_starts_mask = chunk_cats != np.roll(chunk_cats, 1)
  chunk_idxs = np.arange(end_times.shape[0])
  chunk_starts_idxs = chunk_idxs[chunk_starts_mask]
  chunk_ends_idxs = np.roll(chunk_starts_idxs, -1)
  chunk_ends_idxs[-1] = end_times.shape[0]

  lines = []
  subtitle_num = 1
  for chunk_start_idx, chunk_end_idx in zip(chunk_starts_idxs, chunk_ends_idxs):
    chunk_start_offsets = token_offsets[chunk_start_idx:chunk_end_idx]
    chunk_end_offsets = end_offsets[chunk_start_idx:chunk_end_idx]
    chunk_start_times = start_times[chunk_start_idx:chunk_end_idx]
    chunk_end_times = end_times[chunk_start_idx:chunk_end_idx]
    for phrase_idx, (token_start_time, token_end_time) in enumerate(
      zip(chunk_start_times, chunk_end_times)
    ):
      phrase_parts = []
      for token_idx, (offset_start, offset_end) in enumerate(
        zip(chunk_start_offsets, chunk_end_offsets)
      ):
        token_text = text[int(offset_start) : int(offset_end)]
        phrase_parts.append(
          f"<b>{token_text}</b>" if phrase_idx == token_idx else token_text
        )
      phrase = "".join(phrase_parts)
      lines.append(f"{subtitle_num}")
      lines.append(
        f"{format_timestamp(token_start_time)} --> {format_timestamp(token_end_time)}"
      )
      lines.append(phrase)
      lines.append("")
      subtitle_num += 1
  return "\n".join(lines)


def render_to_file(
  dest: str,
  poses: np.ndarray,
  expressions: np.ndarray,
  show_progress: bool = True,
):
  vertices = gesture_vertices(poses, expressions)
  faces = util.load_smplx_faces()
  frames = render_vertices(faces, vertices)
  ffmpeg = subprocess.Popen(
    [
      "ffmpeg",
      "-loglevel",
      "quiet",
      "-y",
      "-f",
      "rawvideo",
      "-pix_fmt",
      "rgb24",
      "-s",
      f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
      "-framerate",
      "30",
      "-i",
      "-",
      "-codec:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      dest,
    ],
    stdin=subprocess.PIPE,
  )
  progress = (
    tqdm(frames, desc=f"render {dest}", total=vertices.shape[0], smoothing=0)
    if show_progress
    else frames
  )
  if ffmpeg.stdin:
    for frame in progress:
      try:
        ffmpeg.stdin.write(frame.tobytes())
      except Exception as e:
        ffmpeg.kill()
        raise e
    ffmpeg.stdin.close()
  ffmpeg.wait()
