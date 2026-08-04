import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding


class MlpContextlessIndexAdapter(nn.Module):
  def __init__(self, hidden_dim: int):
    super().__init__()
    self.stack = nn.Sequential(
      nn.Linear(3072, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.SiLU(),
    )
    self.face_head = nn.Linear(hidden_dim, 256)
    self.upper_head = nn.Linear(hidden_dim, 256)
    self.hands_head = nn.Linear(hidden_dim, 256)
    self.lower_head = nn.Linear(hidden_dim, 256)

  def forward(self, inputs: torch.Tensor):
    features = self.stack(inputs)
    return {
      "face": self.face_head(features),
      "upper": self.upper_head(features),
      "hands": self.hands_head(features),
      "lower": self.lower_head(features),
    }


class FfnContextlessIndexAdapter(nn.Module):
  def __init__(self):
    super().__init__()
    self.gate = nn.Linear(3072, 8192, bias=False)
    self.up = nn.Linear(3072, 8192, bias=False)
    self.down = nn.Linear(8192, 3072, bias=False)
    self.face_head = nn.Linear(3072, 256)
    self.upper_head = nn.Linear(3072, 256)
    self.hands_head = nn.Linear(3072, 256)
    self.lower_head = nn.Linear(3072, 256)

  def forward(self, inputs: torch.Tensor):
    up = self.up(inputs)
    gate = F.silu(self.gate(inputs))
    down = self.down(up * gate)
    return {
      "face": self.face_head(down),
      "upper": self.upper_head(down),
      "hands": self.hands_head(down),
      "lower": self.lower_head(down),
    }


class FfnContextIndexAdapter(nn.Module):
  def __init__(self, context_size: int):
    super().__init__()
    input_dim = context_size * 3072
    self.gate = nn.Linear(input_dim, 8192, bias=False)
    self.up = nn.Linear(input_dim, 8192, bias=False)
    self.down = nn.Linear(8192, 3072, bias=False)
    self.face_head = nn.Linear(3072, 256)
    self.upper_head = nn.Linear(3072, 256)
    self.hands_head = nn.Linear(3072, 256)
    self.lower_head = nn.Linear(3072, 256)

  def forward(self, inputs: torch.Tensor):
    inputs = inputs.flatten(start_dim=1)
    up = self.up(inputs)
    gate = F.silu(self.gate(inputs))
    down = self.down(up * gate)
    return {
      "face": self.face_head(down),
      "upper": self.upper_head(down),
      "hands": self.hands_head(down),
      "lower": self.lower_head(down),
    }


class ProjectDownContextIndexAdapter(nn.Module):
  def __init__(self, context_size: int, project_dims: int, hidden_dims: int):
    super().__init__()
    self.input_proj = FfnLayer(3072, project_dims, hidden_dims)
    self.context_proj = FfnLayer(
      context_size * hidden_dims, project_dims, hidden_dims
    )
    self.face_head = nn.Linear(hidden_dims, 256)
    self.upper_head = nn.Linear(hidden_dims, 256)
    self.hands_head = nn.Linear(hidden_dims, 256)
    self.lower_head = nn.Linear(hidden_dims, 256)

  def forward(self, inputs: torch.Tensor):
    input_proj = self.input_proj(inputs)
    input_proj = self.context_proj(input_proj.flatten(start_dim=1))
    return {
      "face": self.face_head(input_proj),
      "upper": self.upper_head(input_proj),
      "hands": self.hands_head(input_proj),
      "lower": self.lower_head(input_proj),
    }


class FfnLayer(nn.Module):
  def __init__(self, in_features: int, up_features: int, out_features: int):
    super().__init__()
    self.gate = nn.Linear(in_features, up_features, bias=False)
    self.up = nn.Linear(in_features, up_features, bias=False)
    self.down = nn.Linear(up_features, out_features, bias=False)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    up = self.up(inputs)
    gate = F.silu(self.gate(inputs))
    return self.down(up * gate)


def ceil_to_multiple(n: int | float, multiple: int) -> int:
  return int(math.ceil(n / multiple)) * multiple


class TransformerDecoderLayer(nn.Module):
  def __init__(self, context_size: int, model_dim: int, n_heads: int):
    super().__init__()
    self.inp_norm = nn.RMSNorm(model_dim)
    self.attn = nn.MultiheadAttention(
      model_dim, n_heads, bias=False, batch_first=True
    )
    self.ffn_norm = nn.RMSNorm(model_dim)
    self.ffn = FfnLayer(
      model_dim, ceil_to_multiple(model_dim * 8 / 3, 256), model_dim
    )
    self.register_buffer(
      "causal_mask",
      nn.Transformer.generate_square_subsequent_mask(context_size),
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    normed = self.inp_norm(inputs)
    attended, _ = self.attn(
      normed,
      normed,
      normed,
      attn_mask=self.causal_mask,
    )
    inputs = inputs + attended
    return inputs + self.ffn(self.ffn_norm(inputs))


class TransformerDecoderBase(nn.Module):
  def __init__(
    self, context_size: int, n_layers: int, model_dim: int, n_heads: int
  ):
    super().__init__()
    self.proj_in = nn.Linear(3072, model_dim, bias=False)
    self.layers = nn.ModuleList(
      [
        TransformerDecoderLayer(context_size, model_dim, n_heads)
        for _ in range(n_layers)
      ]
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    hidden_states = self.proj_in(inputs)
    for layer in self.layers:
      hidden_states = layer(hidden_states)
    return hidden_states


class TransformerLastIndexAdapter(nn.Module):
  def __init__(
    self, context_size: int, n_layers: int, model_dim: int, n_heads: int
  ):
    super().__init__()
    self.base = TransformerDecoderBase(
      context_size, n_layers, model_dim, n_heads
    )
    self.norm = nn.RMSNorm(model_dim)
    self.face_head = nn.Linear(model_dim, 256)
    self.upper_head = nn.Linear(model_dim, 256)
    self.hands_head = nn.Linear(model_dim, 256)
    self.lower_head = nn.Linear(model_dim, 256)

  def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    hidden_states = self.base(inputs)[:, -1, :]
    hidden_states = self.norm(hidden_states)
    return {
      "face": self.face_head(hidden_states),
      "upper": self.upper_head(hidden_states),
      "hands": self.hands_head(hidden_states),
      "lower": self.lower_head(hidden_states),
    }


class TransformerSequenceIndexAdapter(nn.Module):
  def __init__(
    self, context_size: int, n_layers: int, model_dim: int, n_heads: int
  ):
    super().__init__()
    self.base = TransformerDecoderBase(
      context_size, n_layers, model_dim, n_heads
    )
    self.norm = nn.RMSNorm(model_dim)
    self.face_head = nn.Linear(model_dim, 256)
    self.upper_head = nn.Linear(model_dim, 256)
    self.hands_head = nn.Linear(model_dim, 256)
    self.lower_head = nn.Linear(model_dim, 256)

  def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    hidden_states = self.base(inputs)
    hidden_states = self.norm(hidden_states)
    return {
      "face": self.face_head(hidden_states),
      "upper": self.upper_head(hidden_states),
      "hands": self.hands_head(hidden_states),
      "lower": self.lower_head(hidden_states),
    }


class TransformerSequenceNoBiasAdapter(nn.Module):
  def __init__(
    self, context_size: int, n_layers: int, model_dim: int, n_heads: int
  ):
    super().__init__()
    self.base = TransformerDecoderBase(
      context_size, n_layers, model_dim, n_heads
    )
    self.norm = nn.RMSNorm(model_dim)
    self.face_head = nn.Linear(model_dim, 256, bias=False)
    self.upper_head = nn.Linear(model_dim, 256, bias=False)
    self.hands_head = nn.Linear(model_dim, 256, bias=False)
    self.lower_head = nn.Linear(model_dim, 256, bias=False)

  def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    hidden_states = self.base(inputs)
    hidden_states = self.norm(hidden_states)
    return {
      "face": self.face_head(hidden_states),
      "upper": self.upper_head(hidden_states),
      "hands": self.hands_head(hidden_states),
      "lower": self.lower_head(hidden_states),
    }


class TransformerSequenceNoBiasCustomPartsAdapter(nn.Module):
  def __init__(
    self,
    parts: list[str],
    context_size: int,
    n_layers: int,
    model_dim: int,
    n_heads: int,
  ):
    super().__init__()
    self.base = TransformerDecoderBase(
      context_size, n_layers, model_dim, n_heads
    )
    self.norm = nn.RMSNorm(model_dim)
    self.heads = nn.ModuleDict(
      {part: nn.Linear(model_dim, 256, bias=False) for part in parts}
    )

  def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    hidden_states = self.base(inputs)
    hidden_states = self.norm(hidden_states)
    return {
      part: part_head(hidden_states) for part, part_head in self.heads.items()
    }


class TransformerEncoderLayer(nn.Module):
  def __init__(self, model_dim: int, n_heads: int):
    super().__init__()
    self.inp_norm = nn.RMSNorm(model_dim)
    self.attn = nn.MultiheadAttention(
      model_dim, n_heads, bias=False, batch_first=True
    )
    self.ffn_norm = nn.RMSNorm(model_dim)
    self.ffn = FfnLayer(
      model_dim, ceil_to_multiple(model_dim * 8 / 3, 256), model_dim
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    normed = self.inp_norm(inputs)
    attended, _ = self.attn(normed, normed, normed)
    inputs = inputs + attended
    return inputs + self.ffn(self.ffn_norm(inputs))


class TransformerEncoderBase(nn.Module):
  def __init__(self, n_layers: int, model_dim: int, n_heads: int):
    super().__init__()
    self.proj_in = nn.Linear(3072, model_dim, bias=False)
    self.layers = nn.ModuleList(
      [TransformerEncoderLayer(model_dim, n_heads) for _ in range(n_layers)]
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    hidden_states = self.proj_in(inputs)
    for layer in self.layers:
      hidden_states = layer(hidden_states)
    return hidden_states


class TransformerEncoderSequenceNoBiasAdapter(nn.Module):
  def __init__(self, n_layers: int, model_dim: int, n_heads: int):
    super().__init__()
    self.base = TransformerEncoderBase(n_layers, model_dim, n_heads)
    self.norm = nn.RMSNorm(model_dim)
    self.face_head = nn.Linear(model_dim, 256, bias=False)
    self.upper_head = nn.Linear(model_dim, 256, bias=False)
    self.hands_head = nn.Linear(model_dim, 256, bias=False)
    self.lower_head = nn.Linear(model_dim, 256, bias=False)

  def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    hidden_states = self.base(inputs)
    hidden_states = self.norm(hidden_states)
    return {
      "face": self.face_head(hidden_states),
      "upper": self.upper_head(hidden_states),
      "hands": self.hands_head(hidden_states),
      "lower": self.lower_head(hidden_states),
    }


class TransformerRopeDecoderLayer(nn.Module):
  def __init__(self, context_size: int, model_dim: int, n_heads: int):
    super().__init__()
    self.inp_norm = nn.RMSNorm(model_dim)
    self.positional_embeds = RotaryEmbedding(dim=model_dim // 2)
    self.attn = nn.MultiheadAttention(
      model_dim, n_heads, bias=False, batch_first=True
    )
    self.ffn_norm = nn.RMSNorm(model_dim)
    self.ffn = FfnLayer(
      model_dim, ceil_to_multiple(model_dim * 8 / 3, 256), model_dim
    )
    self.register_buffer(
      "causal_mask",
      nn.Transformer.generate_square_subsequent_mask(context_size),
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    normed = self.inp_norm(inputs)
    positional = self.positional_embeds.rotate_queries_or_keys(normed)
    attended, _ = self.attn(
      positional,
      positional,
      positional,
      attn_mask=self.causal_mask,
    )
    inputs = inputs + attended
    return inputs + self.ffn(self.ffn_norm(inputs))


class TransformerRopeDecoderBase(nn.Module):
  def __init__(
    self, context_size: int, n_layers: int, model_dim: int, n_heads: int
  ):
    super().__init__()
    self.proj_in = nn.Linear(3072, model_dim, bias=False)
    self.layers = nn.ModuleList(
      [
        TransformerRopeDecoderLayer(context_size, model_dim, n_heads)
        for _ in range(n_layers)
      ]
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    hidden_states = self.proj_in(inputs)
    for layer in self.layers:
      hidden_states = layer(hidden_states)
    return hidden_states


class TransformerRopeSequenceNoBiasAdapter(nn.Module):
  def __init__(
    self, context_size: int, n_layers: int, model_dim: int, n_heads: int
  ):
    super().__init__()
    self.base = TransformerRopeDecoderBase(
      context_size, n_layers, model_dim, n_heads
    )
    self.norm = nn.RMSNorm(model_dim)
    self.face_head = nn.Linear(model_dim, 256, bias=False)
    self.upper_head = nn.Linear(model_dim, 256, bias=False)
    self.hands_head = nn.Linear(model_dim, 256, bias=False)
    self.lower_head = nn.Linear(model_dim, 256, bias=False)

  def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    hidden_states = self.base(inputs)
    hidden_states = self.norm(hidden_states)
    return {
      "face": self.face_head(hidden_states),
      "upper": self.upper_head(hidden_states),
      "hands": self.hands_head(hidden_states),
      "lower": self.lower_head(hidden_states),
    }
