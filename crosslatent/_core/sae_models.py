"""
Standardized Sparse Autoencoder (SAE) model definitions.

Derived from InterPLM / dictionary_learning SAE implementations.
Provides a unified base class, consistent ``from_pretrained`` loading,
and a registry for constructing models by name.

Original reference:
https://github.com/saprmarks/dictionary_learning
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Optional, Sequence, Type, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = [
    "Dictionary",
    "ReLUSAE",
    "ReLUSAE_Tied",
    "TopKSAE",
    "JumpReLUSAE",
    "BatchTopKSAE",
    "IdentityDict",
    "SAE_REGISTRY",
    "build_sae",
    "load_sae_from_checkpoint",
    "set_decoder_norm_to_unit_norm",
    "remove_gradient_parallel_to_decoder_directions",
]

SAEType = Literal[
    "relu",
    "relu_tied",
    "topk",
    "jump_relu",
    "batch_topk",
    "identity",
]

Tensor = torch.Tensor
DeviceLike = Optional[Union[str, torch.device]]


def _resolve_device(device: DeviceLike = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_checkpoint(path: str, map_location: DeviceLike = None) -> dict[str, Tensor]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_normalize_flag(state_dict: dict[str, Tensor]) -> bool:
    value = state_dict.get("normalize_to_sqrt_d", torch.tensor(False))
    return bool(value.item())


def _infer_dims(state_dict: dict[str, Tensor]) -> tuple[int, int]:
    dict_size, activation_dim = state_dict["encoder.weight"].shape
    return activation_dim, dict_size


class Dictionary(ABC, nn.Module):
    """Base class for sparse autoencoder dictionaries."""

    dict_size: int
    activation_dim: int

    def __init__(self, normalize_to_sqrt_d: bool = False) -> None:
        super().__init__()
        self.normalize_to_sqrt_d = normalize_to_sqrt_d

    @property
    def has_normalization_factors(self) -> bool:
        if not hasattr(self, "activation_rescale_factor"):
            return False
        return not torch.allclose(
            self.activation_rescale_factor,
            torch.ones_like(self.activation_rescale_factor),
        )

    def _unnormalize_output(
        self, x_normalized: Tensor, original_norms: Optional[Tensor]
    ) -> Tensor:
        if not self.normalize_to_sqrt_d or original_norms is None:
            return x_normalized
        d = x_normalized.shape[-1]
        sqrt_d = torch.sqrt(
            torch.tensor(d, dtype=x_normalized.dtype, device=x_normalized.device)
        )
        return x_normalized * original_norms / sqrt_d

    def _normalize_input_and_get_norms(
        self, x: Tensor
    ) -> tuple[Tensor, Optional[Tensor]]:
        if not self.normalize_to_sqrt_d:
            return x, None

        original_norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        d = x.shape[-1]
        sqrt_d = torch.sqrt(torch.tensor(d, dtype=x.dtype, device=x.device))
        normalized = F.normalize(x, p=2, dim=-1) * sqrt_d
        return normalized, original_norms

    @abstractmethod
    def encode(self, x: Tensor, normalize_features: bool = False) -> Tensor:
        """Encode activations into sparse feature space."""

    @abstractmethod
    def decode(self, features: Tensor) -> Tensor:
        """Decode sparse features back to activation space."""

    @abstractmethod
    def encode_feat_subset(
        self,
        x: Tensor,
        feat_list: Sequence[int],
        normalize_features: bool = False,
    ) -> Tensor:
        """Encode using only a subset of dictionary features."""

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: DeviceLike = None,
        **kwargs: Any,
    ) -> "Dictionary":
        raise NotImplementedError(
            f"{cls.__name__}.from_pretrained is not implemented."
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        norm_factor_key = prefix + "activation_rescale_factor"
        if norm_factor_key in missing_keys:
            missing_keys.remove(norm_factor_key)

        norm_key = prefix + "normalize_to_sqrt_d"
        if norm_key in state_dict:
            self.normalize_to_sqrt_d = bool(state_dict[norm_key].item())
            if norm_key in unexpected_keys:
                unexpected_keys.remove(norm_key)

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "normalize_to_sqrt_d"] = torch.tensor(
            self.normalize_to_sqrt_d, dtype=torch.bool
        )

    def _make_contiguous(self) -> None:
        for param in self.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()


class ReLUSAE(Dictionary):
    """
    ReLU SAE with separate pre-encoding bias and untied encoder/decoder weights.

    - Pre-encoding bias applied before encoder
    - Decoder has no bias
    - Independent encoder/decoder weights
    - Decoder columns are unit-normed
    """

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        normalize_to_sqrt_d: bool = False,
    ) -> None:
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        self.bias = nn.Parameter(torch.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        init.zeros_(self.encoder.bias)

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = torch.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

        self.register_buffer("activation_rescale_factor", torch.ones(dict_size))
        self._make_contiguous()

    def encode(self, x: Tensor, normalize_features: bool = False) -> Tensor:
        features = F.relu(self.encoder(x - self.bias))
        if normalize_features:
            features = features / self.activation_rescale_factor
        return features

    def decode(self, features: Tensor) -> Tensor:
        return self.decoder(features) + self.bias

    def forward(
        self,
        x: Tensor,
        output_features: bool = False,
        ghost_mask: Optional[Tensor] = None,
        unnormalize: bool = False,
    ) -> Union[Tensor, tuple[Tensor, ...]]:
        x, original_norms = self._normalize_input_and_get_norms(x)

        if ghost_mask is None:
            features = self.encode(x)
            x_hat = self.decode(features)
            if unnormalize:
                x_hat = self._unnormalize_output(x_hat, original_norms)
            return (x_hat, features) if output_features else x_hat

        f_pre = self.encoder(x - self.bias)
        f_ghost = torch.exp(f_pre) * ghost_mask.to(f_pre)
        features = F.relu(f_pre)
        x_ghost = self.decoder(f_ghost)
        x_hat = self.decode(features)

        if unnormalize:
            x_ghost = self._unnormalize_output(x_ghost, original_norms)
            x_hat = self._unnormalize_output(x_hat, original_norms)

        if output_features:
            return x_hat, x_ghost, features
        return x_hat, x_ghost

    @torch.no_grad()
    def encode_feat_subset(
        self,
        x: Tensor,
        feat_list: Sequence[int],
        normalize_features: bool = False,
    ) -> Tensor:
        encoder_w_subset = self.encoder.weight[feat_list, :]
        encoder_b_subset = self.encoder.bias[feat_list]
        x, _ = self._normalize_input_and_get_norms(x)
        features = F.relu((x - self.bias) @ encoder_w_subset.T + encoder_b_subset)
        if normalize_features:
            features = features / self.activation_rescale_factor[feat_list]
        return features

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: DeviceLike = None,
        **kwargs: Any,
    ) -> "ReLUSAE":
        device = _resolve_device(device)
        state_dict = _load_checkpoint(path, map_location=device)
        activation_dim, dict_size = _infer_dims(state_dict)
        model = cls(
            activation_dim,
            dict_size,
            normalize_to_sqrt_d=_load_normalize_flag(state_dict),
        )
        model.load_state_dict(state_dict)
        return model.to(device)


class IdentityDict(Dictionary):
    """Identity dictionary for treating neurons as features directly."""

    def __init__(
        self,
        activation_dim: Optional[int] = None,
        normalize_to_sqrt_d: bool = False,
    ) -> None:
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        buffer = torch.ones(activation_dim) if activation_dim else torch.tensor([])
        self.register_buffer("activation_rescale_factor", buffer)

    def encode(self, x: Tensor, normalize_features: bool = False) -> Tensor:
        out = self.encoder(x)
        if normalize_features:
            out = out / self.activation_rescale_factor
        return out

    def decode(self, features: Tensor) -> Tensor:
        return self.decoder(features)

    @torch.no_grad()
    def encode_feat_subset(
        self,
        x: Tensor,
        feat_list: Sequence[int],
        normalize_features: bool = False,
    ) -> Tensor:
        features = x[:, feat_list]
        if normalize_features:
            features = features / self.activation_rescale_factor[feat_list]
        return features

    def forward(
        self,
        x: Tensor,
        output_features: bool = False,
        ghost_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        return (x, x) if output_features else x


class ReLUSAE_Tied(Dictionary):
    """
    ReLU SAE with tied encoder/decoder weights and dual biases.

    Reference: https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        normalize_to_sqrt_d: bool = False,
    ) -> None:
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)

        w = torch.randn(activation_dim, dict_size)
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

        self.register_buffer("activation_rescale_factor", torch.ones(dict_size))
        self._make_contiguous()

    def encode(self, x: Tensor, normalize_features: bool = False) -> Tensor:
        features = F.relu(self.encoder(x))
        if normalize_features:
            features = features / self.activation_rescale_factor
        return features

    def decode(self, features: Tensor) -> Tensor:
        return self.decoder(features)

    def forward(
        self,
        x: Tensor,
        output_features: bool = False,
        unnormalize: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        x, original_norms = self._normalize_input_and_get_norms(x)
        features = self.encode(x)
        x_hat = self.decode(features)
        if unnormalize:
            x_hat = self._unnormalize_output(x_hat, original_norms)
        return (x_hat, features) if output_features else x_hat

    @torch.no_grad()
    def encode_feat_subset(
        self,
        x: Tensor,
        feat_list: Sequence[int],
        normalize_features: bool = False,
    ) -> Tensor:
        encoder_w_subset = self.encoder.weight[feat_list, :]
        encoder_b_subset = self.encoder.bias[feat_list]
        features = F.relu(x @ encoder_w_subset.T + encoder_b_subset)
        if normalize_features:
            features = features / self.activation_rescale_factor[feat_list]
        return features

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: DeviceLike = None,
        **kwargs: Any,
    ) -> "ReLUSAE_Tied":
        device = _resolve_device(device)
        state_dict = _load_checkpoint(path, map_location=device)
        activation_dim, dict_size = _infer_dims(state_dict)
        model = cls(
            activation_dim,
            dict_size,
            normalize_to_sqrt_d=_load_normalize_flag(state_dict),
        )
        model.load_state_dict(state_dict)
        return model.to(device)


class TopKSAE(Dictionary):
    """
    Top-k SAE from https://arxiv.org/abs/2406.04093
    """

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        k: int,
        normalize_to_sqrt_d: bool = False,
    ) -> None:
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k={k} must be a positive integer")

        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

        self.register_buffer("activation_rescale_factor", torch.ones(dict_size))
        self._make_contiguous()

    def encode(
        self,
        x: Tensor,
        return_topk: bool = False,
        use_threshold: bool = False,
        normalize_features: bool = False,
    ) -> Union[Tensor, tuple[Tensor, ...]]:
        post_relu = F.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            encoded = post_relu * (post_relu > self.threshold)
            if normalize_features:
                encoded = encoded / self.activation_rescale_factor
            if return_topk:
                post_topk = post_relu.topk(self.k, sorted=False, dim=-1)
                return encoded, post_topk.values, post_topk.indices, post_relu
            return encoded

        post_topk = post_relu.topk(self.k, sorted=False, dim=-1)
        encoded = torch.zeros_like(post_relu).scatter_(
            dim=-1, index=post_topk.indices, src=post_topk.values
        )
        if normalize_features:
            encoded = encoded / self.activation_rescale_factor

        if return_topk:
            return encoded, post_topk.values, post_topk.indices, post_relu
        return encoded

    def decode(self, features: Tensor) -> Tensor:
        return self.decoder(features) + self.b_dec

    def forward(
        self,
        x: Tensor,
        output_features: bool = False,
        unnormalize: bool = True,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        x, original_norms = self._normalize_input_and_get_norms(x)
        encoded = self.encode(x)
        x_hat = self.decode(encoded)
        if unnormalize:
            x_hat = self._unnormalize_output(x_hat, original_norms)
        return (x_hat, encoded) if output_features else x_hat

    def scale_biases(self, scale: float) -> None:
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @torch.no_grad()
    def encode_feat_subset(
        self,
        x: Tensor,
        feat_list: Sequence[int],
        normalize_features: bool = False,
    ) -> Tensor:
        encoder_w_subset = self.encoder.weight[feat_list, :]
        encoder_b_subset = self.encoder.bias[feat_list]
        post_relu = F.relu((x - self.b_dec) @ encoder_w_subset.T + encoder_b_subset)
        features = post_relu * (post_relu > self.threshold)
        if normalize_features:
            features = features / self.activation_rescale_factor[feat_list]
        return features

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        k: Optional[int] = None,
        device: DeviceLike = None,
        **kwargs: Any,
    ) -> "TopKSAE":
        device = _resolve_device(device)
        state_dict = _load_checkpoint(path, map_location=device)
        activation_dim, dict_size = _infer_dims(state_dict)

        if k is None:
            k = int(state_dict["k"].item())
        elif int(state_dict["k"].item()) != k:
            raise ValueError(f"k={k} != state_dict k={state_dict['k'].item()}")

        model = cls(
            activation_dim,
            dict_size,
            k,
            normalize_to_sqrt_d=_load_normalize_flag(state_dict),
        )
        model.load_state_dict(state_dict)
        return model.to(device)


class JumpReLUSAE(Dictionary):
    """Jump ReLU sparse autoencoder."""

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        device: DeviceLike = "cpu",
        normalize_to_sqrt_d: bool = False,
    ) -> None:
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)

        init.kaiming_uniform_(self.decoder.weight)
        self.decoder.weight.data = self.decoder.weight.data / self.decoder.weight.data.norm(
            dim=0, keepdim=True
        )
        self.encoder.weight.data = self.decoder.weight.data.clone().T
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

        device = _resolve_device(device)
        self.threshold = nn.Parameter(torch.ones(dict_size, device=device) * 0.001)
        self.apply_b_dec_to_input = False

        self.register_buffer("activation_rescale_factor", torch.ones(dict_size))
        self._make_contiguous()

    def encode(
        self,
        x: Tensor,
        output_pre_jump: bool = False,
        normalize_features: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        if self.apply_b_dec_to_input:
            x = x - self.decoder.bias
        pre_jump = self.encoder(x)
        features = F.relu(pre_jump * (pre_jump > self.threshold))
        if normalize_features:
            features = features / self.activation_rescale_factor
        if output_pre_jump:
            return features, pre_jump
        return features

    def decode(self, features: Tensor) -> Tensor:
        return self.decoder(features)

    def forward(
        self,
        x: Tensor,
        output_features: bool = False,
        unnormalize: bool = True,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        x, original_norms = self._normalize_input_and_get_norms(x)
        features = self.encode(x)
        x_hat = self.decode(features)
        if unnormalize:
            x_hat = self._unnormalize_output(x_hat, original_norms)
        return (x_hat, features) if output_features else x_hat

    def scale_biases(self, scale: float) -> None:
        self.decoder.bias.data *= scale
        self.encoder.bias.data *= scale
        self.threshold.data *= scale

    @torch.no_grad()
    def encode_feat_subset(
        self,
        x: Tensor,
        feat_list: Sequence[int],
        normalize_features: bool = False,
    ) -> Tensor:
        if self.apply_b_dec_to_input:
            x = x - self.decoder.bias
        encoder_w_subset = self.encoder.weight[feat_list, :]
        encoder_b_subset = self.encoder.bias[feat_list]
        pre_jump = x @ encoder_w_subset.T + encoder_b_subset
        features = F.relu(pre_jump * (pre_jump > self.threshold[feat_list]))
        if normalize_features:
            features = features / self.activation_rescale_factor[feat_list]
        return features

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: DeviceLike = None,
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> "JumpReLUSAE":
        device = _resolve_device(device)
        state_dict = _load_checkpoint(path, map_location=device)
        activation_dim, dict_size = _infer_dims(state_dict)
        model = cls(
            activation_dim,
            dict_size,
            device=device,
            normalize_to_sqrt_d=_load_normalize_flag(state_dict),
        )
        model.load_state_dict(state_dict)
        return model.to(dtype=dtype, device=device)


class BatchTopKSAE(Dictionary):
    """Batch top-k sparse autoencoder."""

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        k: int,
        normalize_to_sqrt_d: bool = False,
    ) -> None:
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k={k} must be a positive integer")

        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

        self.register_buffer("activation_rescale_factor", torch.ones(dict_size))
        self._make_contiguous()

    def encode(
        self,
        x: Tensor,
        return_active: bool = False,
        use_threshold: bool = True,
        normalize_features: bool = False,
    ) -> Union[Tensor, tuple[Tensor, ...]]:
        post_relu = F.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            encoded = post_relu * (post_relu > self.threshold)
        else:
            flattened = post_relu.flatten()
            post_topk = flattened.topk(self.k * x.size(0), sorted=False, dim=-1)
            encoded = (
                torch.zeros_like(flattened)
                .scatter_(-1, post_topk.indices, post_topk.values)
                .reshape(post_relu.shape)
            )

        if normalize_features:
            encoded = encoded / self.activation_rescale_factor

        if return_active:
            return encoded, encoded.sum(0) > 0, post_relu
        return encoded

    def decode(self, features: Tensor) -> Tensor:
        return self.decoder(features) + self.b_dec

    def forward(
        self,
        x: Tensor,
        output_features: bool = False,
        unnormalize: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        x, original_norms = self._normalize_input_and_get_norms(x)
        encoded = self.encode(x)
        x_hat = self.decode(encoded)
        if unnormalize:
            x_hat = self._unnormalize_output(x_hat, original_norms)
        return (x_hat, encoded) if output_features else x_hat

    def scale_biases(self, scale: float) -> None:
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @torch.no_grad()
    def encode_feat_subset(
        self,
        x: Tensor,
        feat_list: Sequence[int],
        normalize_features: bool = False,
    ) -> Tensor:
        encoder_w_subset = self.encoder.weight[feat_list, :]
        encoder_b_subset = self.encoder.bias[feat_list]
        encoded = F.relu((x - self.b_dec) @ encoder_w_subset.T + encoder_b_subset)
        encoded = encoded * (encoded > self.threshold)
        if normalize_features:
            encoded = encoded / self.activation_rescale_factor[feat_list]
        return encoded

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        device: DeviceLike = None,
        **kwargs: Any,
    ) -> "BatchTopKSAE":
        device = _resolve_device(device)
        state_dict = _load_checkpoint(path, map_location=device)

        if k is None:
            k = int(state_dict["k"].item())
        elif int(state_dict["k"].item()) != k:
            raise ValueError(f"k={k} != state_dict k={state_dict['k'].item()}")

        if threshold is None:
            threshold = float(state_dict["threshold"].item())
        elif float(state_dict["threshold"].item()) != threshold:
            raise ValueError(
                f"threshold={threshold} != state_dict threshold={state_dict['threshold'].item()}"
            )

        activation_dim, dict_size = _infer_dims(state_dict)
        model = cls(
            activation_dim=activation_dim,
            dict_size=dict_size,
            k=k,
            normalize_to_sqrt_d=_load_normalize_flag(state_dict),
        ).to(device)
        model.load_state_dict(state_dict)
        return model.to(device)


SAE_REGISTRY: dict[str, Type[Dictionary]] = {
    "relu": ReLUSAE,
    "relu_tied": ReLUSAE_Tied,
    "topk": TopKSAE,
    "jump_relu": JumpReLUSAE,
    "batch_topk": BatchTopKSAE,
    "identity": IdentityDict,
}


def build_sae(
    sae_type: Union[str, SAEType],
    activation_dim: int,
    dict_size: int,
    **kwargs: Any,
) -> Dictionary:
    """Construct an SAE model by registry name."""
    if sae_type not in SAE_REGISTRY:
        valid = ", ".join(sorted(SAE_REGISTRY))
        raise ValueError(f"Unknown SAE type '{sae_type}'. Valid options: {valid}")
    return SAE_REGISTRY[sae_type](activation_dim, dict_size, **kwargs)


def load_sae_from_checkpoint(
    path: str,
    sae_type: Optional[Union[str, SAEType]] = None,
    device: DeviceLike = None,
    **kwargs: Any,
) -> Dictionary:
    """Load a pretrained SAE checkpoint."""
    if sae_type is None:
        state_dict = _load_checkpoint(path, map_location="cpu")
        if "k" in state_dict:
            sae_type = "topk"
        elif "bias" in state_dict:
            sae_type = "relu"
        elif (
            "threshold" in state_dict
            and state_dict["threshold"].ndim == 1
            and state_dict["threshold"].numel() > 1
        ):
            sae_type = "jump_relu"
        else:
            sae_type = "relu_tied"

    if sae_type not in SAE_REGISTRY:
        valid = ", ".join(sorted(SAE_REGISTRY))
        raise ValueError(f"Unknown SAE type '{sae_type}'. Valid options: {valid}")

    loader: Callable[..., Dictionary] = SAE_REGISTRY[sae_type].from_pretrained
    return loader(path, device=device, **kwargs)


@torch.no_grad()
def set_decoder_norm_to_unit_norm(
    decoder_weight: nn.Parameter,
    activation_dim: int,
    dict_size: int,
) -> Tensor:
    """Normalize decoder columns to unit norm."""
    rows, cols = decoder_weight.shape
    if rows != activation_dim or cols != dict_size:
        raise ValueError(
            f"Expected decoder shape ({activation_dim}, {dict_size}), got ({rows}, {cols})"
        )

    eps = torch.finfo(decoder_weight.dtype).eps
    norm = torch.norm(decoder_weight.data, dim=0, keepdim=True)
    decoder_weight.data /= norm + eps
    return decoder_weight.data


@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(
    decoder_weight: Tensor,
    decoder_grad: Tensor,
    activation_dim: int,
    dict_size: int,
) -> Tensor:
    """Remove gradient components parallel to decoder directions."""
    rows, cols = decoder_weight.shape
    if rows != activation_dim or cols != dict_size:
        raise ValueError(
            f"Expected decoder shape ({activation_dim}, {dict_size}), got ({rows}, {cols})"
        )

    normed = decoder_weight / (torch.norm(decoder_weight, dim=0, keepdim=True) + 1e-6)
    parallel = einops.einsum(
        decoder_grad,
        normed,
        "d_in d_sae, d_in d_sae -> d_sae",
    )
    decoder_grad -= einops.einsum(
        parallel,
        normed,
        "d_sae, d_in d_sae -> d_in d_sae",
    )
    return decoder_grad
