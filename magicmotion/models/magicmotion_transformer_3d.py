import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import register_to_config
from diffusers.models.transformers import CogVideoXTransformer3DModel
from diffusers.utils import (
    USE_PEFT_BACKEND,
    BaseOutput,
    is_accelerate_available,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from einops import rearrange
from models.latent_segmentation import SemanticFPNHead

if is_accelerate_available():
    import accelerate

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class MagicMotionTransformer3DModelOutput(BaseOutput):
    """
    The output of [`MagicMotionTransformer3DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: "torch.Tensor"  # noqa: F821
    mask_pred: Optional[torch.Tensor] = None


class MagicMotionTransformer3DModel(CogVideoXTransformer3DModel):

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        use_perception_head=False,
    ):
        config = {}  # 避免修改原始的 config 对象
        for k, v in self.config.items():
            if "use_perception_head" not in k:
                config[k] = v
        super().__init__(**config)
        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        if use_perception_head:
            self.start_layer = 0
            self.end_layer = self.config.num_layers - 1
            self.perception_head = SemanticFPNHead(
                in_channels=inner_dim,
                out_channels=self.config.in_channels // 2,
                num_tensors=self.end_layer - self.start_layer + 1,
                patch_size=self.config.patch_size,
            )
        else:
            self.perception_head = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        # Support Controlnet
        controlnet_states: torch.Tensor = None,
        controlnet_weights: Optional[
            Union[float, int, list, np.ndarray, torch.FloatTensor]
        ] = 1.0,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        diffusion_features = []
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states, encoder_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

            if (controlnet_states is not None) and (i < len(controlnet_states)):
                controlnet_states_block = controlnet_states[i]
                controlnet_block_weight = 1.0
                if isinstance(
                    controlnet_weights, (list, np.ndarray)
                ) or torch.is_tensor(controlnet_weights):
                    controlnet_block_weight = controlnet_weights[i]
                elif isinstance(controlnet_weights, (float, int)):
                    controlnet_block_weight = controlnet_weights

                hidden_states = (
                    hidden_states + controlnet_states_block * controlnet_block_weight
                )
            diffusion_features.append(hidden_states)

        # Visualize hidden_states
        p = self.config.patch_size
        p_t = self.config.patch_size_t
        if p_t is None:
            p_t = 1

        # Do Latent Segmentation
        mask_pred = None
        if self.perception_head is not None:
            features = []
            for i in range(self.start_layer, self.end_layer + 1):
                spatial_feature = rearrange(
                    diffusion_features[i],
                    "B (T H W) C -> (B T) C H W",
                    T=num_frames // p_t,
                    H=height // p,
                    W=width // p,
                )
                # print(spatial_feature.shape) # torch.Size([26, 3072, 30, 45])
                features.append(spatial_feature)

            mask_pred = self.perception_head(features)
            mask_pred = rearrange(
                mask_pred,
                "(B T) C H W -> B T C H W",
                T=num_frames // p_t,
                H=height,
                W=width,
            )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(
                batch_size, num_frames, height // p, width // p, -1, p, p
            )
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size,
                (num_frames + p_t - 1) // p_t,
                height // p,
                width // p,
                -1,
                p_t,
                p,
                p,
            )
            output = (
                output.permute(0, 1, 5, 4, 2, 6, 3, 7)
                .flatten(6, 7)
                .flatten(4, 5)
                .flatten(1, 2)
            )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, mask_pred)
        return MagicMotionTransformer3DModelOutput(sample=output, mask_pred=mask_pred)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from diffusers import __version__
        from diffusers.models.modeling_utils import (
            _determine_device_map,
            _fetch_index_file,
            _fetch_index_file_legacy,
        )
        from diffusers.utils import _get_checkpoint_shard_files
        from huggingface_hub.constants import HF_HUB_CACHE

        print(
            f"loading transformer's pretrained weights from {pretrained_model_name_or_path} ..."
        )

        cache_dir = kwargs.pop("cache_dir", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", True)
        variant = kwargs.pop("variant", None)
        quantization_config = kwargs.pop("quantization_config", None)
        use_safetensors = kwargs.pop("use_safetensors", True)

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            **kwargs,
        )

        config["_class_name"] = cls.__name__
        # no in-place modification of the original config.
        config = copy.deepcopy(config)

        if low_cpu_mem_usage:
            # Instantiate model with empty weights
            with accelerate.init_empty_weights():
                model = cls.from_config(config, **unused_kwargs)

        # Initialize perception_head
        if unused_kwargs.get("use_perception_head", False):
            inner_dim = (
                model.config.num_attention_heads * model.config.attention_head_dim
            )
            model.perception_head = SemanticFPNHead(
                in_channels=inner_dim,
                out_channels=model.config.in_channels // 2,
                num_tensors=model.config.num_layers,
                patch_size=model.config.patch_size,
            )

        # Determine if we're loading from a directory of sharded checkpoints.
        index_file = None
        is_local = os.path.isdir(pretrained_model_name_or_path)
        index_file_kwargs = {
            "is_local": is_local,
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "subfolder": subfolder or "",
            "use_safetensors": use_safetensors,
            "cache_dir": cache_dir,
            "variant": variant,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
            "user_agent": user_agent,
            "commit_hash": commit_hash,
        }
        index_file = _fetch_index_file(**index_file_kwargs)
        # In case the index file was not found we still have to consider the legacy format.
        # this becomes applicable when the variant is not None.
        if variant is not None and (
            index_file is None or not os.path.exists(index_file)
        ):
            index_file = _fetch_index_file_legacy(**index_file_kwargs)

        is_sharded = True
        force_hook = True
        hf_quantizer = None
        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
            (torch_dtype == torch.float16)
            or hasattr(hf_quantizer, "use_keep_in_fp32_modules")
        )
        if use_keep_in_fp32_modules:
            keep_in_fp32_modules = cls._keep_in_fp32_modules
            if not isinstance(keep_in_fp32_modules, list):
                keep_in_fp32_modules = [keep_in_fp32_modules]

            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.info(
                    "Set `low_cpu_mem_usage` to True as `_keep_in_fp32_modules` is not None."
                )
            elif not low_cpu_mem_usage:
                raise ValueError(
                    "`low_cpu_mem_usage` cannot be False when `keep_in_fp32_modules` is True."
                )
        else:
            keep_in_fp32_modules = []

        device_map = _determine_device_map(
            model,
            device_map,
            max_memory,
            torch_dtype,
            keep_in_fp32_modules,
            hf_quantizer,
        )
        if device_map is None and is_sharded:
            # we load the parameters on the cpu
            device_map = {"": "cpu"}
            force_hook = False

        accelerate.load_checkpoint_and_dispatch(
            model,
            index_file,
            device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            dtype=torch_dtype,
            force_hooks=force_hook,
            strict=True,
        )
        return model
