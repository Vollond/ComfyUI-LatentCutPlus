import logging
import gc
import torch
import logging
import gc
import torch
import numpy as np
import subprocess
import os
import sys
import json
import datetime
import re
from pathlib import Path
from PIL import Image
import folder_paths



class AnyType(str):
    """Special class for representing any type - always returns True on type comparison"""
    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class LatentCutPlus:
    """Slice latent tensor along a dimension (t/x/y) with smart index/amount handling."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "dim": (["t", "x", "y"], {"default": "t"}),
                "index": ("INT", {"default": 0, "min": -2147483647, "max": 2147483647}),
                "amount": ("INT", {"default": 1, "min": 1, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "latent"
    
    @classmethod
    def IS_CHANGED(cls, samples, dim, index, amount):
        """Invalidate cache on shape/dtype/device/params change"""
        if samples is None or not isinstance(samples, dict):
            import random
            return random.random()
        
        if "samples" not in samples:
            import random
            return random.random()
        
        x = samples.get("samples")
        if not isinstance(x, torch.Tensor):
            import random
            return random.random()
        
        import hashlib
        m = hashlib.sha256()
        m.update(str(tuple(x.shape)).encode())
        m.update(str(x.dtype).encode())
        m.update(str(x.device).encode())
        m.update(f"{dim}_{index}_{amount}".encode())
        return m.digest().hex()


    
    def execute(self, samples, dim: str, index: int, amount: int):
        out = {}

        if "samples" not in samples:
            raise RuntimeError("LatentCutPlus: input LATENT has no 'samples' key")

        x: torch.Tensor = samples["samples"]
        if not isinstance(x, torch.Tensor):
            raise RuntimeError("LatentCutPlus: samples['samples'] is not a torch.Tensor")

        # Map dimension name to axis
        if dim == "x":
            axis = x.ndim - 1
        elif dim == "y":
            axis = x.ndim - 2
        else:  # "t"
            axis = x.ndim - 3

        size = int(x.shape[axis])

        logging.info(f"[LatentCutPlus] Input shape: {tuple(x.shape)}, dim={dim} (axis={axis}), size={size}")
        logging.info(f"[LatentCutPlus] Raw params: index={index}, amount={amount}")

        # Protection from overflow: clamp index to valid range
        original_index = index
        if index < 0:
            # Python-style negative indexing
            index = max(0, size + index)
            if original_index != index:
                logging.warning(f"[LatentCutPlus] Negative index {original_index} normalized to {index}")
        else:
            # Protection: if index >= size, clamp to size-1
            if index >= size:
                logging.warning(f"[LatentCutPlus] Index {index} >= size {size}, clamping to {size-1}")
                index = max(0, size - 1)

        start = index

        # Protection: check remaining size
        remaining = size - start
        if remaining <= 0:
            logging.error(f"[LatentCutPlus] No data to slice! start={start}, size={size}")
            # Return empty slice
            out["samples"] = x[:, :, 0:0] if axis == 2 else x
            return (out,)

        # Smart amount handling: if amount >= remaining, slice to end
        if amount >= remaining:
            end = size
            actual_amount = remaining
            logging.info(f"[LatentCutPlus] Amount {amount} >= remaining {remaining}, slicing to end")
        else:
            actual_amount = max(1, int(amount))
            end = start + actual_amount

        logging.info(f"[LatentCutPlus] Final slice: [{start}:{end}] (length={actual_amount})")

        # Build slice tuple
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(start, end)

        out_tensor = x[tuple(sl)].contiguous()
        out["samples"] = out_tensor

        logging.info(f"[LatentCutPlus] Output shape: {tuple(out_tensor.shape)}")

        # Handle batch_index if present (для AnimateDiff и подобных)
        if "batch_index" in samples:
            bi = samples["batch_index"]
            if isinstance(bi, torch.Tensor) and bi.ndim == x.ndim and int(bi.shape[axis]) == size:
                out["batch_index"] = bi[tuple(sl)].contiguous()
                logging.info(f"[LatentCutPlus] Batch index sliced: {tuple(out['batch_index'].shape)}")
            else:
                if isinstance(bi, torch.Tensor):
                    out["batch_index"] = bi.clone()
                else:
                    out["batch_index"] = bi

        # Handle noise_mask if present
        if "noise_mask" in samples:
            nm = samples["noise_mask"]
            if isinstance(nm, torch.Tensor) and nm.ndim == x.ndim and int(nm.shape[axis]) == size:
                out["noise_mask"] = nm[tuple(sl)].contiguous()
                logging.info(f"[LatentCutPlus] Noise mask sliced: {tuple(out['noise_mask'].shape)}")
            else:
                if isinstance(nm, torch.Tensor):
                    out["noise_mask"] = nm.clone()
                else:
                    out["noise_mask"] = nm

        return (out,)



# ============================================================================
# LTXV EMPTY LATENT AUDIO DEBUG (old API)
# ============================================================================
class LTXVEmptyLatentAudioDebug:
    """Debug node for LTXV empty latent with detailed logging."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 97, "min": 1, "max": 1024}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "audio_vae": ("VAE",),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute"
    CATEGORY = "latent/audio"
    
    def execute(self, width: int, height: int, length: int, batch_size: int, audio_vae=None):
        logging.info("=" * 80)
        logging.info("[LTXVEmptyLatentAudioDebug] Creating empty latent")
        logging.info(f"[LTXVEmptyLatentAudioDebug] width={width}, height={height}, length={length}, batch_size={batch_size}")
        if audio_vae is not None:
            logging.info(f"[LTXVEmptyLatentAudioDebug] audio_vae provided: {type(audio_vae).__name__}")

        # LTXV latent dimensions: (batch, channels, frames, height/8, width/8)
        latent_t = length
        latent_h = height // 8
        latent_w = width // 8
        channels = 128

        shape = (batch_size, channels, latent_t, latent_h, latent_w)
        logging.info(f"[LTXVEmptyLatentAudioDebug] Creating latent with shape: {shape}")

        latent = torch.zeros(shape, dtype=torch.float32, device="cpu")
        
        logging.info(f"[LTXVEmptyLatentAudioDebug] Created tensor: shape={tuple(latent.shape)}, dtype={latent.dtype}, device={latent.device}")
        logging.info(f"[LTXVEmptyLatentAudioDebug] Memory size: {latent.element_size() * latent.nelement() / 1024 / 1024:.2f} MB")
        logging.info("=" * 80)

        return ({"samples": latent},)



class LatentDebugInfo:
    """Debug node for inspecting latent tensor with detailed statistics."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "label": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "execute"
    CATEGORY = "latent"
    OUTPUT_NODE = False
    
    def execute(self, samples, label: str = ""):
        log_id = f"[LatentDebugInfo:{label}]" if label else "[LatentDebugInfo]"
        
        if "samples" not in samples:
            logging.warning(f"{log_id} No 'samples' key in latent dict!")
            return (samples,)
        
        x = samples["samples"]
        
        if not isinstance(x, torch.Tensor):
            logging.warning(f"{log_id} samples['samples'] is not a tensor!")
            return (samples,)
        
        logging.info("=" * 80)
        logging.info(f"{log_id} LATENT TENSOR INFORMATION")
        logging.info("=" * 80)
        logging.info(f"{log_id} Shape: {tuple(x.shape)}")
        logging.info(f"{log_id} Dtype: {x.dtype}")
        logging.info(f"{log_id} Device: {x.device}")
        logging.info(f"{log_id} Total elements: {x.numel()}")
        logging.info(f"{log_id} Memory (MB): {x.element_size() * x.numel() / 1024 / 1024:.2f}")
        
        try:
            logging.info(f"{log_id} Min value: {x.min().item():.6f}")
            logging.info(f"{log_id} Max value: {x.max().item():.6f}")
            logging.info(f"{log_id} Mean value: {x.mean().item():.6f}")
            logging.info(f"{log_id} Std value: {x.std().item():.6f}")
        except Exception as e:
            logging.warning(f"{log_id} Could not compute statistics: {e}")
        
        # Log metadata keys
        logging.info(f"{log_id} Metadata keys in latent dict:")
        for key in samples.keys():
            if key != "samples":
                value = samples[key]
                if isinstance(value, torch.Tensor):
                    logging.info(f"{log_id}   - {key}: Tensor{tuple(value.shape)}")
                else:
                    value_str = str(value)[:100]
                    logging.info(f"{log_id}   - {key}: {value_str}")
        
        logging.info("=" * 80)
        
        return (samples,)



class DebugAny:
    """Universal debug node that accepts ANY type using AnyType class."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "value": (any_type, {}),
            },
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "execute"
    CATEGORY = "utils"
    OUTPUT_NODE = False
    
    def execute(self, label: str = "", value=None):
        log_id = f"[DebugAny:{label}]" if label else "[DebugAny]"
        
        if value is None:
            logging.warning(f"{log_id} No value connected!")
            return (None,)
        
        try:
            if isinstance(value, torch.Tensor):
                value_str = f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
                try:
                    value_details = f"min={value.min().item():.4f}, max={value.max().item():.4f}, mean={value.mean().item():.4f}"
                except Exception:
                    value_details = ""
            elif isinstance(value, dict):
                value_str = f"Dict with keys: {list(value.keys())}"
                value_details = ""
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        value_details += f"\n{log_id}   {k}: Tensor{tuple(v.shape)}"
                    else:
                        value_details += f"\n{log_id}   {k}: {type(v).__name__} = {str(v)[:100]}"
            elif isinstance(value, (list, tuple)):
                value_str = f"{type(value).__name__}(length={len(value)})"
                value_details = f"First 3 items: {str(value[:3])[:200]}"
            elif isinstance(value, (int, float, str, bool)):
                value_str = f"{type(value).__name__} = {value}"
                value_details = ""
            else:
                value_str = f"{type(value).__name__}"
                try:
                    value_details = f"repr: {repr(value)[:300]}"
                except Exception:
                    value_details = "<cannot repr>"
        except Exception as e:
            value_str = f"<Error: {e}>"
            value_details = ""
        
        logging.info("=" * 80)
        logging.info(f"{log_id} VALUE DEBUG")
        logging.info("=" * 80)
        logging.info(f"{log_id} Python type: {type(value).__name__}")
        logging.info(f"{log_id} Value: {value_str}")
        if value_details:
            logging.info(f"{log_id} Details: {value_details}")
        logging.info("=" * 80)
        
        return (value,)
# nodes_direct_encode_PRODUCTION.py
# Production-ready LTXV Direct MP4 Encoder
# All fixes applied and validated

import logging
import gc
import torch
import numpy as np
import subprocess
import os
import shutil
import time
from pathlib import Path
import folder_paths


def compute_chunk_boundaries_FIXED(chunk_start, tile_length, overlap, total_frames):
    """
    Compute chunk boundaries with proper overlap

    Args:
        chunk_start: Starting frame index (output position, not latent position)
        tile_length: Chunk size in latent frames
        overlap: Overlap in latent frames
        total_frames: Total latent frames

    Returns:
        (latent_start, latent_end, frames_to_drop)
    """
    if chunk_start == 0:
        # First chunk: no overlap
        latent_start = 0
        latent_end = min(tile_length, total_frames)
        frames_to_drop = 0
    else:
        # Subsequent chunks: include overlap for smooth transition
        latent_start = max(0, chunk_start - overlap)
        latent_end = min(chunk_start - overlap + tile_length, total_frames)
        frames_to_drop = overlap

    return latent_start, latent_end, frames_to_drop


def compute_overlap_decoded_frames(overlap_latent, time_scale):
    """
    Calculate decoded frames to drop for given latent overlap

    CRITICAL: VAE formula is output = 1 + (latent-1)*scale
    The +1 is NON-LINEAR and must be accounted for!

    Args:
        overlap_latent: Number of overlapping latent frames
        time_scale: VAE time scale factor (typically 8 for LTXV)

    Returns:
        Number of decoded frames to drop

    Example:
        overlap_latent=8, time_scale=8
        → 1 + (8-1)*8 = 57 frames (NOT 64!)
    """
    if overlap_latent == 0:
        return 0
    return 1 + (overlap_latent - 1) * time_scale


EXTENSION_MAP = {
    "h264-mp4": "mp4",
    "h265-mp4": "mp4",
    "hevc-mp4": "mp4",
    "webm": "webm",
}


def check_disk_space(output_path, estimated_size_gb):
    """Check if sufficient disk space is available"""
    output_dir = Path(output_path).parent
    try:
        stat = shutil.disk_usage(output_dir)
        free_gb = stat.free / (1024**3)

        if free_gb < estimated_size_gb * 1.2:  # 20% safety margin
            raise RuntimeError(
                f"Insufficient disk space! Need ~{estimated_size_gb:.1f} GB, "
                f"only {free_gb:.1f} GB available in {output_dir}"
            )

        logging.info(f"[DirectEncode] Disk space: {free_gb:.1f} GB free")
        return free_gb
    except Exception as e:
        logging.warning(f"[DirectEncode] Could not check disk space: {e}")
        return None


class LTXVTiledVAEDecode:
    """Base spatial tiled VAE decoder"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "latents": ("LATENT",),
                "horizontal_tiles": ("INT", {"default": 1, "min": 1, "max": 6}),
                "vertical_tiles": ("INT", {"default": 1, "min": 1, "max": 6}),
                "overlap": ("INT", {"default": 1, "min": 1, "max": 8}),
                "last_frame_fix": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "working_device": (["cpu", "auto"], {"default": "auto"}),
                "working_dtype": (["float16", "float32", "auto"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent"

    def decode(self, vae, latents, horizontal_tiles, vertical_tiles, overlap, last_frame_fix, working_device="auto", working_dtype="auto"):
        samples = latents["samples"]
        if last_frame_fix:
            last_frame = samples[:, :, -1:, :, :]
            samples = torch.cat([samples, last_frame], dim=2)

        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = vae.downscale_index_formula
        image_frames = 1 + (frames - 1) * time_scale_factor
        output_height = height * height_scale_factor
        output_width = width * width_scale_factor

        base_tile_height = (height + (vertical_tiles - 1) * overlap) // vertical_tiles
        base_tile_width = (width + (horizontal_tiles - 1) * overlap) // horizontal_tiles

        target_device = samples.device if working_device == "auto" else working_device
        if working_dtype == "auto":
            target_dtype = samples.dtype
        elif working_dtype == "float16":
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32

        output = torch.zeros((batch, image_frames, output_height, output_width, 3), device=target_device, dtype=target_dtype)
        weights = torch.zeros((batch, image_frames, output_height, output_width, 1), device=target_device, dtype=target_dtype)

        for v in range(vertical_tiles):
            for h in range(horizontal_tiles):
                h_start = h * (base_tile_width - overlap)
                v_start = v * (base_tile_height - overlap)
                h_end = min(h_start + base_tile_width, width) if h < horizontal_tiles - 1 else width
                v_end = min(v_start + base_tile_height, height) if v < vertical_tiles - 1 else height

                tile = samples[:, :, :, v_start:v_end, h_start:h_end]
                decoded_tile = vae.decode(tile)

                out_h_start = v_start * height_scale_factor
                out_h_end = v_end * height_scale_factor
                out_w_start = h_start * width_scale_factor
                out_w_end = h_end * width_scale_factor
                tile_out_height = out_h_end - out_h_start
                tile_out_width = out_w_end - out_w_start

                tile_weights = torch.ones((batch, image_frames, tile_out_height, tile_out_width, 1), 
                                          device=decoded_tile.device, dtype=decoded_tile.dtype)

                overlap_out_h = overlap * height_scale_factor
                overlap_out_w = overlap * width_scale_factor

                if h > 0:
                    h_blend = torch.linspace(0, 1, overlap_out_w, device=decoded_tile.device)
                    tile_weights[:, :, :, :overlap_out_w, :] *= h_blend.view(1, 1, 1, -1, 1)
                if h < horizontal_tiles - 1:
                    h_blend = torch.linspace(1, 0, overlap_out_w, device=decoded_tile.device)
                    tile_weights[:, :, :, -overlap_out_w:, :] *= h_blend.view(1, 1, 1, -1, 1)
                if v > 0:
                    v_blend = torch.linspace(0, 1, overlap_out_h, device=decoded_tile.device)
                    tile_weights[:, :, :overlap_out_h, :, :] *= v_blend.view(1, 1, -1, 1, 1)
                if v < vertical_tiles - 1:
                    v_blend = torch.linspace(1, 0, overlap_out_h, device=decoded_tile.device)
                    tile_weights[:, :, -overlap_out_h:, :, :] *= v_blend.view(1, 1, -1, 1, 1)

                output[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += (decoded_tile * tile_weights).to(target_device, target_dtype)
                weights[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += tile_weights.to(target_device, target_dtype)

        output /= weights + 1e-8
        output = output.view(batch * image_frames, output_height, output_width, output.shape[-1])
        if last_frame_fix:
            output = output[:-time_scale_factor, :, :]
        return (output,)


class LTXVSpatioTemporalTiledVAEDecode_DirectEncode(LTXVTiledVAEDecode):
    """
    PRODUCTION-READY Direct MP4 Encoder with Audio Support

    Features:
    - Zero intermediate storage (direct ffmpeg pipe)
    - Correct temporal chunking with proper overlap calculation
    - Lossless audio support (FLAC)
    - Memory-efficient frame-by-frame processing
    - Comprehensive error handling
    - Memory management (CPU offload, aggressive cleanup)
    - Frame count verification
    - H.265/HEVC support with 10-bit color

    Optimizations Applied:
    1. Separate codec and pix_fmt parameters
    2. Proper H.265/libx265 support for 10-bit
    3. Frame-by-frame conversion to reduce memory overhead
    4. FLAC lossless audio encoding
    5. Proper temporal chunk boundaries with overlap
    6. Audio codec selection (FLAC/AAC-320k/AAC-192k)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "latents": ("LATENT",),
                "spatial_tiles": ("INT", {"default": 2, "min": 1, "max": 8}),
                "spatial_overlap": ("INT", {"default": 1, "min": 0, "max": 8}),
                "temporal_tile_length": ("INT", {"default": 30, "min": 2, "max": 1000}),
                "temporal_overlap": ("INT", {"default": 8, "min": 0, "max": 16}),
                "last_frame_fix": ("BOOLEAN", {"default": False}),
                "working_device": (["cpu", "auto"], {"default": "auto"}),
                "working_dtype": (["float16", "float32", "auto"], {"default": "auto"}),
                "decode_device": (["gpu", "cpu"], {"default": "cpu"}),
                "aggressive_cleanup": ("BOOLEAN", {"default": True}),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "codec": (["h264-mp4", "h265-mp4", "hevc-mp4"], {"default": "h265-mp4"}),
                "crf": ("INT", {"default": 20, "min": 0, "max": 51}),
                "preset": (["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], {"default": "slow"}),
                "pix_fmt": (["yuv420p", "yuv422p", "yuv444p", "yuv420p10le"], {"default": "yuv420p10le"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_codec": (["flac", "aac-320k", "aac-192k"], {"default": "flac"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "decode_to_video"
    CATEGORY = "latent"
    OUTPUT_NODE = True

    def decode_to_video(
        self,
        vae,
        latents,
        spatial_tiles=2,
        spatial_overlap=1,
        temporal_tile_length=30,
        temporal_overlap=8,
        last_frame_fix=False,
        working_device="auto",
        working_dtype="auto",
        decode_device="cpu",
        aggressive_cleanup=True,
        output_path="",
        fps=24,
        codec="h265-mp4",
        crf=20,
        preset="slow",
        pix_fmt="yuv420p10le",
        audio=None,
        audio_codec="flac",
    ):
        # Validation
        if temporal_tile_length < temporal_overlap + 1:
            raise ValueError(f"temporal_tile_length ({temporal_tile_length}) must be > temporal_overlap ({temporal_overlap}) + 1")

        # Preprocessing
        samples = latents["samples"]
        original_frames = None
        if last_frame_fix:
            original_frames = samples.shape[2]
            last_frame = samples[:, :, -1:, :, :]
            samples = torch.cat([samples, last_frame], dim=2)

        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = vae.downscale_index_formula

        # Calculate expected output
        if original_frames is not None:
            expected_output_frames = 1 + (original_frames - 1) * time_scale_factor
        else:
            expected_output_frames = 1 + (frames - 1) * time_scale_factor

        output_height = height * height_scale_factor
        output_width = width * width_scale_factor

        logging.info(f"[DirectEncode] Input: {frames} latent frames → Expected output: {expected_output_frames} frames")
        logging.info(f"[DirectEncode] time_scale_factor: {time_scale_factor}")

        # Setup output path with correct extension based on codec
        if not output_path:
            output_dir = Path(folder_paths.get_output_directory())
            file_ext = EXTENSION_MAP.get(codec, "mp4")
            timestamp = int(time.time())
            output_path = str(output_dir / f"ltxv_{timestamp}.{file_ext}")

        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Estimate size and check disk space
        bytes_per_pixel = 4 if pix_fmt == 'yuv444p' else (2.5 if pix_fmt == 'yuv420p10le' else 1.5)
        estimated_size_gb = (expected_output_frames * output_height * output_width * bytes_per_pixel * 0.1) / (1024**3)

        logging.info(f"[DirectEncode] Estimated output: ~{estimated_size_gb:.2f} GB")
        check_disk_space(output_path, estimated_size_gb)

        # Audio handling
        if audio is not None:
            temp_video = output_path.with_suffix('.temp.mp4')
            final_output = output_path
            video_encode_path = temp_video
            logging.info(f"[DirectEncode] Audio detected - will mux after encoding (codec: {audio_codec})")
        else:
            final_output = output_path
            video_encode_path = output_path

        logging.info(f"[DirectEncode] Output: {final_output}")
        logging.info(f"[DirectEncode] Resolution: {output_width}x{output_height} @ {fps} FPS")
        logging.info(f"[DirectEncode] Codec: {codec}, Quality: CRF {crf}, Preset: {preset}, Pix: {pix_fmt}")

        # Encode video
        try:
            total_encoded_frames = self._encode_video_stream(
                samples, vae, video_encode_path, output_height, output_width,
                time_scale_factor, fps, codec, crf, preset, pix_fmt,
                spatial_tiles, spatial_overlap, temporal_tile_length, temporal_overlap,
                decode_device, working_device, working_dtype, aggressive_cleanup,
                expected_output_frames
            )
        except Exception as e:
            logging.error(f"[DirectEncode] Encoding failed: {e}")
            if audio is not None and temp_video.exists():
                temp_video.unlink()
            raise

        # Verify frame count
        frame_diff = abs(total_encoded_frames - expected_output_frames)
        if frame_diff > 1:
            logging.warning(
                f"[DirectEncode] ⚠️  Frame count mismatch! "
                f"Expected {expected_output_frames}, got {total_encoded_frames} (diff: {frame_diff})"
            )
        else:
            logging.info(f"[DirectEncode] ✅ Frame count correct: {total_encoded_frames}")

        # Mux audio if present
        if audio is not None:
            try:
                self._mux_audio(temp_video, audio, final_output, fps, total_encoded_frames, audio_codec)
                temp_video.unlink()
                logging.info("[DirectEncode] Cleaned up temp video")
            except Exception as e:
                logging.error(f"[DirectEncode] Audio muxing failed: {e}")
                # Fallback: save video without audio
                if temp_video.exists():
                    shutil.copy(temp_video, final_output)
                    temp_video.unlink()
                    logging.warning("[DirectEncode] Saved video without audio")

        file_size_mb = final_output.stat().st_size / (1024**2)
        logging.info(f"[DirectEncode] ✅ Complete! {total_encoded_frames} frames, {file_size_mb:.2f} MB")

        return (str(final_output),)

    def _encode_video_stream(
        self, samples, vae, output_path, output_height, output_width,
        time_scale_factor, fps, codec, crf, preset, pix_fmt,
        spatial_tiles, spatial_overlap, temporal_tile_length, temporal_overlap,
        decode_device, working_device, working_dtype, aggressive_cleanup,
        expected_output_frames
    ):
        """Encode video stream with proper temporal chunking and memory-efficient frame processing"""

        # Generate and start ffmpeg
        ffmpeg_cmd = self._get_ffmpeg_command(
            output_width, output_height, fps, codec, crf, preset, pix_fmt, str(output_path)
        )

        logging.info(f"[DirectEncode] Starting ffmpeg: {' '.join(ffmpeg_cmd)}...")

        try:
            ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found! Install: sudo apt install ffmpeg")

        # Verify ffmpeg started successfully
        time.sleep(0.5)
        if ffmpeg_process.poll() is not None:
            _, stderr = ffmpeg_process.communicate()
            stderr_text = stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f"FFmpeg failed to start:\n{stderr_text}")

        logging.info("[DirectEncode] FFmpeg started successfully")

        # VAE device management
        vae_original_device = None
        if decode_device == "cpu":
            vae_original_device = next(vae.parameters()).device
            if vae_original_device.type != "cpu":
                logging.info(f"[DirectEncode] Moving VAE to CPU for decode")
                vae = vae.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Chunked decoding and streaming
        batch, channels, total_latent_frames, height, width = samples.shape
        chunk_start = 0
        chunk_idx = 0
        total_encoded_frames = 0

        try:
            while chunk_start < total_latent_frames:
                # Compute chunk boundaries
                latent_start, latent_end, frames_to_drop = compute_chunk_boundaries_FIXED(
                    chunk_start, temporal_tile_length, temporal_overlap, total_latent_frames
                )

                latent_frames = latent_end - latent_start

                logging.info(
                    f"[DirectEncode] Chunk {chunk_idx}: "
                    f"latent[{latent_start}:{latent_end}] ({latent_frames} frames), "
                    f"overlap {frames_to_drop} latent"
                )

                # Extract and decode chunk
                tile = samples[:, :, latent_start:latent_end]
                if decode_device == "cpu":
                    tile = tile.cpu()

                tile_latents = {"samples": tile}

                decoded_tile = self.decode(
                    vae=vae,
                    latents=tile_latents,
                    vertical_tiles=spatial_tiles,
                    horizontal_tiles=spatial_tiles,
                    overlap=spatial_overlap,
                    last_frame_fix=False,
                    working_device=working_device,
                    working_dtype=working_dtype,
                )[0]

                decoded_tile = decoded_tile.view(batch, -1, output_height, output_width, 3)
                decoded_frames = decoded_tile.shape[1]

                logging.info(f"[DirectEncode]   Decoded: {decoded_frames} frames")

                # Drop overlap frames with CORRECT calculation
                if frames_to_drop > 0:
                    decoded_frames_to_drop = compute_overlap_decoded_frames(
                        frames_to_drop, time_scale_factor
                    )

                    # Safety check
                    if decoded_frames_to_drop >= decoded_frames:
                        logging.warning(
                            f"[DirectEncode] Would drop {decoded_frames_to_drop} but only have {decoded_frames}!"
                        )
                        decoded_frames_to_drop = max(0, decoded_frames - 1)

                    decoded_tile = decoded_tile[:, decoded_frames_to_drop:]
                    logging.info(
                        f"[DirectEncode]   Dropped: {decoded_frames_to_drop} frames → "
                        f"{decoded_tile.shape[1]} remain"
                    )

                # Move to CPU and get frame count
                decoded_tile = decoded_tile.cpu()
                chunk_output_frames = decoded_tile.shape[1]

                # Write frames to ffmpeg ONE AT A TIME (memory efficient)
                for frame_idx in range(chunk_output_frames):
                    # Convert single frame to uint8
                    frame = decoded_tile[0, frame_idx].numpy()
                    frame_uint8 = (frame * 255).clip(0, 255).astype(np.uint8)
                    
                    try:
                        ffmpeg_process.stdin.write(frame_uint8.tobytes())
                        ffmpeg_process.stdin.flush()
                        total_encoded_frames += 1
                    except BrokenPipeError:
                        _, stderr = ffmpeg_process.communicate()
                        stderr_text = stderr.decode('utf-8', errors='ignore')
                        logging.error(f"[DirectEncode] ❌ FFmpeg pipe broken at frame {total_encoded_frames}")
                        logging.error(f"[DirectEncode] FFmpeg stderr:\n{stderr_text}")
                        raise RuntimeError(
                            f"FFmpeg encoding failed after {total_encoded_frames} frames:\n{stderr_text}"
                        )
                    except Exception as e:
                        logging.error(f"[DirectEncode] Error writing frame {total_encoded_frames}: {e}")
                        raise
                    
                    # Free memory immediately after each frame
                    del frame, frame_uint8

                logging.info(f"[DirectEncode]   Wrote: {chunk_output_frames} frames (total: {total_encoded_frames})")

                # Memory cleanup
                if aggressive_cleanup:
                    del tile, tile_latents, decoded_tile
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                chunk_start = latent_end
                chunk_idx += 1

        except Exception as e:
            try:
                ffmpeg_process.stdin.close()
            except:
                pass
            ffmpeg_process.terminate()
            raise

        finally:
            # Always restore VAE to original device
            if vae_original_device is not None and vae_original_device.type != "cpu":
                logging.info(f"[DirectEncode] Restoring VAE to {vae_original_device}")
                vae = vae.to(vae_original_device)
                if aggressive_cleanup and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Finalize ffmpeg
        try:
            ffmpeg_process.stdin.close()
        except:
            pass

        logging.info("[DirectEncode] Waiting for FFmpeg to finish...")

        stdout, stderr = ffmpeg_process.communicate()
        stderr_output = stderr.decode('utf-8', errors='ignore')

        if ffmpeg_process.returncode != 0:
            logging.error(f"[DirectEncode] FFmpeg exited with code {ffmpeg_process.returncode}")
            logging.error(f"[DirectEncode] FFmpeg stderr:\n{stderr_output}")
            raise RuntimeError(f"FFmpeg encoding failed (exit code {ffmpeg_process.returncode}):\n{stderr_output}")

        # Show last part of ffmpeg output for info
        if stderr_output:
            last_lines = '\n'.join(stderr_output.splitlines()[-5:])
            logging.info(f"[DirectEncode] FFmpeg output (last 5 lines):\n{last_lines}")

        return total_encoded_frames

    def _mux_audio(self, video_path, audio, output_path, fps, total_frames, audio_codec="flac"):
        """Mux audio with video with selectable codec (FLAC lossless or AAC)"""
        try:
            # Validate video file first
            logging.info("[DirectEncode] Validating video file before audio muxing...")

            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
                capture_output=True,
                timeout=10,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Video file validation failed: {result.stderr}")

            try:
                video_duration = float(result.stdout.strip())
                expected_duration = total_frames / fps

                logging.info(
                    f"[DirectEncode] Video duration: {video_duration:.2f}s "
                    f"(expected: {expected_duration:.2f}s)"
                )

                # Warn if significant mismatch
                if abs(video_duration - expected_duration) > 1.0:
                    logging.warning(
                        f"[DirectEncode] ⚠️  Duration mismatch > 1s "
                        f"(video: {video_duration:.2f}s, expected: {expected_duration:.2f}s)"
                    )
            except ValueError:
                logging.warning("[DirectEncode] Could not parse video duration")

            # Extract audio metadata
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']
            batch, channels, audio_samples = waveform.shape

            audio_duration = audio_samples / sample_rate

            logging.info(
                f"[DirectEncode] Muxing audio: {channels}ch @ {sample_rate}Hz, "
                f"duration {audio_duration:.2f}s, codec: {audio_codec}"
            )

            # Select audio encoding arguments based on codec
            if audio_codec == "flac":
                audio_args = ['-c:a', 'flac', '-compression_level', '8']
            elif audio_codec == "aac-320k":
                audio_args = ['-c:a', 'aac', '-b:a', '320k', '-q:a', '2']
            else:  # aac-192k
                audio_args = ['-c:a', 'aac', '-b:a', '192k']

            # FFmpeg mux command
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-f', 'f32le',
                '-i', 'pipe:0',
                '-c:v', 'copy',
            ] + audio_args + [
                '-shortest',
                str(output_path)
            ]

            logging.info("[DirectEncode] Starting audio muxing...")

            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Convert audio: [batch, channels, samples] → [samples, channels]
            audio_np = waveform[0].transpose(0, 1).numpy()
            audio_bytes = audio_np.tobytes()

            try:
                stdout, stderr = proc.communicate(input=audio_bytes, timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                raise RuntimeError("Audio muxing timeout (>60s)")

            if proc.returncode != 0:
                stderr_text = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"Audio muxing failed (exit {proc.returncode}):\n{stderr_text}")

            logging.info(f"[DirectEncode] Audio muxing complete ({audio_codec})")

        except subprocess.TimeoutExpired as e:
            logging.error(f"[DirectEncode] Timeout during audio muxing: {e}")
            raise
        except Exception as e:
            logging.error(f"[DirectEncode] Audio muxing error: {e}")
            raise

    def _get_ffmpeg_command(self, width, height, fps, codec, crf, preset, pix_fmt, output):
        """Generate ffmpeg encoding command with proper codec selection"""

        # Map codec parameter to ffmpeg encoder
        codec_map = {
            "h264-mp4": "libx264",
            "h265-mp4": "libx265",
            "hevc-mp4": "libx265",
        }
        
        encoder = codec_map.get(codec, "libx265")

        cmd = [
            'ffmpeg', '-y',
            # Input specification
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            # Input color space (sRGB)
            '-color_range', 'pc',
            '-colorspace', 'rgb',
            '-color_primaries', 'bt709',
            '-color_trc', 'iec61966-2-1',
            '-i', 'pipe:0',
            # Encoding settings
            '-c:v', encoder,
            '-crf', str(crf),
            '-preset', preset,
            '-pix_fmt', pix_fmt,
            '-movflags', '+faststart',
            # Output color space (BT.709 for video)
            '-colorspace', 'bt709',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            output
        ]

        return cmd


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LatentCutPlus": LatentCutPlus,
    "LTXVEmptyLatentAudioDebug": LTXVEmptyLatentAudioDebug,
    "LatentDebugInfo": LatentDebugInfo,
    "DebugAny": DebugAny,
    "LTXVTiledVAEDecode": LTXVTiledVAEDecode,
    "LTXVSpatioTemporalTiledVAEDecode_DirectEncode": LTXVSpatioTemporalTiledVAEDecode_DirectEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentCutPlus": "✂️ Latent Cut Plus",
    "LTXVEmptyLatentAudioDebug": "🔊 LTXV Empty Latent Audio (Debug)",
    "LatentDebugInfo": "📊 Latent Debug Info",
    "DebugAny": "🔍 Debug Any",
    "LTXVTiledVAEDecode": "🔲 LTXV Tiled VAE Decode",
    "LTXVSpatioTemporalTiledVAEDecode_DirectEncode": "🎬 Video Combine (LTXV Auto-Chunked)",
}
