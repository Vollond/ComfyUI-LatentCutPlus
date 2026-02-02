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



# ============================================================================
# ANYTYPE CLASS (for universal debug node)
# ============================================================================

class AnyType(str):
    """Special class for representing any type - always returns True on type comparison"""
    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")


# ============================================================================
# LATENT CUT PLUS (old API)
# ============================================================================
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


# ============================================================================
# LATENT DEBUG INFO (old API)
# ============================================================================

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


# ============================================================================
# DEBUG ANY (old API)
# ============================================================================

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

def compute_chunk_boundaries(chunk_start, temporal_tile_length, temporal_overlap, total_latent_frames):
    if chunk_start == 0:
        chunk_end = min(chunk_start + temporal_tile_length, total_latent_frames)
        overlap_start = chunk_start
    else:
        overlap_start = max(1, chunk_start - temporal_overlap - 1)
        extra_frames = chunk_start - overlap_start
        chunk_end = min(chunk_start + temporal_tile_length - extra_frames, total_latent_frames)
    return overlap_start, chunk_end


# Extension mapping for file names
EXTENSION_MAP = {
    "h264-mp4": "mp4",
    "h265-mp4": "mp4",
    "hevc-mp4": "mp4",
    "webm": "webm",
    "gif": "gif",
    "webp": "webp",
}


def check_disk_space(output_path, estimated_size_gb):
    """Check if enough disk space available"""
    output_dir = Path(output_path).parent
    try:
        stat = shutil.disk_usage(output_dir)
        free_gb = stat.free / (1024**3)

        if free_gb < estimated_size_gb * 1.2:  # 20% buffer
            raise RuntimeError(
                f"Insufficient disk space! Need ~{estimated_size_gb:.1f} GB, "
                f"only {free_gb:.1f} GB available in {output_dir}"
            )

        logging.info(f"[DirectEncode] Disk space check: {free_gb:.1f} GB free")
        return free_gb
    except Exception as e:
        logging.warning(f"[DirectEncode] Could not check disk space: {e}")
        return None


class LTXVTiledVAEDecode:
    """Base spatial tiled decoder (same as before)"""
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
    FIXED VERSION with proper error handling

    Fixes:
    - Correct file extension mapping (h265-mp4 → .mp4)
    - Early FFmpeg health check
    - Better BrokenPipeError handling
    - Disk space verification
    - Detailed error logging
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "latents": ("LATENT",),
                "spatial_tiles": ("INT", {"default": 2, "min": 1, "max": 8}),
                "spatial_overlap": ("INT", {"default": 1, "min": 0, "max": 8}),
                "temporal_tile_length": ("INT", {"default": 8, "min": 2, "max": 1000}),
                "temporal_overlap": ("INT", {"default": 1, "min": 0, "max": 8}),
                "last_frame_fix": ("BOOLEAN", {"default": False}),
                "working_device": (["cpu", "auto"], {"default": "auto"}),
                "working_dtype": (["float16", "float32", "auto"], {"default": "auto"}),
                "decode_device": (["gpu", "cpu"], {"default": "cpu"}),
                "aggressive_cleanup": ("BOOLEAN", {"default": True}),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120}),
                "crf": ("INT", {"default": 23, "min": 0, "max": 51}),
                "preset": (["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], {"default": "medium"}),
                "pix_fmt": (["yuv420p", "yuv422p", "yuv444p", "yuv420p10le"], {"default": "yuv420p"}),
            },
            "optional": {
                "audio": ("AUDIO",),
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
        temporal_tile_length=8,
        temporal_overlap=1,
        last_frame_fix=False,
        working_device="auto",
        working_dtype="auto",
        decode_device="cpu",
        aggressive_cleanup=True,
        output_path="",
        fps=30,
        crf=23,
        preset="medium",
        pix_fmt="yuv420p",
        audio=None,
    ):
        if temporal_tile_length < temporal_overlap + 1:
            raise ValueError("Temporal tile length must be greater than temporal overlap + 1")

        # Preprocessing
        samples = latents["samples"]
        original_frames = None
        if last_frame_fix:
            original_frames = samples.shape[2]
            last_frame = samples[:, :, -1:, :, :]
            samples = torch.cat([samples, last_frame], dim=2)

        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = vae.downscale_index_formula

        image_frames = 1 + (frames - 1) * time_scale_factor
        output_height = height * height_scale_factor
        output_width = width * width_scale_factor

        # Estimate output size
        bytes_per_pixel = 4 if pix_fmt in ['yuv444p', 'rgb24'] else (2.5 if pix_fmt == 'yuv420p10le' else 1.5)
        estimated_size_gb = (image_frames * output_height * output_width * bytes_per_pixel * 0.1) / (1024**3)  # ~10% of raw for CRF 20-25

        logging.info(f"[DirectEncode] Estimated output: ~{estimated_size_gb:.2f} GB")

        # Setup output path with CORRECT extension
        if not output_path:
            output_dir = Path(folder_paths.get_output_directory())
            # Use proper file extension
            file_ext = EXTENSION_MAP.get(pix_fmt, "mp4")
            output_path = str(output_dir / f"ltxv_output_{os.getpid()}.{file_ext}")
        else:
            output_path = str(Path(output_path))

        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Check disk space
        check_disk_space(output_path, estimated_size_gb)

        # If audio present, encode to temp file first
        if audio is not None:
            temp_video = output_path.with_suffix('.temp.mp4')
            final_output = output_path
            video_encode_path = temp_video
        else:
            final_output = output_path
            video_encode_path = output_path

        logging.info(f"[DirectEncode] Output: {final_output}")
        logging.info(f"[DirectEncode] Resolution: {output_width}x{output_height}, FPS: {fps}")
        logging.info(f"[DirectEncode] Quality: CRF {crf}, Preset: {preset}, Pix: {pix_fmt}")

        target_device = samples.device if working_device == "auto" else working_device
        if working_dtype == "auto":
            target_dtype = samples.dtype
        elif working_dtype == "float16":
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32

        # Encode video
        try:
            total_encoded_frames = self._encode_video_stream(
                samples, vae, video_encode_path, output_height, output_width,
                time_scale_factor, fps, crf, preset, pix_fmt,
                spatial_tiles, spatial_overlap, temporal_tile_length, temporal_overlap,
                decode_device, working_device, working_dtype, aggressive_cleanup,
                last_frame_fix, original_frames, target_device, target_dtype
            )
        except Exception as e:
            logging.error(f"[DirectEncode] Encoding failed: {e}")
            # Clean up temp file if exists
            if audio is not None and temp_video.exists():
                temp_video.unlink()
            raise

        # Mux audio if present
        if audio is not None:
            try:
                self._mux_audio(temp_video, audio, final_output, fps, total_encoded_frames)
                # Clean up temp video
                temp_video.unlink()
                logging.info(f"[DirectEncode] Cleaned up temp video")
            except Exception as e:
                logging.error(f"[DirectEncode] Audio muxing failed: {e}")
                # At least copy video without audio
                if temp_video.exists():
                    shutil.copy(temp_video, final_output)
                    temp_video.unlink()
                    logging.warning("[DirectEncode] Saved video without audio")

        file_size_mb = final_output.stat().st_size / (1024**2)
        logging.info(f"[DirectEncode] ✅ Complete!")
        logging.info(f"[DirectEncode] Frames: {total_encoded_frames}")
        logging.info(f"[DirectEncode] Size: {file_size_mb:.2f} MB")
        logging.info(f"[DirectEncode] Path: {final_output}")

        return (str(final_output),)

    def _encode_video_stream(
        self, samples, vae, output_path, output_height, output_width,
        time_scale_factor, fps, crf, preset, pix_fmt,
        spatial_tiles, spatial_overlap, temporal_tile_length, temporal_overlap,
        decode_device, working_device, working_dtype, aggressive_cleanup,
        last_frame_fix, original_frames, target_device, target_dtype
    ):
        """Encode video stream with robust error handling"""

        # Generate ffmpeg command
        ffmpeg_cmd = self._get_ffmpeg_command(
            output_width, output_height, fps, crf, preset, pix_fmt, str(output_path)
        )

        logging.info(f"[DirectEncode] FFmpeg command:")
        logging.info(f"  {' '.join(ffmpeg_cmd[:15])}...")

        # Start ffmpeg process
        try:
            ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found! Please install: sudo apt install ffmpeg")

        # Wait a bit and check if ffmpeg started successfully
        time.sleep(0.5)
        if ffmpeg_process.poll() is not None:
            # FFmpeg already exited - something is wrong
            stderr = ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
            raise RuntimeError(f"FFmpeg failed to start:\n{stderr}")

        logging.info("[DirectEncode] FFmpeg started successfully")

        # VAE device management
        vae_original_device = None
        if decode_device == "cpu":
            vae_original_device = next(vae.parameters()).device
            if vae_original_device.type != "cpu":
                logging.info(f"[DirectEncode] Moving VAE to CPU")
                vae = vae.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Decode and stream to ffmpeg
        batch, channels, total_latent_frames, height, width = samples.shape
        chunk_start = 0
        chunk_idx = 0
        total_encoded_frames = 0

        try:
            while chunk_start < total_latent_frames:
                overlap_start, chunk_end = compute_chunk_boundaries(
                    chunk_start, temporal_tile_length, temporal_overlap, total_latent_frames
                )

                logging.info(f"[DirectEncode] Chunk {chunk_idx}: latent {overlap_start}:{chunk_end}")

                # Decode chunk
                tile = samples[:, :, overlap_start:chunk_end]
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

                # Drop overlapping frame
                if chunk_start > 0 and decoded_tile.shape[1] > 1:
                    decoded_tile = decoded_tile[:, 1:]

                # Convert to uint8 RGB
                decoded_tile = decoded_tile.cpu()
                decoded_np = (decoded_tile[0].numpy() * 255).clip(0, 255).astype(np.uint8)

                chunk_frames = decoded_np.shape[0]

                # Write frames to ffmpeg stdin with error handling
                for frame_idx in range(chunk_frames):
                    frame = decoded_np[frame_idx]
                    try:
                        ffmpeg_process.stdin.write(frame.tobytes())
                        ffmpeg_process.stdin.flush()
                        total_encoded_frames += 1
                    except BrokenPipeError:
                        # FFmpeg closed pipe - get error
                        stderr = ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                        logging.error(f"[DirectEncode] ❌ FFmpeg pipe broken at frame {total_encoded_frames}!")
                        logging.error(f"[DirectEncode] FFmpeg stderr:\n{stderr}")
                        raise RuntimeError(f"FFmpeg encoding failed after {total_encoded_frames} frames:\n{stderr}")
                    except Exception as e:
                        logging.error(f"[DirectEncode] Error writing frame {total_encoded_frames}: {e}")
                        raise

                logging.info(f"[DirectEncode] Wrote {chunk_frames} frames (total: {total_encoded_frames})")

                # Aggressive cleanup
                if aggressive_cleanup:
                    del tile, tile_latents, decoded_tile, decoded_np
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                chunk_start = chunk_end
                chunk_idx += 1

        except Exception as e:
            # Clean up on error
            try:
                ffmpeg_process.stdin.close()
            except:
                pass
            ffmpeg_process.terminate()
            ffmpeg_process.wait()
            raise

        finally:
            # Restore VAE
            if vae_original_device is not None and vae_original_device.type != "cpu":
                logging.info(f"[DirectEncode] Restoring VAE to {vae_original_device}")
                vae = vae.to(vae_original_device)
                if aggressive_cleanup and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Close ffmpeg stdin and wait for completion
        try:
            ffmpeg_process.stdin.close()
        except:
            pass

        logging.info("[DirectEncode] Waiting for FFmpeg to finish...")

        stderr_output = ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
        return_code = ffmpeg_process.wait()

        if return_code != 0:
            logging.error(f"[DirectEncode] FFmpeg exited with code {return_code}")
            logging.error(f"[DirectEncode] FFmpeg stderr:\n{stderr_output}")
            raise RuntimeError(f"FFmpeg encoding failed (exit code {return_code}):\n{stderr_output}")

        if stderr_output:
            logging.info(f"[DirectEncode] FFmpeg output:\n{stderr_output[-500:]}")  # Last 500 chars

        return total_encoded_frames

    def _mux_audio(self, video_path, audio, output_path, fps, total_frames):
        """Mux audio with video"""
        try:
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']
            batch, channels, audio_samples = waveform.shape

            video_duration = total_frames / fps

            logging.info(f"[DirectEncode] Muxing audio: {channels}ch @ {sample_rate}Hz, duration {video_duration:.2f}s")

            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-f', 'f32le',
                '-i', 'pipe:0',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                str(output_path)
            ]

            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            audio_np = waveform.squeeze(0).transpose(0, 1).numpy()
            audio_bytes = audio_np.tobytes()

            proc.stdin.write(audio_bytes)
            proc.stdin.close()

            stderr = proc.stderr.read().decode('utf-8', errors='ignore')
            return_code = proc.wait()

            if return_code != 0:
                raise RuntimeError(f"Audio muxing failed (exit code {return_code}):\n{stderr}")

            logging.info("[DirectEncode] Audio muxing complete")

        except Exception as e:
            logging.error(f"[DirectEncode] Audio muxing error: {e}")
            raise

    def _get_ffmpeg_command(self, width, height, fps, crf, preset, pix_fmt, output):
        """Generate ffmpeg command"""

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-color_range', 'pc',
            '-colorspace', 'rgb',
            '-color_primaries', 'bt709',
            '-color_trc', 'iec61966-2-1',
            '-i', 'pipe:0',
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', preset,
            '-pix_fmt', pix_fmt,
            '-movflags', '+faststart',
            '-colorspace', 'bt709',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            output
        ]

        return cmd


NODE_CLASS_MAPPINGS = {
    "LTXVTiledVAEDecode": LTXVTiledVAEDecode,
    "LTXVSpatioTemporalTiledVAEDecode_DirectEncode": LTXVSpatioTemporalTiledVAEDecode_DirectEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVTiledVAEDecode": "🔲 LTXV Tiled VAE Decode",
    "LTXVSpatioTemporalTiledVAEDecode_DirectEncode": "🎬 LTXV Direct MP4 (FIXED)",
}
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
    "LTXVSpatioTemporalTiledVAEDecode_MemOptimized_v2": "LTXVSpatioTemporalTiledVAEDecode_MemOptimized_v2",
    "LTXVTiledVAEDecode": "🔲 LTXV Tiled VAE Decode",
    "LTXVSpatioTemporalTiledVAEDecode_DirectEncode": "🎬 Video Combine (LTXV Auto-Chunked)",
}
