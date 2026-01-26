from __future__ import annotations

import torch
from comfy_api.latest import ComfyExtension, io
import nodes
import logging
import comfy.model_management

# Try to import AudioVAE, handle if not available
try:
    from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
    AUDIO_VAE_AVAILABLE = True
except ImportError:
    AUDIO_VAE_AVAILABLE = False
    logging.warning("[LatentCutPlus] AudioVAE not available - LTXVEmptyLatentAudioDebug will be disabled")

_INT_MAX = 2_147_483_647


class LatentCutPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentCutPlus",
            display_name="Latent Cut Plus",
            search_aliases=["crop latent", "slice latent", "extract region", "latentcut plus"],
            category="latent/advanced",
            description="Slice latents along x/y/t with extended range and Python-style negative indexing.",
            inputs=[
                io.Latent.Input("samples"),
                io.Combo.Input("dim", options=["x", "y", "t"]),
                io.Int.Input("index", default=0, min=-_INT_MAX, max=_INT_MAX, step=1),
                io.Int.Input("amount", default=1, min=1, max=_INT_MAX, step=1),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, samples, dim: str, index: int, amount: int) -> io.NodeOutput:
        out = samples.copy()

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
            return io.NodeOutput(out)
        
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

        # Handle noise_mask if present
        if "noise_mask" in out and isinstance(out["noise_mask"], torch.Tensor):
            nm: torch.Tensor = out["noise_mask"]
            if nm.ndim == x.ndim and int(nm.shape[axis]) == size:
                out["noise_mask"] = nm[tuple(sl)].contiguous()
                logging.info(f"[LatentCutPlus] Noise mask sliced: {tuple(out['noise_mask'].shape)}")

        return io.NodeOutput(out)


class LatentDebugInfo(io.ComfyNode):
    """Debug node to inspect latent tensor information."""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentDebugInfo",
            display_name="Latent Debug Info",
            search_aliases=["debug latent", "inspect latent", "latent info"],
            category="latent/advanced",
            description="Display detailed information about latent tensor (shape, metadata, statistics).",
            inputs=[
                io.Latent.Input("samples"),
                io.String.Input(
                    "label",
                    default="",
                    multiline=False,
                    tooltip="Custom label to identify this debug point in logs (e.g., 'after_concat', 'before_mask')",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="passthrough"),
            ],
        )

    @classmethod
    def execute(cls, samples, label: str = "") -> io.NodeOutput:
        if "samples" not in samples:
            logging.error(f"[LatentDebugInfo:{label or 'UNLABELED'}] No 'samples' key in latent dict")
            return io.NodeOutput(samples)
        
        x: torch.Tensor = samples["samples"]
        
        # Create identifier for logs
        log_id = f"[LatentDebugInfo:{label}]" if label else "[LatentDebugInfo]"
        
        # Log comprehensive info
        logging.info("=" * 80)
        logging.info(f"{log_id} LATENT TENSOR INFORMATION")
        logging.info("=" * 80)
        logging.info(f"{log_id} Shape: {tuple(x.shape)}")
        logging.info(f"{log_id} Dtype: {x.dtype}")
        logging.info(f"{log_id} Device: {x.device}")
        logging.info(f"{log_id} Total elements: {x.numel()}")
        logging.info(f"{log_id} Memory (MB): {x.element_size() * x.numel() / 1024 / 1024:.2f}")
        
        # Statistics
        logging.info(f"{log_id} Min value: {x.min().item():.6f}")
        logging.info(f"{log_id} Max value: {x.max().item():.6f}")
        logging.info(f"{log_id} Mean value: {x.mean().item():.6f}")
        logging.info(f"{log_id} Std value: {x.std().item():.6f}")
        
        # Metadata
        logging.info(f"{log_id} Metadata keys in latent dict:")
        for key, value in samples.items():
            if key != "samples":
                logging.info(f"{log_id}   - {key}: {value}")
        
        logging.info("=" * 80)
        
        # Passthrough
        return io.NodeOutput(samples)


if AUDIO_VAE_AVAILABLE:
    class LTXVEmptyLatentAudioDebug(io.ComfyNode):
        """Debug version of LTXVEmptyLatentAudio with detailed logging."""
        
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="LTXVEmptyLatentAudioDebug",
                display_name="LTXV Empty Latent Audio (Debug)",
                search_aliases=["audio latent", "empty audio", "ltxv audio debug"],
                category="latent/audio",
                description="Generate empty audio latents with diagnostic logging to track size issues.",
                inputs=[
                    io.Int.Input(
                        "frames_number",
                        default=97,
                        min=1,
                        max=10000,
                        step=1,
                        display_mode=io.NumberDisplay.number,
                        tooltip="Number of frames.",
                    ),
                    io.Int.Input(
                        "frame_rate",
                        default=25,
                        min=1,
                        max=1000,
                        step=1,
                        display_mode=io.NumberDisplay.number,
                        tooltip="Number of frames per second.",
                    ),
                    io.Int.Input(
                        "batch_size",
                        default=1,
                        min=1,
                        max=4096,
                        display_mode=io.NumberDisplay.number,
                        tooltip="The number of latent audio samples in the batch.",
                    ),
                    io.Vae.Input(
                        id="audio_vae",
                        display_name="Audio VAE",
                        tooltip="The Audio VAE model to get configuration from.",
                    ),
                ],
                outputs=[io.Latent.Output(display_name="Latent")],
            )

        @classmethod
        def execute(
            cls,
            frames_number: int,
            frame_rate: int,
            batch_size: int,
            audio_vae: AudioVAE,
        ) -> io.NodeOutput:
            """Generate empty audio latents with diagnostic logging."""
            assert audio_vae is not None, "Audio VAE model is required"
            
            z_channels = audio_vae.latent_channels
            audio_freq = audio_vae.latent_frequency_bins
            sampling_rate = int(audio_vae.sample_rate)
            
            # Detailed logging
            logging.info("=" * 80)
            logging.info("[LTXVEmptyLatentAudioDebug] GENERATION START")
            logging.info("=" * 80)
            logging.info(f"INPUT PARAMETERS:")
            logging.info(f"  frames_number = {frames_number}")
            logging.info(f"  frame_rate = {frame_rate}")
            logging.info(f"  batch_size = {batch_size}")
            logging.info(f"AUDIO VAE CONFIG:")
            logging.info(f"  latent_channels (z_channels) = {z_channels}")
            logging.info(f"  latent_frequency_bins (audio_freq) = {audio_freq}")
            logging.info(f"  sample_rate = {sampling_rate}")
            
            # Calculate audio latents count using AudioVAE method
            logging.info(f"CALLING: audio_vae.num_of_latents_from_frames({frames_number}, {frame_rate})")
            num_audio_latents = audio_vae.num_of_latents_from_frames(frames_number, frame_rate)
            logging.info(f"RESULT: num_audio_latents = {num_audio_latents}")
            
            # Check if result makes sense
            expected_approx = frames_number  # Rough estimate
            if abs(num_audio_latents - expected_approx) > expected_approx * 0.5:
                logging.warning(f"⚠️  SUSPICIOUS VALUE! Expected ~{expected_approx}, got {num_audio_latents}")
                logging.warning(f"⚠️  Difference: {num_audio_latents - expected_approx} ({(num_audio_latents/expected_approx - 1)*100:.1f}% larger)")
            
            # Create tensor
            audio_latents = torch.zeros(
                (batch_size, z_channels, num_audio_latents, audio_freq),
                device=comfy.model_management.intermediate_device(),
            )
            
            logging.info(f"OUTPUT TENSOR:")
            logging.info(f"  Shape: {tuple(audio_latents.shape)}")
            logging.info(f"  Dtype: {audio_latents.dtype}")
            logging.info(f"  Device: {audio_latents.device}")
            logging.info(f"  Memory (MB): {audio_latents.element_size() * audio_latents.numel() / 1024 / 1024:.2f}")
            logging.info("=" * 80)
            
            return io.NodeOutput(
                {
                    "samples": audio_latents,
                    "sample_rate": sampling_rate,
                    "type": "audio",
                }
            )


class LatentCutPlusExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        nodes_list = [LatentCutPlus, LatentDebugInfo]
        
        # Add audio debug node only if AudioVAE is available
        if AUDIO_VAE_AVAILABLE:
            nodes_list.append(LTXVEmptyLatentAudioDebug)
            logging.info("[LatentCutPlus] Registered LTXVEmptyLatentAudioDebug node")
        
        return nodes_list
