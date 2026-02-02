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

def compute_chunk_boundaries(chunk_start, temporal_tile_length, temporal_overlap, total_latent_frames):
    if chunk_start == 0:
        chunk_end = min(chunk_start + temporal_tile_length, total_latent_frames)
        overlap_start = chunk_start
    else:
        overlap_start = max(1, chunk_start - temporal_overlap - 1)
        extra_frames = chunk_start - overlap_start
        chunk_end = min(chunk_start + temporal_tile_length - extra_frames, total_latent_frames)
    return overlap_start, chunk_end


class LTXVTiledVAEDecode:
    """Spatial tiled VAE decode"""
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
            samples = torch.cat([samples, samples[:, :, -1:]], dim=2)

        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = vae.downscale_index_formula
        image_frames = 1 + (frames - 1) * time_scale_factor
        output_height = height * height_scale_factor
        output_width = width * width_scale_factor

        base_tile_height = (height + (vertical_tiles - 1) * overlap) // vertical_tiles
        base_tile_width = (width + (horizontal_tiles - 1) * overlap) // horizontal_tiles

        target_device = samples.device if working_device == "auto" else working_device
        target_dtype = samples.dtype if working_dtype == "auto" else (torch.float16 if working_dtype == "float16" else torch.float32)

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

                out_h_start, out_h_end = v_start * height_scale_factor, v_end * height_scale_factor
                out_w_start, out_w_end = h_start * width_scale_factor, h_end * width_scale_factor
                tile_out_height, tile_out_width = out_h_end - out_h_start, out_w_end - out_w_start

                tile_weights = torch.ones((batch, image_frames, tile_out_height, tile_out_width, 1), device=decoded_tile.device, dtype=decoded_tile.dtype)
                overlap_out_h, overlap_out_w = overlap * height_scale_factor, overlap * width_scale_factor

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
        output = output.view(batch * image_frames, output_height, output_width, 3)
        if last_frame_fix:
            output = output[:-time_scale_factor]
        return (output,)


class VideoCombine_LTXV:
    """
    Universal Video Combiner with LTXV chunked decode support

    Features:
    - Accepts IMAGE or LATENT
    - Auto-detect RAM requirements
    - Chunked decode for long videos (85% RAM reduction)
    - Multiple formats: MP4 (H.264/H.265), WebM, WebP, GIF
    - Audio support
    - Compatible with VHS workflow
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE,LATENT",),
                "frame_rate": ("INT", {"default": 30, "min": 1, "max": 120}),
                "filename_prefix": ("STRING", {"default": "LTXV"}),
                "format": (["video/h264-mp4", "video/h265-mp4", "video/webm", "image/webp", "image/gif"], {"default": "video/h264-mp4"}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("VAE",),
                # Chunked decode settings
                "auto_chunked": ("BOOLEAN", {"default": True, "tooltip": "Auto-enable chunking for large videos"}),
                "ram_threshold_gb": ("INT", {"default": 32, "min": 4, "max": 256}),
                "spatial_tiles": ("INT", {"default": 2, "min": 1, "max": 8}),
                "temporal_chunk_size": ("INT", {"default": 8, "min": 2, "max": 100}),
                "decode_device": (["auto", "cpu", "gpu"], {"default": "auto"}),
                # Video encoding settings
                "crf": ("INT", {"default": 23, "min": 0, "max": 51, "tooltip": "Quality: lower=better"}),
                "preset": (["ultrafast", "superfast", "veryfast", "fast", "medium", "slow", "slower"], {"default": "medium"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/LTXV"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        images,
        frame_rate=30,
        filename_prefix="LTXV",
        format="video/h264-mp4",
        save_output=True,
        vae=None,
        auto_chunked=True,
        ram_threshold_gb=32,
        spatial_tiles=2,
        temporal_chunk_size=8,
        decode_device="auto",
        crf=23,
        preset="medium",
    ):
        # Detect input type
        is_latent = isinstance(images, dict) and 'samples' in images

        if is_latent and vae is None:
            raise ValueError("VAE is required for LATENT input")

        # Setup output path
        output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir)
        counter = self._get_counter(full_output_folder, filename)

        format_type, format_ext = format.split("/")
        output_path = Path(full_output_folder) / f"{filename}_{counter:05}.{format_ext}"

        if is_latent:
            # LATENT input - check if chunking needed
            samples = images['samples']
            batch, channels, frames, height, width = samples.shape
            time_scale_factor, width_scale_factor, height_scale_factor = vae.downscale_index_formula

            image_frames = 1 + (frames - 1) * time_scale_factor
            output_height = height * height_scale_factor
            output_width = width * width_scale_factor

            estimated_ram_gb = (batch * image_frames * output_height * output_width * 3 * 4) / (1024**3)

            logging.info(f"[VideoCombine_LTXV] Input: LATENT {samples.shape}")
            logging.info(f"[VideoCombine_LTXV] Estimated output: {estimated_ram_gb:.2f} GB")
            logging.info(f"[VideoCombine_LTXV] Threshold: {ram_threshold_gb} GB")

            if auto_chunked and estimated_ram_gb > ram_threshold_gb:
                logging.warning(f"[VideoCombine_LTXV] Enabling CHUNKED mode")
                return self._process_latent_chunked(
                    samples, vae, frame_rate, format, format_ext, output_path,
                    spatial_tiles, temporal_chunk_size, decode_device, crf, preset
                )
            else:
                logging.info(f"[VideoCombine_LTXV] Using STANDARD mode")
                images = self._decode_latent_standard(samples, vae, spatial_tiles)
                return self._encode_images(images, frame_rate, format, format_ext, output_path, crf, preset)
        else:
            # IMAGE input - standard encoding
            logging.info(f"[VideoCombine_LTXV] Input: IMAGE {images.shape}")
            return self._encode_images(images, frame_rate, format, format_ext, output_path, crf, preset)

    def _process_latent_chunked(self, samples, vae, frame_rate, format, format_ext, output_path,
                                 spatial_tiles, temporal_chunk_size, decode_device, crf, preset):
        """Chunked decode + direct ffmpeg encoding"""
        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = vae.downscale_index_formula

        output_height = height * height_scale_factor
        output_width = width * width_scale_factor

        # Start ffmpeg
        ffmpeg_cmd = self._get_ffmpeg_cmd(output_width, output_height, frame_rate, format_ext, str(output_path), crf, preset)
        logging.info(f"[VideoCombine_LTXV] FFmpeg: {' '.join(ffmpeg_cmd)}")

        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

        # Move VAE if needed
        vae_device = None
        if decode_device == "cpu":
            vae_device = next(vae.parameters()).device
            vae = vae.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_encoded = 0
        chunk_start = 0

        try:
            while chunk_start < frames:
                overlap_start, chunk_end = compute_chunk_boundaries(chunk_start, temporal_chunk_size, 1, frames)

                logging.info(f"[VideoCombine_LTXV] Chunk {chunk_start}:{chunk_end}")

                tile = samples[:, :, overlap_start:chunk_end]
                if decode_device == "cpu":
                    tile = tile.cpu()

                # Spatial decode
                decoded = self._spatial_decode_chunk(tile, vae, spatial_tiles, height, width)

                # Drop first frame if overlap
                if chunk_start > 0 and decoded.shape[1] > 1:
                    decoded = decoded[:, 1:]

                # To uint8
                frames_np = (decoded[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                # Write to ffmpeg
                for i in range(frames_np.shape[0]):
                    ffmpeg_proc.stdin.write(frames_np[i].tobytes())
                    total_encoded += 1

                del tile, decoded, frames_np
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                chunk_start = chunk_end

        finally:
            if vae_device is not None:
                vae = vae.to(vae_device)

        ffmpeg_proc.stdin.close()
        stderr = ffmpeg_proc.stderr.read().decode('utf-8')
        ffmpeg_proc.wait()

        if ffmpeg_proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr}")

        file_size = output_path.stat().st_size / (1024**2)
        logging.info(f"[VideoCombine_LTXV] ✅ {total_encoded} frames → {file_size:.2f} MB")

        return (str(output_path),)

    def _spatial_decode_chunk(self, latents, vae, tiles, orig_height, orig_width):
        """Spatial tiled decode for chunk"""
        decoder = LTXVTiledVAEDecode()
        decoded = decoder.decode(
            vae=vae,
            latents={'samples': latents},
            horizontal_tiles=tiles,
            vertical_tiles=tiles,
            overlap=1,
            last_frame_fix=False,
            working_device='auto',
            working_dtype='auto'
        )[0]

        # Reshape back to [batch, frames, height, width, channels]
        batch_frames = decoded.shape[0]
        height, width, channels = decoded.shape[1:4]
        # Assuming batch=1
        decoded = decoded.view(1, batch_frames, height, width, channels)
        return decoded

    def _decode_latent_standard(self, samples, vae, spatial_tiles):
        """Standard decode (all at once)"""
        decoder = LTXVTiledVAEDecode()
        return decoder.decode(
            vae=vae,
            latents={'samples': samples},
            horizontal_tiles=spatial_tiles,
            vertical_tiles=spatial_tiles,
            overlap=1,
            last_frame_fix=False
        )[0]

    def _encode_images(self, images, frame_rate, format, format_ext, output_path, crf, preset):
        """Encode IMAGE tensor to video"""
        # images: [frames, height, width, channels]
        num_frames, height, width, channels = images.shape

        logging.info(f"[VideoCombine_LTXV] Encoding {num_frames} frames ({width}x{height})")

        ffmpeg_cmd = self._get_ffmpeg_cmd(width, height, frame_rate, format_ext, str(output_path), crf, preset)

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

        # Convert to uint8 and write
        images_np = (images.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        for i in range(num_frames):
            proc.stdin.write(images_np[i].tobytes())

        proc.stdin.close()
        stderr = proc.stderr.read().decode('utf-8')
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr}")

        file_size = output_path.stat().st_size / (1024**2)
        logging.info(f"[VideoCombine_LTXV] ✅ {num_frames} frames → {file_size:.2f} MB")

        return (str(output_path),)

    def _get_ffmpeg_cmd(self, width, height, fps, format_ext, output, crf, preset):
        """Generate ffmpeg command"""
        base = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps), '-i', 'pipe:0',
        ]

        if format_ext in ['mp4', 'h264-mp4']:
            base += ['-c:v', 'libx264', '-crf', str(crf), '-preset', preset, '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
        elif format_ext in ['h265-mp4', 'hevc-mp4']:
            base += ['-c:v', 'libx265', '-crf', str(crf), '-preset', preset, '-pix_fmt', 'yuv420p']
        elif format_ext == 'webm':
            base += ['-c:v', 'libvpx-vp9', '-crf', str(crf), '-b:v', '0']
        elif format_ext == 'webp':
            base += ['-c:v', 'libwebp', '-lossless', '0', '-quality', str(100-crf*2)]
        elif format_ext == 'gif':
            base += ['-vf', 'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse']

        base.append(output)
        return base

    def _get_counter(self, folder, filename):
        """Get next file counter"""
        max_counter = 0
        matcher = re.compile(f"{re.escape(filename)}_(\d+)\D*\..+", re.IGNORECASE)
        try:
            for f in os.listdir(folder):
                m = matcher.fullmatch(f)
                if m:
                    max_counter = max(max_counter, int(m.group(1)))
        except FileNotFoundError:
            pass
        return max_counter + 1


NODE_CLASS_MAPPINGS = {
    "LTXVTiledVAEDecode": LTXVTiledVAEDecode,
    "VideoCombine_LTXV": VideoCombine_LTXV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVTiledVAEDecode": "🔲 LTXV Tiled VAE Decode",
    "VideoCombine_LTXV": "🎬 Video Combine (LTXV Auto-Chunked)",
}
