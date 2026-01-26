from __future__ import annotations

import torch
from comfy_api.latest import ComfyExtension, io
import nodes
import logging

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
        logging.info(f"[LatentCutPlus] Params: index={index}, amount={amount}")
        
        # Python-style negative index handling
        original_index = index
        if index < 0:
            index = size + index
            logging.info(f"[LatentCutPlus] Negative index {original_index} → normalized to {index}")
        
        start = max(0, index)
        amount = max(1, int(amount))
        end = start + amount

        logging.info(f"[LatentCutPlus] Slice: [{start}:{end}] (length={end-start})")

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


class LatentCutPlusExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LatentCutPlus]
