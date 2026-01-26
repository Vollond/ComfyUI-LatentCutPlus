from __future__ import annotations

import torch
from comfy_api.latest import ComfyExtension, io
import nodes

_INT_MAX = 2_147_483_647


class LatentCutPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentCutPlus",
            display_name="Latent Cut Plus",
            search_aliases=["crop latent", "slice latent", "extract region", "latentcut plus"],
            category="latent/advanced",
            description="Extended LatentCut with larger range and amount=-1 (to end) support.",
            inputs=[
                io.Latent.Input("samples"),
                io.Combo.Input("dim", options=["x", "y", "t"]),
                io.Int.Input("index", default=0, min=-_INT_MAX, max=_INT_MAX, step=1),
                io.Int.Input("amount", default=1, min=-1, max=_INT_MAX, step=1),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, samples, dim: str, index: int, amount: int) -> io.NodeOutput:
        samples_out = samples.copy()
        s1 = samples["samples"]
        
        # Map dimension name to tensor axis (same as original)
        if "x" in dim:
            dim_axis = s1.ndim - 1
        elif "y" in dim:
            dim_axis = s1.ndim - 2
        elif "t" in dim:
            dim_axis = s1.ndim - 3
        else:
            dim_axis = s1.ndim - 1  # fallback
        
        # ✅ EXACT logic from original LatentCut
        if index >= 0:
            # Clamp index to valid range
            index = min(index, s1.shape[dim_axis] - 1)
            
            # Special: amount=-1 means "to end"
            if amount == -1:
                amount = s1.shape[dim_axis] - index
            else:
                # Clamp amount so we don't exceed tensor size
                amount = min(s1.shape[dim_axis] - index, amount)
        else:
            # Negative index: clamp to valid negative range
            index = max(index, -s1.shape[dim_axis])
            
            # Special: amount=-1 means "to end from negative index"
            if amount == -1:
                amount = -index
            else:
                # Original logic: amount limited by distance from start
                amount = min(-index, amount)
        
        # Use torch.narrow (same as original)
        samples_out["samples"] = torch.narrow(s1, dim_axis, index, amount)
        
        return io.NodeOutput(samples_out)


class LatentCutPlusExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LatentCutPlus]
