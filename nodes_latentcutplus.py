from __future__ import annotations

import torch
from comfyapi.latest import ComfyExtension, io
import nodes

_INT_MAX = 2_147_483_647  # big enough for UI; runtime clamps to tensor size anyway

class LatentCutPlusExtension(ComfyExtension):
    async def get_node_list(self):
        return [LatentCutPlus, LatentCutPlusAdvanced] 
def _normalize_index(idx: int, size: int) -> int:
    # Python-style: negative indices count from end.
    if size <= 0:
        return 0
    if idx < 0:
        idx = size + idx
    return max(0, min(idx, size))


def _slice_tensor_along_dim(x: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    # Build an index tuple like [:, :, start:end, :] etc.
    sl = [slice(None)] * x.ndim
    sl[dim] = slice(start, end)
    return x[tuple(sl)]


class LatentCutPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentCutPlus",
            display_name="Latent Cut Plus",
            search_aliases=["crop latent", "slice latent", "extract region", "latentcut plus"],
            category="latent/advanced",
            description="Slice latents along x/y/t with Python-style indexing; supports amount=-1 (to end).",
            inputs=[
                io.Latent.Input("samples"),
                io.Combo.Input("dim", options=["x", "y", "t"]),
                io.Int.Input("index", default=0, min=-_INT_MAX, max=_INT_MAX, step=1),
                io.Int.Input("amount", default=-1, min=-1, max=_INT_MAX, step=1),
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

        # Map dim -> axis (follow the convention from your previous LatentCut: x=-1, y=-2, t=-3).
        if dim == "x":
            axis = x.ndim - 1
        elif dim == "y":
            axis = x.ndim - 2
        else:  # "t"
            axis = x.ndim - 3

        size = int(x.shape[axis])

        # Normalize start like Python indexing.
        start = _normalize_index(index, size)

        # amount semantics:
        #  -1 => slice to end
        #  >=1 => fixed length
        if amount == -1:
            end = size
        else:
            amount = max(1, int(amount))
            end = min(size, start + amount)

        # Ensure non-empty slice (optional; but keep stable behavior)
        end = max(start, end)

        out_tensor = _slice_tensor_along_dim(x, axis, start, end).contiguous()
        out["samples"] = out_tensor

        # If there is a noise mask with the same shape along that axis, slice it too.
        if "noise_mask" in out and isinstance(out["noise_mask"], torch.Tensor):
            nm: torch.Tensor = out["noise_mask"]
            if nm.ndim == x.ndim and int(nm.shape[axis]) == size:
                out["noise_mask"] = _slice_tensor_along_dim(nm, axis, start, end).contiguous()

        return io.NodeOutput(out)


class LatentCutPlusExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LatentCutPlus]
