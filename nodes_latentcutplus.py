import logging
import torch
from comfy_api.latest import io


class LatentCutPlus(io.ComfyNode):
    """Slice latent tensor along a dimension (t/x/y) with smart index/amount handling."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentCutPlus",
            display_name="Latent Cut Plus",
            category="latent",
            description="Slice latent tensor along t/x/y dimension with overflow protection",
            inputs=[
                io.Latent.Input("samples"),
                io.Enum.Input(
                    "dim",
                    choices=["t", "x", "y"],
                    default="t",
                    tooltip="Dimension to slice: t=temporal, x=width, y=height",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=-9999,
                    max=9999,
                    tooltip="Start index (supports negative indexing)",
                ),
                io.Int.Input(
                    "amount",
                    default=1,
                    min=1,
                    max=2147483647,
                    tooltip="Number of frames/slices to keep (or use large value to slice to end)",
                ),
            ],
            outputs=[
                io.Latent.Output("result"),
            ],
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


class LatentCutPlusExtension(io.ComfyAPIExtension):
    """Extension that registers LatentCutPlus node."""

    @classmethod
    def get_nodes(cls):
        return [LatentCutPlus]


# ============================================================================
# OLD API DEBUG NODE (accepts any type)
# ============================================================================

class AnyType(str):
    """Special class for representing any type - always returns True on type comparison"""
    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")


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


# Export for old API registration
NODE_CLASS_MAPPINGS = {
    "DebugAny": DebugAny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DebugAny": "🔍 Debug Any",
}
