import logging
import torch


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
    
    def execute(self, samples, dim: str, index: int, amount: int):
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

        # Handle noise_mask if present
        if "noise_mask" in out and isinstance(out["noise_mask"], torch.Tensor):
            nm: torch.Tensor = out["noise_mask"]
            if nm.ndim == x.ndim and int(nm.shape[axis]) == size:
                out["noise_mask"] = nm[tuple(sl)].contiguous()
                logging.info(f"[LatentCutPlus] Noise mask sliced: {tuple(out['noise_mask'].shape)}")

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
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute"
    CATEGORY = "latent/audio"
    
    def execute(self, width: int, height: int, length: int, batch_size: int):
        logging.info("=" * 80)
        logging.info("[LTXVEmptyLatentAudioDebug] Creating empty latent")
        logging.info(f"[LTXVEmptyLatentAudioDebug] width={width}, height={height}, length={length}, batch_size={batch_size}")

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


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LatentCutPlus": LatentCutPlus,
    "LTXVEmptyLatentAudioDebug": LTXVEmptyLatentAudioDebug,
    "LatentDebugInfo": LatentDebugInfo,
    "DebugAny": DebugAny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentCutPlus": "✂️ Latent Cut Plus",
    "LTXVEmptyLatentAudioDebug": "🔊 LTXV Empty Latent Audio (Debug)",
    "LatentDebugInfo": "📊 Latent Debug Info",
    "DebugAny": "🔍 Debug Any",
}
