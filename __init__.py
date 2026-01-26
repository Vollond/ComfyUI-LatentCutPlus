from .nodes_latentcutplus import LatentCutPlusExtension, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

async def comfy_entrypoint():
    return LatentCutPlusExtension()

# Export old API nodes
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
