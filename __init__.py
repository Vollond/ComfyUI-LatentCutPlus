from .nodes_latentcutplus import LatentCutPlusExtension, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Old API export (for DebugAny)
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# New API entrypoint (for LatentCutPlus + LTXVEmptyLatentAudioDebug)
async def comfy_entrypoint():
    extension = LatentCutPlusExtension()
    print("[LatentCutPlus] Extension initialized with nodes:", [node.__name__ for node in await extension.get_node_list()])
    return extension
