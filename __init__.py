
from .nodes_latentcutplus import LatentCutPlusExtension

async def comfy_entrypoint():
    return LatentCutPlusExtension()
