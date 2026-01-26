import os, sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from nodes_latentcutplus import LatentCutPlusExtension

async def comfy_entrypoint():
    return LatentCutPlusExtension()
