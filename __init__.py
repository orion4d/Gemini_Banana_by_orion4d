# Fichier : ComfyUI/custom_nodes/ComfyUI_Gemini_Nano/__init__.py
# VERSION FINALE

from .gemini_nodes import GeminiNanoStudio

NODE_CLASS_MAPPINGS = {
    "GeminiNanoStudio": GeminiNanoStudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiNanoStudio": "🍌 Gemini Nano Banana",
}

print("### Loading: 🍌 Gemini Nano Banana ###")