# Fichier : ComfyUI/custom_nodes/ComfyUI_Gemini_Nano/gemini_nodes.py

import os
import io
import torch
import numpy as np
from PIL import Image
import google.generativeai as genai
import cv2

try:
    from folder_paths import get_filename_list
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader
    upscalers_available = True
except ImportError:
    print("Avertissement [Gemini_Nano]: Nodes d'upscale non trouvés. L'upscaling par modèle sera désactivé.", flush=True)
    upscalers_available = False

NODE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
API_KEY_FILE = os.path.join(NODE_DIRECTORY, "apikey.txt")

def get_api_key():
    if not os.path.exists(API_KEY_FILE): raise Exception(f"Fichier apikey.txt non trouvé. Créez-le dans {NODE_DIRECTORY}.")
    with open(API_KEY_FILE, 'r') as f: api_key = f.read().strip()
    if not api_key: raise Exception("Le fichier apikey.txt est vide.")
    return api_key

def tensor_to_pil(tensor):
    if tensor is None: return None
    return Image.fromarray((tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8), 'RGB')

def pil_to_tensor(pil_image):
    if pil_image is None: return None
    return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0)[None,]

class GeminiNanoStudio:
    @classmethod
    def INPUT_TYPES(s):
        models = ["None"]
        if upscalers_available: models.extend(get_filename_list("upscale_models"))
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful cinematic photo."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image1": ("IMAGE",), "image2": ("IMAGE",), "image3": ("IMAGE",), "image4": ("IMAGE",),
                "enable_upscale": ("BOOLEAN", {"default": False}), "upscale_model": (models,),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "upscale_method": (["lanczos", "nearest-exact", "bilinear", "area"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text_output")
    FUNCTION = "run"
    CATEGORY = "Gemini Banana" # On garde ta catégorie personnalisée

    def _do_upscale(self, image_pil, factor, model_name, method):
        print(f"[Gemini_Nano] Upscaling x{factor}...", flush=True)
        if upscalers_available and model_name != "None":
            try:
                loader, upscaler = UpscaleModelLoader(), ImageUpscaleWithModel()
                upscale_model = loader.load_model(model_name)[0]
                upscaled_tensor = upscaler.upscale(upscale_model, pil_to_tensor(image_pil))[0]
                print(f"[Gemini_Nano] Upscale avec modèle '{model_name}' réussi.", flush=True)
                return tensor_to_pil(upscaled_tensor)
            except Exception as e: 
                print(f"[Gemini_Nano] Upscale modèle a échoué : {e}. Fallback OpenCV.", flush=True)
        
        h, w = image_pil.height, image_pil.width
        nh, nw = int(round(h * factor)), int(round(w * factor))
        interp_methods = {"nearest-exact": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA, "lanczos": cv2.INTER_LANCZOS4}
        upscaled_np = cv2.resize(np.array(image_pil), (nw, nh), interpolation=interp_methods.get(method, cv2.INTER_LANCZOS4))
        print(f"[Gemini_Nano] Upscale avec OpenCV ({method}) réussi.", flush=True)
        return Image.fromarray(upscaled_np)

    def run(self, prompt, seed, image1=None, image2=None, image3=None, image4=None, enable_upscale=False, upscale_model="None", upscale_factor=2.0, upscale_method="lanczos"):
        api_key = get_api_key()
        genai.configure(api_key=api_key)
        
        contents = [prompt]
        if image1 is not None: contents.append(tensor_to_pil(image1))
        if image2 is not None: contents.append(tensor_to_pil(image2))
        if image3 is not None: contents.append(tensor_to_pil(image3))
        if image4 is not None: contents.append(tensor_to_pil(image4))

        mode = "Text-to-Image" if image1 is None else "Image-to-Image"
        print(f"Lancement de Gemini Nano en mode {mode}...", flush=True)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")
        try: 
            response = model.generate_content(contents)
        except Exception as e: 
            raise Exception(f"Erreur API Gemini : {e}")

        output_pil_image, output_text = None, ""
        for part in response.candidates[0].content.parts:
            if part.text: 
                output_text += part.text + "\n"
            if hasattr(part, 'inline_data'):
                try: 
                    output_pil_image = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                except Image.UnidentifiedImageError: 
                    print("Avertissement : Données non-image ignorées.", flush=True)
        
        if output_pil_image is None:
            print("Avertissement : Pas d'image retournée. Création d'une image noire.", flush=True)
            w, h = (image1.shape[2], image1.shape[1]) if image1 is not None else (1024, 1024)
            output_pil_image = Image.new('RGB', (w, h))
            if not output_text: 
                output_text = "Le modèle n'a retourné ni image, ni texte."
        elif enable_upscale and upscale_factor > 1.0:
            output_pil_image = self._do_upscale(output_pil_image, upscale_factor, upscale_model, upscale_method)

        return (pil_to_tensor(output_pil_image), output_text.strip())