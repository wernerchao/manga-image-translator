import base64
import io
from typing import List
import numpy as np
import cv2
from PIL import Image
import google.generativeai as genai

from .common import OfflineOCR
from ..config import OcrConfig
from ..utils import TextBlock, Quadrilateral

class ModelGeminiOCR(OfflineOCR):
    def __init__(self, api_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = api_key
        
    async def _load(self, device: str):
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        # Initialize the model
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    async def _unload(self):
        # Clean up resources if needed
        pass

    def _prepare_image(self, image: np.ndarray) -> str:
        # Convert numpy array to base64 string
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    async def _infer(self, image: np.ndarray, textlines: List[Quadrilateral], config: OcrConfig, verbose: bool = False) -> List[TextBlock]:
        quadrilaterals = list(self._generate_text_direction(textlines))  # Returns (quad, direction) tuples
        out_regions = []

        for idx, (quad, direction) in enumerate(quadrilaterals):
            # Set direction if Quadrilateral supports it
            if hasattr(quad, 'src_is_vertical'):
                quad.src_is_vertical = (direction == 'v')

            # Extract the text region with direction parameter
            region_img = quad.get_transformed_region(image, direction=direction, textheight=48)

            # Convert image for Gemini API
            encoded_image = self._prepare_image(region_img)

            try:
                # Call Gemini API for OCR with more specific prompt
                response =  self.model.generate_content({
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": encoded_image
                            }
                        },
                        {
                            "text": "Extract and return only the text shown in this manga/comic text region. Return only the raw text without any explanation or context."
                        }
                    ]
                })

                print(f"Response: {response.text}")

                # Extract text from response
                text = response.text.strip()

                print(f"Text: {text}")

                # Update the quadrilateral with the detected text
                if isinstance(quad, Quadrilateral):
                    quad.text = text
                    quad.prob = 1.0  # Gemini doesn't provide confidence scores
                    quad.fg_r = quad.fg_g = quad.fg_b = 0  # black text
                    quad.bg_r = quad.bg_g = quad.bg_b = 255  # white background
                else:
                    quad.text.append(text)
                    quad.update_font_colors(
                        np.array([0, 0, 0]),  # black text
                        np.array([255, 255, 255])  # white background
                    )

                out_regions.append(quad)

            except Exception as e:
                self.logger.error(f"Error processing region with Gemini: {str(e)}")
                continue

        return out_regions if isinstance(textlines[0], Quadrilateral) else textlines