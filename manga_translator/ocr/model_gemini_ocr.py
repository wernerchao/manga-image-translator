import base64
import io
from typing import List, Tuple, Optional, Any
import numpy as np
import cv2
from PIL import Image
import google.generativeai as genai

from .common import OfflineOCR
from ..config import OcrConfig
from ..utils import TextBlock, Quadrilateral


class ModelGeminiOCR(OfflineOCR):
    """
    OCR implementation using Google's Gemini vision model to extract text from images.
    
    This class handles loading the Gemini model, preparing images, and inferring text
    from image regions defined by quadrilaterals.
    """
    
    def __init__(self, api_key: str, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the Gemini OCR model.
        
        Args:
            api_key: The Google API key for accessing Gemini models
            *args: Additional positional arguments to pass to the parent class
            **kwargs: Additional keyword arguments to pass to the parent class
        """
        super().__init__(*args, **kwargs)
        self.api_key: str = api_key
        self.model: Optional[Any] = None
        
    async def _load(self, device: str) -> None:
        """
        Load and initialize the Gemini model.
        
        Args:
            device: The device to use for inference (not used for Gemini API)
        """
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        # Initialize the model
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    async def _unload(self) -> None:
        """
        Clean up resources when the model is no longer needed.
        """
        # Clean up resources if needed
        self.model = None

    def _prepare_image(self, image: np.ndarray) -> str:
        """
        Convert a numpy image array to a base64-encoded string for API requests.
        
        Args:
            image: The input image as a numpy array in BGR format
            
        Returns:
            Base64-encoded string representation of the image
        """
        # Convert numpy array to base64 string
        img: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img: Image.Image = Image.fromarray(img)
        buffer: io.BytesIO = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    async def _infer(
        self, 
        image: np.ndarray, 
        textlines: List[Quadrilateral], 
        config: OcrConfig, 
        verbose: bool = False
    ) -> List[TextBlock]:
        """
        Perform OCR on text regions in the image.
        
        Args:
            image: The input image as a numpy array
            textlines: List of quadrilaterals representing text regions
            config: OCR configuration settings
            verbose: Whether to output detailed logs
            
        Returns:
            List of TextBlock objects containing the extracted text and metadata
        """
        quadrilaterals: List[Tuple[Quadrilateral, str]] = list(self._generate_text_direction(textlines))  # Returns (quad, direction) tuples
        out_regions: List[TextBlock] = []

        for idx, (quad, direction) in enumerate(quadrilaterals):
            # Set direction if Quadrilateral supports it
            if hasattr(quad, 'src_is_vertical'):
                quad.src_is_vertical = (direction == 'v')

            # Extract the text region with direction parameter
            region_img: np.ndarray = quad.get_transformed_region(image, direction=direction, textheight=48)

            # Convert image for Gemini API
            encoded_image: str = self._prepare_image(region_img)

            try:
                # Call Gemini API for OCR with more specific prompt
                response = self.model.generate_content({
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

                if verbose:
                    self.logger.debug(f"Response: {response.text}")

                # Extract text from response
                text: str = response.text.strip()

                if verbose:
                    self.logger.debug(f"Text: {text}")

                # Update the quadrilateral with the detected text
                if isinstance(quad, Quadrilateral):
                    quad.text = text
                    quad.prob = 1.0  # Gemini doesn't provide confidence scores
                    quad.fg_r = quad.fg_g = quad.fg_b = 0  # black text
                    quad.bg_r = quad.bg_g = quad.bg_b = 255  # white background
                else:
                    quad.text.append(text)
                    quad.update_font_colors(
                        np.array([0, 0, 0], dtype=np.uint8),  # black text
                        np.array([255, 255, 255], dtype=np.uint8)  # white background
                    )

                out_regions.append(quad)

            except Exception as e:
                self.logger.error(f"Error processing region with Gemini: {str(e)}")
                continue

        return out_regions if isinstance(textlines[0], Quadrilateral) else textlines