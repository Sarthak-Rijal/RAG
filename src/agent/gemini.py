import os
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from PIL import Image

from src.agent.base import GenAIModel

class GeminiModel(GenAIModel):
    """Implementation of Gemini Pro model."""
    
    def __init__(self, model_name: str = "models/gemini-1.5-pro", 
                 api_key: Optional[str] = None, 
                 **kwargs):
        """Initialize Gemini model.
        
        Args:
            model_name: The Gemini model name to use
            api_key: Google API key (if not provided, will use env var GOOGLE_API_KEY)
        """
        self.model_name = model_name
        
        # Set API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("GOOGLE_API_KEY")
            
        if not self.api_key:
            raise ValueError("Google API key is required. Either pass it directly or set GOOGLE_API_KEY environment variable.")
        
        # Configure the genai library
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(model_name=self.model_name)
            self._model_info = genai.get_model(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")
    
    def generate_text(self, prompt: str) -> str:
        """Generate text based on a prompt."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    def generate_with_images(self, text_prompt: str, image_paths: List[str]) -> str:
        """Generate text based on text and image inputs."""
        try:
            # Load images
            images = [Image.open(img_path) for img_path in image_paths]
            
            # Prepare the prompt with images
            content = [text_prompt] + images
            
            # Generate response
            response = self.model.generate_content(content)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Multimodal generation failed: {str(e)}")
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "name": self.model_name,
            "provider": "Google",
            "capabilities": getattr(self._model_info, "supported_generation_methods", []),
            "description": getattr(self._model_info, "description", "Gemini multimodal model")
        } 