from typing import Dict, Optional, Any

from src.genai.base import GenAIModel
from src.genai.gemini import GeminiModel

class ModelFactory:
    """Factory for creating GenAI model instances."""

    @staticmethod
    def create_model(model_type: str, **kwargs) -> GenAIModel:
        """Create a model instance based on type.
        
        Args:
            model_type: Type of model to create ("gemini", "claude", etc.)
            **kwargs: Arguments to pass to the model constructor
            
        Returns:
            An instance of the requested model
        """
        model_type = model_type.lower()
        
        if model_type == "gemini":
            return GeminiModel(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}") 