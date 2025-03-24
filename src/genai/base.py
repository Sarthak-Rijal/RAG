from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any

class GenAIModel(ABC):
    """Base class for all GenAI models."""
    
    @abstractmethod
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """Initialize the model with credentials and configuration."""
        pass
    
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generate text based on a text prompt."""
        pass
    
    @abstractmethod
    def generate_with_images(self, text_prompt: str, image_paths: List[str]) -> str:
        """Generate text based on text and image inputs."""
        pass
    
    @property
    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Return model information."""
        pass 