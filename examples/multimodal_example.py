import os
import sys
from pathlib import Path
from config.config import API_KEYS

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.genai.factory import ModelFactory

def main():
    # Get API key from environment
    api_key = API_KEYS["GOOGLE_API_KEY"]
    
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
        return
    
    # Create a Gemini model
    model = ModelFactory.create_model(
        model_type="gemini",
        model_name="models/gemini-1.5-pro",
        api_key=api_key
    )
    
    # Print model information
    print(f"Loaded model: {model.model_info['name']}")
    print(f"Provider: {model.model_info['provider']}")
    
    # Multimodal generation (if image paths provided)
    image_paths = [
        "/home/sarthak/Documents/workspace/simpleRAG/extracted_images/tableTest.png"
    ]
    
    if image_paths:
        print("\nGenerating multimodal response...\n")
        image_prompt = "Describe what you see in these images in detail."
        response = model.generate_with_images(image_prompt, image_paths)
        print(f"Prompt: {image_prompt}")
        print(f"Response: {response}")

if __name__ == "__main__":
    main() 