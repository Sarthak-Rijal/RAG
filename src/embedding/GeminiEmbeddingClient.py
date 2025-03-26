from typing import List
import google.generativeai as genai
from google.generativeai import types
import numpy as np
import os # Import os module
from config.config import API_KEYS
from PIL import Image  # For image loading

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.core import StorageContext
import qdrant_client
import uuid
from llama_index.core.schema import ImageDocument # Import ImageDocument


# Set up the API key from environment variable
genai.configure(api_key=API_KEYS["GOOGLE_API_KEY"])

class GeminiEmbeddingClient:
    def __init__(self, documents):
        self.client = qdrant_client.QdrantClient(path="qdrant_gemini_3")
        collection_name = "mm_collection"

        # Check if the collection already exists
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)

        if collection_exists:
            # Connect to existing collection
            self.vector_store = QdrantVectorStore(client=self.client, collection_name=collection_name)
            print(f"Using existing Qdrant collection: {collection_name}")
        else:
            # Create new collection
            self.vector_store = QdrantVectorStore(client=self.client, collection_name=collection_name, create_collection=True) # Explicitly create collection
            print(f"Created new Qdrant collection: {collection_name}")


        Settings.embed_model = GeminiEmbedding(
            model_name="models/embedding-001", api_key=API_KEYS["GOOGLE_API_KEY"]
        )

        Settings.llm = Gemini(
            model_name="models/gemini-1.5-flash-001", api_key=API_KEYS["GOOGLE_API_KEY"]
        )
        
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        text_documents = self.to_embed_text(documents)
        # image_documents = self.to_embed_images()

        ingested_documents = text_documents #+ image_documents

        self.index = VectorStoreIndex.from_documents(documents=ingested_documents, storage_context=self.storage_context)

    def to_embed_text(self, documents):
        """Prepare text for embedding."""
        vector_documents = []
        for parsed_content in documents:
            for i in range(0, len(parsed_content.nodes)):
                vector_documents.append(Document(text=parsed_content.nodes[i].text, doc_id=str(uuid.uuid4())))

        return vector_documents

    def getIndex(self):
        return self.index

    def to_embed_images(self) -> List[ImageDocument]:
        """
        Gets all images from extracted_images directory and returns a list of ImageDocument objects.
        """
        image_dir = "/home/sarthak/Documents/workspace/simpleRAG/extracted_images/"
        image_documents = []

        try:
            for filename in os.listdir(image_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(image_dir, filename)
                    try:
                        with open(image_path, "rb") as f:
                            image_data = f.read()
                            # Create ImageDocument and append to the list
                            image_document = ImageDocument(
                                image_path=image_path,
                                doc_id=str(uuid.uuid4()),
                                image=image_data,
                                content=image_data
                            )
                            image_documents.append(image_document)
                    except Exception as e:
                        print(f"Error reading image file: {filename}. Error: {e}")
        except FileNotFoundError:
            print(f"Error: Directory not found: {image_dir}")
        except Exception as e:
            print(f"Error accessing directory: {image_dir}. Error: {e}")

        print(f"Extracted {len(image_documents)} image byte arrays.")
        return image_documents