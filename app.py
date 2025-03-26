from src.ingestion.parser import Parser
from src.embedding.GeminiEmbeddingClient import GeminiEmbeddingClient
from llama_index.core import Document

import json
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

DATA_DIR = "./data/"


def main():

    parser = Parser()
    documents = parser.load_from_directory(DATA_DIR)

    # components
    embeddingClient = GeminiEmbeddingClient(documents)

    index = embeddingClient.getIndex()
    query_engine = index.as_query_engine()

    print("--------------------------------\n")
    print("Setup complete\n")
    print("--------------------------------\n\n")

    while True:
        prompt = input("Enter a query: ")
        response = query_engine.query(prompt)
        print(response)
        print("\n\n")

if __name__ == "__main__":
    main() 