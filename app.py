from src.ingestion.loader import Parser

import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATA_DIR = "./data/"


def main():
    # Initialize components
    parser = Parser()
    documents = parser.load_from_directory(DATA_DIR)

    print(f"Loaded {len(documents)} documents")
    for doc in documents:
        print(f"Document: {doc['metadata']['file_name']}")
        print(f"Content length: {len(doc['content'])} characters")
        print("---")
    
    # processor = TextProcessor()
    # chunker = TextChunker(chunk_size=config['chunk_size'], 
    #                       chunk_overlap=config['chunk_overlap'])
    # embedder = Embedder(model_name=config['embedding_model'])
    # vector_store = VectorStore(connection_string=config['vector_db_connection'])
    # retriever = Retriever(vector_store=vector_store, embedder=embedder)
    # prompt_builder = PromptBuilder()
    # llm = LLMInterface(model_name=config['llm_model'], 
    #                    api_key=config['api_key'])
    


    # # Example RAG pipeline
    # def process_query(query):
    #     # Retrieve relevant documents
    #     documents = retriever.retrieve(query, top_k=config['retrieval_top_k'])
        
    #     # Build prompt with context
    #     prompt = prompt_builder.build_prompt(query, documents)
        
    #     # Generate response
    #     response = llm.generate(prompt)
        
    #     return response
    
    # Start API or CLI interface
    # ...

if __name__ == "__main__":
    main() 