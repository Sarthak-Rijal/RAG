from typing import List, Union, Dict, Any
import os
from pathlib import Path
import fitz
import logging
import re

class Parser:
    """Loads documents from PDFs sources."""
    
    def load_from_directory(self, dir_path: Union[str, Path]) -> List[dict]:
        """Load all documents from a directory.
        
        Args:
            dir_path: Path to the directory containing documents
            
        Returns:
            List of document dictionaries, each containing text content and metadata
        """
        dir_path = Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory does not exist or is not a directory: {dir_path}")
        
        documents = []
        
        # Get all PDF files in the directory
        pdf_files = list(dir_path.glob("**/*.pdf"))
        
        if not pdf_files:
            logging.warning(f"No PDF files found in {dir_path}")
            return documents
        
        for pdf_path in pdf_files:
            try:
                # Load the document using PyMuPDF
                doc = self._load_pdf(pdf_path)
                documents.append(doc)
            except Exception as e:
                logging.error(f"Error loading {pdf_path}: {str(e)}")
        
        return documents
    
    def _load_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Load a PDF file and extract text and metadata."""
        # Open the PDF file
        pdf_document = fitz.open(file_path)
        
        # Extract text from all pages
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()

        
        # Extract basic metadata
        metadata = {
            "file_name": file_path.name,
            "file_type": "pdf",
            "page_count": len(pdf_document),
            "title": pdf_document.metadata.get('title', ''),
            "author": pdf_document.metadata.get('author', ''),
            "creation_date": pdf_document.metadata.get('creationDate', ''),
        }
        
        # Close the document
        pdf_document.close()
        
        print(text)
        return {
            "content": text,
            "metadata": metadata
        }