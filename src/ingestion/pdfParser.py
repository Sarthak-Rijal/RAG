from typing import List, Union, Dict, Any
from pathlib import Path
import fitz
import logging
import openparse

class Parser:
    """Loads documents from PDFs sources."""
    
    def load_from_directory(self, dir_path: Union[str, Path]) -> List[dict]:
        """Load all documents from a directory.
        
        Args:
            dir_path: Path to the directory containing documents
            
        Returns:
            parsed_dcoument with nodes that have contain parsed meta-data
        """
        dir_path = Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory does not exist or is not a directory: {dir_path}")
        
        documents = []
        
        # Get all PDF files in the directory
        pdf_files = list(dir_path.glob("**/*.pdf"))

        print(pdf_files)
        
        if not pdf_files:
            logging.warning(f"No PDF files found in {dir_path}")
            return documents
        
        for pdf_path in pdf_files:
            try:
                # Load the document using openparse
                doc = self._parse_pdf(pdf_path)
                documents.append(doc)
            except Exception as e:
                logging.error(f"Error loading {pdf_path}: {str(e)}")
        
        return documents
    
    def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Load a PDF file  extract text and metadata."""
    
        parser = openparse.DocumentParser()
        parsed_pdf = parser.parse(file_path)

        pdf = openparse.Pdf(file_path)

        pdf.export_with_bboxes(
            parsed_pdf.nodes,
            output_pdf="output.pdf"
        )

        return parsed_pdf