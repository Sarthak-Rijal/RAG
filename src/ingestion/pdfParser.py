from typing import List, Union, Dict, Any
from pathlib import Path
import fitz
import logging
from tqdm import tqdm
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
                parsed_document, pdf = self._parse_pdf(pdf_path)
                documents.append(parsed_document)

                pdf.export_with_bboxes(
                    parsed_document.nodes,
                    output_pdf=f"{pdf_path}_output.pdf"
                )

                print(parsed_document.nodes)
                
            except Exception as e:
                logging.error(f"Error loading {pdf_path}: {str(e)}")
        
        return documents
    
    def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Load a PDF file  extract text and metadata."""
    
        parser = openparse.DocumentParser()
        parsed_pdf = parser.parse(file_path)
        self.__extract_images(file_path)

        pdf = openparse.Pdf(file_path)

        return parsed_pdf, pdf

    def __extract_images(self, file_path):
        """Extract images from a PDF file."""
        doc = fitz.Document(file_path)
        img_count = 0
        
        # Iterate through each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Get image list for this page
            image_list = page.get_images(full=True)
            
            # Process each image on the page
            for img_index, img in enumerate(tqdm(image_list, desc=f"Page {page_num+1} images")):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                # Use unique naming to avoid overwrites (page_image#.png)
                output_filename = f"page{page_num+1}_img{img_index+1}.png"
                
                # Save images that can be directly saved
                if pix.n < 5:  # RGB or grayscale
                    pix.save(output_filename)
                else:  # CMYK: convert to RGB first
                    pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                    pix_rgb.save(output_filename)
                    pix_rgb = None  # free memory
                
                pix = None  # free memory
                img_count += 1

        print(f"Extracted {img_count} images from {file_path}")
        doc.close()
        
        