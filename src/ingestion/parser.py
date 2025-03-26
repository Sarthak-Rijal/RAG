from typing import List, Union, Dict, Any
from pathlib import Path
import fitz
import logging
import os
import re
import traceback
import gc
import openparse
from openparse import processing, Pdf
from config.config import API_KEYS

from src.ingestion.extractors.image import ImageExtractor
from src.ingestion.extractors.table import TableExtractor

class Parser:
    """Loads and parses documents from various sources."""
    
    def __init__(self):
        """Initialize the document parser."""
        self.image_extractor = ImageExtractor()
        self.table_extractor = TableExtractor()
    
    def load_from_directory(self, dir_path: Union[str, Path]) -> List[dict]:
        """Load all documents from a directory with better memory management."""
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
                # Skip files that are output PDFs from previous runs
                if "_output.pdf" in str(pdf_path):
                    continue
                
                # First extract images and figures to avoid memory issues
                self.extract_figures_and_tables(pdf_path)
                self.image_extractor.extract_images(file_path=pdf_path)
                
                # Clean up before parsing with OpenParse
                gc.collect()
                
                # Then parse with OpenParse
                try:
                    # Load the document using openparse
                    parsed_content = self._parse_pdf(pdf_path)                    
                    documents.append(parsed_content)
                    
                except Exception as e:
                    logging.error(f"Error in OpenParse processing for {pdf_path}: {str(e)}")
                
            except Exception as e:
                logging.error(f"Error loading {pdf_path}: {str(e)}")
                traceback.print_exc()
        
        return documents
    
    def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Load a PDF file and extract text and metadata."""
        # Force garbage collection
        gc.collect()

        # initialize the semantic pipeline
        # semantic_pipeline = processing.SemanticIngestionPipeline(
        #     openai_api_key=API_KEYS["OPENAI_API_KEY"],
        #     model="text-embedding-3-large",
        #     min_tokens=64,
        #     max_tokens=1024,
        # )
        # parser = openparse.DocumentParser(
        #     processing_pipeline=semantic_pipeline,
        # )
        parser = openparse.DocumentParser()
        parsed_content = parser.parse(file_path)
        return parsed_content

    def extract_figures_and_tables(self, file_path, output_dir="./extracted_images/"):
        """Extract figures and tables with improved caption pattern matching."""
        os.makedirs(output_dir, exist_ok=True)
        doc = None
        
        try:
            doc = fitz.Document(file_path)
            figure_count = 0
            table_count = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_width = page.rect.width
                page_height = page.rect.height
                
                # Get text blocks
                page_dict = page.get_text("dict")
                blocks = page_dict["blocks"]
                
                # Find figure captions with MORE FLEXIBLE patterns
                figure_captions = []
                table_captions = []
                
                for block_idx, block in enumerate(blocks):
                    if block.get("type") == 0:  # Text block
                        text = ""
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text += span.get("text", "")
                        
                        text = text.strip()
                        
                        # EXPANDED patterns for figures
                        figure_patterns = [
                            r'^(Figure|Fig\.?)\s+\d+\s*[:.]',  # Figure 1: or Fig. 1.
                            r'^(Figure|Fig\.?)\s+\d+\s*[-–]',  # Figure 1 - or Fig 1–
                            r'^(Figure|Fig\.?)\s+\d+$',        # Just "Figure 1" at end of line
                            r'^(Figure|Fig\.?)\s+\d+[A-Za-z]$', # Figure 1A
                            r'^(Figure|Fig\.?)\s+\d+[A-Za-z]\s*[:.]'  # Figure 1A: or Figure 1B.
                        ]
                        
                        # EXPANDED patterns for tables
                        table_patterns = [
                            r'^(Table)\s+\d+\s*[:.]',        # Table 1: or Table 1.
                            r'^(Table)\s+\d+\s*[-–]',        # Table 1 - or Table 1–
                            r'^(Table)\s+\d+$',              # Just "Table 1" at end of line
                            r'^(Table)\s+\d+[A-Za-z]$',      # Table 1A
                            r'^(Table)\s+\d+[A-Za-z]\s*[:.:]' # Table 1A: or Table 1B.
                        ]
                        
                        # Check if text matches any figure pattern
                        is_figure_caption = any(re.match(pattern, text, re.IGNORECASE) for pattern in figure_patterns)
                        
                        if is_figure_caption:
                            # Extract figure number using a more flexible pattern
                            match = re.search(r'(Figure|Fig\.?)\s*(\d+[A-Za-z]?)', text, re.IGNORECASE)
                            figure_num = match.group(2) if match else "unknown"
                            
                            figure_captions.append({
                                "bbox": block["bbox"],
                                "text": text,
                                "figure_num": figure_num,
                                "block_idx": block_idx
                            })
                            continue  # Skip to next block
                        
                        # Check if text matches any table pattern
                        is_table_caption = any(re.match(pattern, text, re.IGNORECASE) for pattern in table_patterns)
                        
                        if is_table_caption:
                            # Extract table number
                            match = re.search(r'(Table)\s*(\d+[A-Za-z]?)', text, re.IGNORECASE)
                            table_num = match.group(2) if match else "unknown"
                            
                            table_captions.append({
                                "bbox": block["bbox"],
                                "text": text,
                                "table_num": table_num,
                                "block_idx": block_idx
                            })
                
                # Process each figure caption
                for caption in figure_captions:
                    try:
                        caption_bbox = caption["bbox"]
                        cx0, cy0, cx1, cy1 = caption_bbox
                        figure_num = caption["figure_num"]
                        
                        # Content is usually above the caption
                        content_top = max(0, cy0 - page_height * 0.4)
                        
                        # Create a capture region from content_top to just below caption
                        capture_bbox = (0, content_top, page_width, cy1 + 10)
                        
                        # Capture at slightly lower resolution to save memory
                        pix = page.get_pixmap(clip=capture_bbox, matrix=fitz.Matrix(2.5, 2.5))
                        output_filename = f"{output_dir}/figure_{figure_num}.png"
                        pix.save(output_filename)
                        pix = None  # Free memory
                        
                        figure_count += 1
                        print(f"Extracted Figure {figure_num} from page {page_num+1}")
                    except Exception as e:
                        logging.error(f"Error processing figure caption on page {page_num+1}: {str(e)}")
                
                # Process each table caption
                for caption in table_captions:
                    try:
                        caption_bbox = caption["bbox"]
                        cx0, cy0, cx1, cy1 = caption_bbox
                        table_num = caption["table_num"]
                        
                        # Tables often have caption above, so search both directions
                        # Look slightly above and more below the caption
                        content_top = max(0, cy0 - page_height * 0.1)
                        content_bottom = min(page_height, cy1 + page_height * 0.3)
                        
                        # Create a capture region
                        capture_bbox = (0, content_top, page_width, content_bottom)
                        
                        # Capture at slightly lower resolution to save memory
                        pix = page.get_pixmap(clip=capture_bbox, matrix=fitz.Matrix(2.5, 2.5))
                        output_filename = f"{output_dir}/table_{table_num}.png"
                        pix.save(output_filename)
                        pix = None  # Free memory
                        
                        table_count += 1
                        print(f"Extracted Table {table_num} from page {page_num+1}")
                    except Exception as e:
                        logging.error(f"Error processing table caption on page {page_num+1}: {str(e)}")
                
                # Free memory after processing each page
                page = None
                gc.collect()
            
            print(f"Extracted {figure_count} figures and {table_count} tables from {file_path}")
        
        except Exception as e:
            logging.error(f"Failed to extract figures/tables from {file_path}: {str(e)}")
            traceback.print_exc()
        
        finally:
            # Always close the document
            if doc:
                doc.close()
            gc.collect() 