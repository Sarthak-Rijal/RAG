import os
import re
import fitz
import logging
import gc
from typing import Union, List
from pathlib import Path

class TableExtractor:
    """Extract tables from PDF documents."""
    
    def extract_tables(self, file_path: Union[str, Path], output_dir="./extracted_images/"):
        """Extract tables with captions."""
        os.makedirs(output_dir, exist_ok=True)
        doc = None
        
        try:
            doc = fitz.Document(file_path)
            table_count = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_width = page.rect.width
                page_height = page.rect.height
                
                # Get text blocks
                page_dict = page.get_text("dict")
                blocks = page_dict["blocks"]
                
                # Find table captions
                table_captions = []
                
                for block_idx, block in enumerate(blocks):
                    if block.get("type") == 0:  # Text block
                        text = ""
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text += span.get("text", "")
                        
                        text = text.strip()
                        
                        # EXPANDED patterns for tables
                        table_patterns = [
                            r'^(Table)\s+\d+\s*[:.]',        # Table 1: or Table 1.
                            r'^(Table)\s+\d+\s*[-–]',        # Table 1 - or Table 1–
                            r'^(Table)\s+\d+$',              # Just "Table 1" at end of line
                            r'^(Table)\s+\d+[A-Za-z]$',      # Table 1A
                            r'^(Table)\s+\d+[A-Za-z]\s*[:.:]' # Table 1A: or Table 1B.
                        ]
                        
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
            
            print(f"Extracted {table_count} tables from {file_path}")
        
        except Exception as e:
            logging.error(f"Failed to extract tables from {file_path}: {str(e)}")
        
        finally:
            # Always close the document
            if doc:
                doc.close()
            gc.collect() 