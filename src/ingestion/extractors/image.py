import os
import re
import fitz
import logging
import gc
from typing import Union, List
from pathlib import Path

class ImageExtractor:
    """Extract images from PDF documents."""
    
    def extract_images(self, file_path: Union[str, Path], output_dir="./extracted_images/"):
        """
        Extract only properly captioned images (Figures and Tables).
        Skip images without clear captions to reduce clutter.
        """
        os.makedirs(output_dir, exist_ok=True)
        doc = None
        
        try:
            doc = fitz.Document(file_path)
            img_count = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_width = page.rect.width
                page_height = page.rect.height
                
                # Get all text blocks to search for captions
                page_dict = page.get_text("dict")
                text_blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]
                
                # Get all caption blocks, categorizing as Figure or Table
                figure_captions = []
                table_captions = []
                
                for block in text_blocks:
                    text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text += span.get("text", "")
                
                    # More specific caption matching to avoid false positives
                    if re.search(r'^(Figure|Fig\.)\s+\d+', text, re.IGNORECASE):
                        match = re.search(r'(Figure|Fig\.?)\s*(\d+[A-Za-z]?)', text, re.IGNORECASE)
                        if match:
                            figure_num = match.group(2)
                            figure_captions.append((block["bbox"], text, figure_num))
                    elif re.search(r'^Table\s+\d+', text, re.IGNORECASE):
                        match = re.search(r'Table\s*(\d+[A-Za-z]?)', text, re.IGNORECASE)
                        if match:
                            table_num = match.group(1)
                            table_captions.append((block["bbox"], text, table_num))
                
                # Extract images on the page only if they have proper captions
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        
                        # Find image location
                        img_bbox = None
                        for item in page.get_image_info():
                            if "xref" in item and item["xref"] == xref:
                                img_bbox = item["bbox"]
                                break
                        
                        if not img_bbox:
                            for block in page_dict.get("blocks", []):
                                if block.get("type") == 1:  # Image block
                                    img_bbox = block["bbox"]
                                    break
                        
                        if not img_bbox:
                            continue  # Skip if no bbox found
                        
                        # Set initial bbox
                        x0, y0, x1, y1 = img_bbox
                        
                        # Find nearby figure captions first
                        nearby_figure_captions = []
                        for (c_bbox, c_text, figure_num) in figure_captions:
                            cx0, cy0, cx1, cy1 = c_bbox
                            
                            # Check if caption is within reasonable distance
                            distance_threshold = page_height * 0.25
                            
                            # Above or below image
                            if (cy1 <= y0 and (y0 - cy1) < distance_threshold) or \
                               (cy0 >= y1 and (cy0 - y1) < distance_threshold):
                                nearby_figure_captions.append((c_bbox, c_text, figure_num))
                        
                        # If no figure captions found, only then check for table captions
                        if nearby_figure_captions:
                            # Process figure
                            caption_type = "Figure"
                            figure_num = nearby_figure_captions[0][2]  # Get the figure number
                            
                            # Expand bbox to include captions
                            for (c_bbox, _, _) in nearby_figure_captions:
                                cx0, cy0, cx1, cy1 = c_bbox
                                y0 = min(y0, cy0)
                                y1 = max(y1, cy1)
                            
                            # Add padding
                            y0 = max(0, y0 - 10)
                            y1 = min(page_height, y1 + 10)
                            
                            # Use full page width
                            final_bbox = (0, y0, page_width, y1)
                            
                            # Extract the image at lower resolution
                            pix = page.get_pixmap(clip=final_bbox, matrix=fitz.Matrix(2.5, 2.5))
                            
                            # Save with figure number
                            output_filename = f"{output_dir}/figure_{figure_num}.png"
                            pix.save(output_filename)
                            pix = None  # Free memory
                            img_count += 1
                            print(f"Extracted Figure {figure_num} from page {page_num+1}")
                            
                        else:
                            # Check for table captions
                            nearby_table_captions = []
                            for (c_bbox, c_text, table_num) in table_captions:
                                cx0, cy0, cx1, cy1 = c_bbox
                                
                                distance_threshold = page_height * 0.25
                                
                                if (cy1 <= y0 and (y0 - cy1) < distance_threshold) or \
                                   (cy0 >= y1 and (cy0 - y1) < distance_threshold):
                                    nearby_table_captions.append((c_bbox, c_text, table_num))
                            
                            if nearby_table_captions:
                                # Process table
                                caption_type = "Table"
                                table_num = nearby_table_captions[0][2]  # Get the table number
                                
                                # Expand bbox to include captions
                                for (c_bbox, _, _) in nearby_table_captions:
                                    cx0, cy0, cx1, cy1 = c_bbox
                                    y0 = min(y0, cy0)
                                    y1 = max(y1, cy1)
                                
                                # Add padding
                                y0 = max(0, y0 - 10)
                                y1 = min(page_height, y1 + 10)
                                
                                # Use full page width
                                final_bbox = (0, y0, page_width, y1)
                                
                                # Extract the image
                                pix = page.get_pixmap(clip=final_bbox, matrix=fitz.Matrix(2.5, 2.5))
                                
                                # Save with table number
                                output_filename = f"{output_dir}/table_{table_num}.png"
                                pix.save(output_filename)
                                pix = None  # Free memory
                                img_count += 1
                                print(f"Extracted Table {table_num} from page {page_num+1}")
                        
                    except Exception as e:
                        logging.warning(f"Error extracting image {img_index} on page {page_num+1}: {str(e)}")
                
                # Free memory after processing each page
                page = None
                gc.collect()
            
            print(f"Extracted {img_count} captioned images from {file_path}")
        
        except Exception as e:
            logging.error(f"Failed to extract images from {file_path}: {str(e)}")
        
        finally:
            # Always close the document
            if doc:
                doc.close()
            gc.collect() 