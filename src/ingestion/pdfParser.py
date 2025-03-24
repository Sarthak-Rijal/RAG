from typing import List, Union, Dict, Any
from pathlib import Path
import fitz
import logging
from tqdm import tqdm
import openparse
import os
import re
import traceback

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

        
        if not pdf_files:
            logging.warning(f"No PDF files found in {dir_path}")
            return documents
        
        for pdf_path in pdf_files:
            try:
                # Skip files that are output PDFs from previous runs
                if "_output.pdf" in str(pdf_path):
                    continue
                
                # Load the document using openparse
                parsed_document, pdf = self._parse_pdf(pdf_path)
                documents.append(parsed_document)

                # # Handle the export with a try-except block
                # try:
                #     pdf.export_with_bboxes(
                #         parsed_document.nodes,
                #         output_pdf=f"{pdf_path}_output.pdf"
                #     )
                # except Exception as e:
                #     logging.warning(f"Could not export PDF with bboxes: {str(e)}")
                
            except Exception as e:
                logging.error(f"Error loading {pdf_path}: {str(e)}")
        
        return documents
    
    def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Load a PDF file and extract text and metadata."""
        parser = openparse.DocumentParser()
        parsed_pdf = parser.parse(file_path)
        
        # Extract images using both methods
        self.extract_images(file_path)  # Original method
        self.extract_figures_by_caption(file_path)  # New caption-first method
        
        pdf = openparse.Pdf(file_path)
        return parsed_pdf, pdf

    def extract_images(self, file_path, output_dir="./extracted_images/"):
        """Extract full images with their captions, prioritizing Figures over Tables."""
        os.makedirs(output_dir, exist_ok=True)
        
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
                    
                    if re.search(r'(Figure|Fig\.)[:\s]', text, re.IGNORECASE):
                        figure_captions.append((block["bbox"], text))
                    elif re.search(r'Table[:\s]', text, re.IGNORECASE):
                        table_captions.append((block["bbox"], text))
                
                # Extract images on the page
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
                            logging.warning(f"Could not find bounding box for image on page {page_num+1}")
                            continue
                        
                        # Set initial bbox
                        x0, y0, x1, y1 = img_bbox
                        
                        # Find nearby figure captions first
                        nearby_figure_captions = []
                        for (c_bbox, c_text) in figure_captions:
                            cx0, cy0, cx1, cy1 = c_bbox
                            
                            # Check if caption is within reasonable distance
                            distance_threshold = page_height * 0.25
                            
                            # Above or below image
                            if (cy1 <= y0 and (y0 - cy1) < distance_threshold) or \
                               (cy0 >= y1 and (cy0 - y1) < distance_threshold):
                                nearby_figure_captions.append((c_bbox, c_text))
                        
                        # If no figure captions found, only then check for table captions
                        nearby_captions = nearby_figure_captions
                        caption_type = "Figure"
                        
                        if not nearby_figure_captions:
                            # Use table captions only if no figure captions are found
                            nearby_table_captions = []
                            for (c_bbox, c_text) in table_captions:
                                cx0, cy0, cx1, cy1 = c_bbox
                                
                                distance_threshold = page_height * 0.25
                                
                                if (cy1 <= y0 and (y0 - cy1) < distance_threshold) or \
                                   (cy0 >= y1 and (cy0 - y1) < distance_threshold):
                                    nearby_table_captions.append((c_bbox, c_text))
                            
                            nearby_captions = nearby_table_captions
                            caption_type = "Table" if nearby_table_captions else "Unknown"
                        
                        # Expand bbox to include captions
                        for (c_bbox, _) in nearby_captions:
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
                        
                        # Save the image with type in filename
                        output_filename = f"{output_dir}/page{page_num+1}_{caption_type}{img_index+1}.png"
                        pix.save(output_filename)
                        img_count += 1
                        
                        # If no captions were found, add fallback expansion
                        if not nearby_captions:
                            height = y1 - y0
                            expand_top = height * 0.1
                            expand_bottom = height * 0.5
                            
                            fallback_bbox = (0, max(0, y0 - expand_top), 
                                             page_width, min(page_height, y1 + expand_bottom))
                            
                            pix = page.get_pixmap(clip=fallback_bbox, matrix=fitz.Matrix(2.5, 2.5))
                            
                            output_filename = f"{output_dir}/page{page_num+1}_Unknown{img_index+1}_expanded.png"
                            pix.save(output_filename)
                        
                    except Exception as e:
                        logging.warning(f"Error extracting image {img_index} on page {page_num+1}: {str(e)}")
            
            print(f"Extracted {img_count} images from {file_path}")
            doc.close()
            
        except Exception as e:
            logging.error(f"Failed to extract images from {file_path}: {str(e)}")
        
    def extract_figures_by_caption(self, file_path, output_dir="./extracted_images/"):
        """Extract figures that are strictly labeled with 'Figure X:' captions."""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            doc = fitz.Document(file_path)
            figure_count = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_width = page.rect.width
                page_height = page.rect.height
                
                # Get text blocks
                page_dict = page.get_text("dict")
                blocks = page_dict["blocks"]
                
                # Find strictly formatted figure captions
                figure_captions = []
                
                for block_idx, block in enumerate(blocks):
                    if block.get("type") == 0:  # Text block
                        text = ""
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text += span.get("text", "")
                        
                        text = text.strip()
                        
                        # STRICT pattern matching - must start with Figure/Fig followed by number and colon
                        if re.match(r'^(Figure|Fig\.)\s+\d+\s*:', text, re.IGNORECASE):
                            match = re.search(r'(Figure|Fig\.)\s+(\d+)', text, re.IGNORECASE)
                            figure_num = match.group(2) if match else "unknown"
                            
                            figure_captions.append({
                                "bbox": block["bbox"],
                                "text": text,
                                "figure_num": figure_num,
                                "block_idx": block_idx
                            })
                
                # Process each strictly-matched caption
                for caption in figure_captions:
                    caption_bbox = caption["bbox"]
                    cx0, cy0, cx1, cy1 = caption_bbox
                    figure_num = caption["figure_num"]
                    
                    # Content is usually above the caption - look 40% of page up
                    content_top = max(0, cy0 - page_height * 0.4)
                    
                    # Create a capture region from content_top to just below caption
                    capture_bbox = (0, content_top, page_width, cy1 + 10)
                    
                    # Capture at high resolution
                    pix = page.get_pixmap(clip=capture_bbox, matrix=fitz.Matrix(3, 3))
                    output_filename = f"{output_dir}/figure_{figure_num}_page{page_num+1}.png"
                    pix.save(output_filename)
                    figure_count += 1
                    
                    print(f"Extracted Figure {figure_num} from page {page_num+1}")
            
            print(f"Extracted {figure_count} figures with strict caption detection from {file_path}")
            doc.close()
        
        except Exception as e:
            logging.error(f"Failed to extract figures from {file_path}: {str(e)}")
            traceback.print_exc()
        
        