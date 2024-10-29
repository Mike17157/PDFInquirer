import layoutparser as lp
from pdf2image import convert_from_path
from pathlib import Path
import json
from PIL import Image
from typing import Dict, List, Optional
import logging
import os

class DocumentAnalyzer:
    def __init__(self, output_dir: str = "processed_documents"):
        self.model = lp.Detectron2LayoutModel(
            config_path = 'lp://HJDataset/faster_rcnn_R_50_FPN_3x/config',
            label_map = {
                1: "Title",
                2: "Text",
                3: "List",
                4: "Table",
                5: "Figure",
                6: "Caption"
            }
        )
        
        self.ocr_agent = lp.TesseractAgent()
        self.output_dir = Path(output_dir)
        self.setup_directories()

    def setup_directories(self):
        for subdir in ['figures', 'tables', 'sections', 'metadata']:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    def find_element_caption_pairs(self, layout, image_height) -> List[Dict]:
        """
        Group figures/tables with their captions using adaptive distance thresholds
        """
        elements = [block for block in layout if block.type in ["Figure", "Table"]]
        captions = [block for block in layout if block.type == "Caption"]
        
        # Calculate adaptive threshold based on image height
        base_threshold = image_height * 0.05  # 5% of image height
        
        paired_elements = []
        used_captions = set()
        
        for element in elements:
            element_bottom = element.coordinates[3]
            element_center = (element.coordinates[0] + element.coordinates[2]) / 2
            element_width = element.coordinates[2] - element.coordinates[0]
            
            # Find closest caption
            closest_caption = None
            min_distance = float('inf')
            
            for caption in captions:
                if caption in used_captions:
                    continue
                    
                caption_top = caption.coordinates[1]
                caption_center = (caption.coordinates[0] + caption.coordinates[2]) / 2
                
                # Vertical and horizontal distances
                vertical_dist = caption_top - element_bottom
                horizontal_dist = abs(caption_center - element_center)
                
                # Adaptive thresholds
                vertical_threshold = base_threshold
                horizontal_threshold = element_width * 0.5  # 50% of element width
                
                # Check if caption is below and centered
                if (0 <= vertical_dist <= vertical_threshold and 
                    horizontal_dist <= horizontal_threshold):
                    # Combined distance score (weighted more towards vertical distance)
                    distance = vertical_dist * 2 + horizontal_dist
                    if distance < min_distance:
                        min_distance = distance
                        closest_caption = caption
            
            if closest_caption:
                used_captions.add(closest_caption)
            
            paired_elements.append({
                'element': element,
                'caption': closest_caption,
                'distance_score': min_distance if closest_caption else None
            })
        
        return paired_elements

    def process_element_pair(self, 
                           pair: Dict, 
                           image: Image, 
                           document_id: str, 
                           page_num: int, 
                           counter: int) -> Dict:
        """Process a figure/table and its caption as a pair"""
        element = pair['element']
        caption = pair['caption']
        
        # Base metadata
        metadata = {
            'id': f"{element.type.lower()}_{page_num}_{counter}",
            'type': element.type,
            'page_number': page_num,
            'element_number': counter,
            'confidence': float(element.score),
            'bbox': element.coordinates.tolist()
        }
        
        # Save element image
        element_dir = self.output_dir / element.type.lower()
        image_filename = f"{document_id}_{metadata['id']}.png"
        image_path = element_dir / image_filename
        
        element_image = image.crop(element.coordinates)
        element_image.save(image_path)
        metadata['file_path'] = str(image_path)
        
        # Process caption if exists
        if caption:
            caption_text = self.ocr_agent.detect(image.crop(caption.coordinates))
            metadata['caption'] = {
                'text': caption_text,
                'bbox': caption.coordinates.tolist(),
                'confidence': float(caption.score)
            }
        
        return metadata

    def analyze_document(self, file_path: Path) -> Dict:
        """Analyze document with unified section handling"""
        try:
            images = convert_from_path(str(file_path))
            document_id = file_path.stem
            
            # Single list for all sections
            document_sections = []
            
            for page_num, image in enumerate(images, 1):
                # Get image dimensions
                image_height = image.height
                
                # Detect layout
                layout = self.model.detect(image)
                
                # Find figure/table-caption pairs
                element_pairs = self.find_element_caption_pairs(layout, image_height)
                
                # Get text sections
                text_sections = [
                    block for block in layout 
                    if block.type in ["Title", "Text", "List"]
                ]
                
                # Combine all elements and sort by vertical position
                page_elements = []
                
                # Add text sections
                for block in text_sections:
                    page_elements.append({
                        'type': block.type,
                        'block': block,
                        'y_position': block.coordinates[1]
                    })
                
                # Add figures and tables with their captions
                for pair in element_pairs:
                    element = pair['element']
                    y_position = element.coordinates[1]
                    
                    page_elements.append({
                        'type': element.type,
                        'block': element,
                        'caption_block': pair['caption'],
                        'y_position': y_position,
                        'distance_score': pair['distance_score']
                    })
                
                # Sort all elements by vertical position
                page_elements.sort(key=lambda x: x['y_position'])
                
                # Process elements in order
                for idx, element in enumerate(page_elements):
                    section_metadata = {
                        'page_number': page_num,
                        'section_number': idx,  # Starts from 0 for each page
                        'type': element['type'],
                        'confidence': float(element['block'].score),
                        'bbox': element['block'].coordinates.tolist()
                    }
                    
                    if element['type'] in ["Figure", "Table"]:
                        # Process figure/table
                        element_dir = self.output_dir / element['type'].lower()
                        image_filename = f"{document_id}_p{page_num}_s{idx}.png"
                        image_path = element_dir / image_filename
                        
                        # Save element image
                        element_image = image.crop(element['block'].coordinates)
                        element_image.save(image_path)
                        
                        section_metadata.update({
                            'file_path': str(image_path),
                            'element_type': element['type']
                        })
                        
                        # Add caption if exists
                        if element.get('caption_block'):
                            caption_text = self.ocr_agent.detect(
                                image.crop(element['caption_block'].coordinates)
                            )
                            section_metadata['caption'] = {
                                'text': caption_text,
                                'bbox': element['caption_block'].coordinates.tolist(),
                                'confidence': float(element['caption_block'].score),
                                'distance_score': element['distance_score']
                            }
                    
                    else:  # Text sections
                        # Process text content
                        text = self.ocr_agent.detect(
                            image.crop(element['block'].coordinates)
                        )
                        
                        # Save text content
                        text_path = self.output_dir / 'sections' / f"{document_id}_p{page_num}_s{idx}.txt"
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(text)
                        
                        section_metadata.update({
                            'text': text,
                            'file_path': str(text_path)
                        })
                    
                    document_sections.append(section_metadata)
            
            # Generate metadata
            metadata = {
                'document_id': document_id,
                'total_pages': len(images),
                'total_sections': len(document_sections),
                'section_types': {
                    type_: len([s for s in document_sections if s['type'] == type_])
                    for type_ in ["Title", "Text", "List", "Figure", "Table"]
                },
                'sections_by_page': {
                    page: len([s for s in document_sections if s['page_number'] == page])
                    for page in range(1, len(images) + 1)
                }
            }
            
            # Save metadata
            metadata_path = self.output_dir / 'metadata' / f"{document_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'success': True,
                'document_id': document_id,
                'sections': document_sections,
                'metadata': metadata
            }
            
        except Exception as e:
            logging.error(f"Error analyzing document: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }