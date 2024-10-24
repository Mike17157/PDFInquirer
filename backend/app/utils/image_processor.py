import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import List, Tuple, Dict, Optional, Union
import logging
from pathlib import Path
from scipy.stats import entropy
import math
from pdf2image import convert_from_path  # Add this import

class ImageSectionProcessor:
    """Process images to detect and section based on text and visual elements."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize the ImageSectionProcessor.
        
        Args:
            debug (bool): If True, saves intermediate processing steps and enables detailed logging
        """
        self.debug = debug
        self.debug_dir = Path('debug_sections') if debug else None
        
        # Set up logging
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if debug:
            self._setup_debug_directory()
        
        self.MAX_IMAGE_LONG_EDGE = 1568
        self.current_page = 1
        self.current_section = 1
        
    def _setup_debug_directory(self) -> None:
        """Create debug directory if it doesn't exist."""
        if self.debug_dir:
            self.debug_dir.mkdir(exist_ok=True)
            self.logger.debug(f"Debug directory created at {self.debug_dir}")

    def _find_split_line(self, image: np.ndarray, orientation: str = 'horizontal') -> Optional[int]:
        """
        Find the best line to split the image by looking for low-entropy regions.
        
        Args:
            image: Input image as numpy array
            orientation: 'horizontal' or 'vertical'
            
        Returns:
            Position of the split line, or None if no good split found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        height, width = gray.shape
        
        # Define the region to search for splits (middle 40% of the image)
        if orientation == 'horizontal':
            start = int(height * 0.3)
            end = int(height * 0.7)
        else:
            start = int(width * 0.3)
            end = int(width * 0.7)
            
        # Calculate entropy for each line in the search region
        entropies = []
        for i in range(start, end):
            if orientation == 'horizontal':
                line = gray[i, :]
            else:
                line = gray[:, i]
            
            # Calculate local entropy using a histogram
            hist = np.histogram(line, bins=256, range=(0, 256))[0]
            hist = hist / hist.sum()
            line_entropy = entropy(hist + 1e-10)  # Add small constant to avoid log(0)
            
            # Weight entropy by distance from center
            center = height/2 if orientation == 'horizontal' else width/2
            distance_from_center = abs(i - center)
            weighted_entropy = line_entropy * (1 + distance_from_center / center)
            
            entropies.append((weighted_entropy, i))
            
        if not entropies:
            return None
            
        # Find the line with minimum entropy
        min_entropy_line = min(entropies, key=lambda x: x[0])[1]
        
        if self.debug:
            debug_img = image.copy()
            if orientation == 'horizontal':
                cv2.line(debug_img, (0, min_entropy_line), (width, min_entropy_line), (0, 255, 0), 2)
            else:
                cv2.line(debug_img, (min_entropy_line, 0), (min_entropy_line, height), (0, 255, 0), 2)
            cv2.imwrite(str(self.debug_dir / f'split_line_{orientation}.png'), debug_img)
            
        return min_entropy_line

    def split_large_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Recursively split image if it exceeds maximum dimensions.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of image sections
        """
        def split_section(section: np.ndarray) -> List[np.ndarray]:
            height, width = section.shape[:2]
            
            # Base case: section is within size limit
            if max(width, height) <= self.MAX_IMAGE_LONG_EDGE:
                return [section]
            
            # Determine split orientation based on aspect ratio
            orientation = 'horizontal' if height > width else 'vertical'
            split_pos = self._find_split_line(section, orientation)
            
            if split_pos is None:
                self.logger.warning("Could not find suitable split line. Forcing split at midpoint.")
                split_pos = height // 2 if orientation == 'horizontal' else width // 2
            
            # Split the section
            if orientation == 'horizontal':
                top_half = section[:split_pos]
                bottom_half = section[split_pos:]
            else:
                top_half = section[:, :split_pos]
                bottom_half = section[:, split_pos:]
            
            # Recursively split each half if needed
            result = []
            result.extend(split_section(top_half))
            result.extend(split_section(bottom_half))
            
            return result
        
        # Start the recursive splitting process
        sections = split_section(image)
        
        if self.debug:
            for i, section in enumerate(sections):
                cv2.imwrite(str(self.debug_dir / f'split_section_{i}.png'), section)
                self.logger.debug(f"Section {i} size: {section.shape[:2]}")
        
        return sections

    def detect_sections(self, image: np.ndarray) -> Dict[str, List[Tuple]]:
        """
        Detect different sections in the image and split if they exceed size limits.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing lists of section coordinates
        """
        # Initialize sections dictionary
        sections = {
            'text_blocks': [],
            'visual_elements': []
        }
        
        # Detect initial sections
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        initial_text_blocks = self._detect_text_blocks(gray)
        initial_visual_elements = self._detect_visual_elements(gray)
        
        # Process text blocks
        for x, y, w, h in initial_text_blocks:
            if max(w, h) > self.MAX_IMAGE_LONG_EDGE:
                # Extract and split large section
                section_image = image[y:y+h, x:x+w]
                split_sections = self.split_large_image(section_image)
                
                # Add each split section with adjusted coordinates
                y_offset = y
                for split in split_sections:
                    split_h, split_w = split.shape[:2]
                    sections['text_blocks'].append((x, y_offset, split_w, split_h))
                    y_offset += split_h
            else:
                sections['text_blocks'].append((x, y, w, h))
        
        # Process visual elements
        for x, y, w, h in initial_visual_elements:
            if max(w, h) > self.MAX_IMAGE_LONG_EDGE:
                # Extract and split large section
                section_image = image[y:y+h, x:x+w]
                split_sections = self.split_large_image(section_image)
                
                # Add each split section with adjusted coordinates
                y_offset = y
                for split in split_sections:
                    split_h, split_w = split.shape[:2]
                    sections['visual_elements'].append((x, y_offset, split_w, split_h))
                    y_offset += split_h
            else:
                sections['visual_elements'].append((x, y, w, h))
        
        if self.debug:
            self._save_debug_image(image, sections, 'detected_sections.png')
            self.logger.debug(f"Detected {len(sections['text_blocks'])} text blocks and "
                            f"{len(sections['visual_elements'])} visual elements")
        
        return sections

    def save_sections(self, image: np.ndarray, sections: Dict[str, List[Tuple]], output_dir: str) -> List[str]:
        """
        Save detected sections as individual PNG files with original colors.
        
        Args:
            image: Original color image
            sections: Dictionary of detected sections
            output_dir: Directory to save the sections
            
        Returns:
            List of paths to saved section images
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        saved_paths = []

        # Process and save text blocks
        for i, (x, y, w, h) in enumerate(sections['text_blocks']):
            # Extract section from original color image
            section = image[y:y+h, x:x+w]
            
            # Convert from BGR to RGB
            section_rgb = cv2.cvtColor(section, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image for better PNG saving
            pil_image = Image.fromarray(section_rgb)
            
            # Save as PNG with original colors
            output_file = output_path / f'text_section_{i}.png'
            pil_image.save(str(output_file), 'PNG')
            saved_paths.append(str(output_file))
            
            if self.debug:
                self.logger.debug(f"Saved text section {i} to {output_file}")

        # Process and save visual elements
        for i, (x, y, w, h) in enumerate(sections['visual_elements']):
            # Extract section from original color image
            section = image[y:y+h, x:x+w]
            
            # Convert from BGR to RGB
            section_rgb = cv2.cvtColor(section, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image for better PNG saving
            pil_image = Image.fromarray(section_rgb)
            
            # Save as PNG with original colors
            output_file = output_path / f'visual_section_{i}.png'
            pil_image.save(str(output_file), 'PNG')
            saved_paths.append(str(output_file))
            
            if self.debug:
                self.logger.debug(f"Saved visual section {i} to {output_file}")

        return saved_paths

    def _detect_text_blocks(self, gray_image: np.ndarray) -> List[Tuple]:
        """Detect text blocks using grayscale image."""
        # Improve text detection by applying some preprocessing
        preprocessed = cv2.threshold(gray_image, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        if self.debug:
            cv2.imwrite(str(self.debug_dir / 'preprocessed_text.png'), preprocessed)

        data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT)
        
        blocks = []
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            if int(data['level'][i]) == 4:  # Text block level
                (x, y, w, h) = (data['left'][i], data['top'][i], 
                               data['width'][i], data['height'][i])
                confidence = float(data['conf'][i])
                
                # Filter out low-confidence detections
                if confidence > 30:  # Adjust threshold as needed
                    blocks.append((x, y, w, h))
        
        return self._merge_overlapping_blocks(blocks)

    def _detect_visual_elements(self, gray_image: np.ndarray) -> List[Tuple]:
        """
        Detect visual elements like graphs and charts using grayscale image.
        
        Returns:
            List of tuples containing coordinates (x, y, w, h) of visual elements
        """
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray_image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        if self.debug:
            cv2.imwrite(str(self.debug_dir / 'binary_visual.png'), binary)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        visual_elements = []
        min_area = 1000  # Minimum area to consider as a visual element
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                visual_elements.append((x, y, w, h))
        
        return visual_elements

    def _merge_overlapping_blocks(self, blocks: List[Tuple]) -> List[Tuple]:
        """Merge overlapping blocks into larger sections."""
        if not blocks:
            return []
            
        # Sort blocks by x coordinate
        blocks = sorted(blocks, key=lambda b: b[0])
        
        merged = []
        current = list(blocks[0])
        
        for block in blocks[1:]:
            if self._blocks_overlap(current, block):
                # Merge blocks
                current[2] = max(current[0] + current[2], block[0] + block[2]) - current[0]
                current[3] = max(current[1] + current[3], block[1] + block[3]) - current[1]
            else:
                merged.append(tuple(current))
                current = list(block)
        
        merged.append(tuple(current))
        return merged

    def _blocks_overlap(self, block1: List[int], block2: Tuple[int, int, int, int]) -> bool:
        """Check if two blocks overlap."""
        x1, y1, w1, h1 = block1
        x2, y2, w2, h2 = block2
        
        return not (x1 + w1 < x2 or x2 + w2 < x1 or
                   y1 + h1 < y2 or y2 + h2 < y1)

    def _save_debug_image(self, image: np.ndarray, sections: Dict[str, List[Tuple]], 
                         filename: str) -> None:
        """Save debug image with detected sections highlighted."""
        if self.debug:
            debug_image = image.copy()
            
            # Draw text blocks in blue
            for x, y, w, h in sections['text_blocks']:
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw visual elements in green
            for x, y, w, h in sections['visual_elements']:
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Convert to RGB for consistent color display
            debug_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
            debug_pil = Image.fromarray(debug_rgb)
            debug_pil.save(str(self.debug_dir / filename))

    def convert_pdf_to_continuous_image(self, pdf_path: str) -> np.ndarray:
        """
        Convert PDF to a single continuous image.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Numpy array of concatenated pages
        """
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path)
        
        # Convert PIL images to numpy arrays
        page_arrays = [np.array(page) for page in pages]
        
        # Get maximum width
        max_width = max(page.shape[1] for page in page_arrays)
        
        # Standardize widths and concatenate vertically
        standardized_pages = []
        for page in page_arrays:
            if page.shape[1] < max_width:
                # Pad width to match max_width
                pad_width = max_width - page.shape[1]
                page = np.pad(page, ((0, 0), (0, pad_width), (0, 0)), 'constant', constant_values=255)
            standardized_pages.append(page)
        
        continuous_image = np.vstack(standardized_pages)
        
        if self.debug:
            cv2.imwrite(str(self.debug_dir / 'continuous_image.png'), 
                       cv2.cvtColor(continuous_image, cv2.COLOR_RGB2BGR))
            
        return continuous_image

    def _calculate_page_number(self, y_coordinate: int, page_heights: List[int]) -> int:
        """
        Calculate page number based on y-coordinate and page heights.
        
        Args:
            y_coordinate: Y-coordinate in continuous image
            page_heights: List of individual page heights
            
        Returns:
            Page number (1-based)
        """
        cumulative_height = 0
        for page_num, height in enumerate(page_heights, 1):
            cumulative_height += height
            if y_coordinate < cumulative_height:
                return page_num
        return len(page_heights)

    def process_document(self, input_path: Union[str, np.ndarray], 
                        output_dir: str) -> List[Dict]:
        """
        Process a PDF or image document and split into indexed sections.
        
        Args:
            input_path: Path to PDF file or numpy array of image
            output_dir: Directory to save section images
            
        Returns:
            List of dictionaries containing section information
        """
        # Convert input to continuous image
        if isinstance(input_path, str) and input_path.lower().endswith('.pdf'):
            continuous_image = self.convert_pdf_to_continuous_image(input_path)
            page_heights = [page.shape[0] for page in convert_from_path(input_path)]
        else:
            continuous_image = input_path if isinstance(input_path, np.ndarray) else cv2.imread(input_path)
            page_heights = [continuous_image.shape[0]]

        # Detect and split sections
        sections = self.detect_sections(continuous_image)
        
        # Process sections with metadata
        processed_sections = []
        section_number = 1
        
        for section_type in ['text_blocks', 'visual_elements']:
            for x, y, w, h in sections[section_type]:
                page_number = self._calculate_page_number(y, page_heights)
                
                section = {
                    'section_number': section_number,
                    'page_number': page_number,
                    'type': section_type.rstrip('s'),  # Remove plural
                    'coordinates': {
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h
                    }
                }
                
                # Extract and save section image
                section_image = continuous_image[y:y+h, x:x+w]
                output_file = Path(output_dir) / f'section_{section_number}_page_{page_number}.png'
                
                # Convert BGR to RGB for PIL
                section_rgb = cv2.cvtColor(section_image, cv2.COLOR_BGR2RGB)
                Image.fromarray(section_rgb).save(str(output_file))
                
                section['file_path'] = str(output_file)
                
                # Extract text if it's a text block
                if section_type == 'text_blocks':
                    section['text'] = pytesseract.image_to_string(section_image)
                
                processed_sections.append(section)
                section_number += 1
                
                if self.debug:
                    self.logger.debug(f"Processed section {section_number} on page {page_number}")
        
        return processed_sections

