import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional, Union
import logging
from pathlib import Path
from scipy.stats import entropy
from pdf2image import convert_from_path

class ImageSectionProcessor:
    """Process images to detect and section based on text and visual elements."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.debug_dir = Path('debug_sections') if debug else None
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self.logger = logging.getLogger(__name__)
        if debug:
            self._setup_debug_directory()
        self.MAX_IMAGE_LONG_EDGE = 1568

    def _setup_debug_directory(self) -> None:
        if self.debug_dir:
            self.debug_dir.mkdir(exist_ok=True)
            self.logger.debug(f"Debug directory created at {self.debug_dir}")

    def _find_split_line(self, image: np.ndarray, orientation: str = 'horizontal') -> Optional[int]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        height, width = gray.shape
        start, end = (int(height * 0.3), int(height * 0.7)) if orientation == 'horizontal' else (int(width * 0.3), int(width * 0.7))
        entropies = [(entropy(np.histogram(gray[i, :] if orientation == 'horizontal' else gray[:, i], bins=256, range=(0, 256))[0] / gray.size + 1e-10), i) for i in range(start, end, 16)]
        if not entropies:
            return None
        min_entropy_line = min(entropies, key=lambda x: x[0])[1]
        if self.debug:
            debug_img = image.copy()
            cv2.line(debug_img, (0, min_entropy_line) if orientation == 'horizontal' else (min_entropy_line, 0), (width, min_entropy_line) if orientation == 'horizontal' else (min_entropy_line, height), (0, 255, 0), 2)
            cv2.imwrite(str(self.debug_dir / f'split_line_{orientation}.png'), debug_img)
        return min_entropy_line

    def split_large_image(self, image: np.ndarray) -> List[np.ndarray]:
        def split_section(section: np.ndarray) -> List[np.ndarray]:
            height, width = section.shape[:2]
            if max(width, height) <= self.MAX_IMAGE_LONG_EDGE:
                return [section]
            orientation = 'horizontal' if height > width else 'vertical'
            split_pos = self._find_split_line(section, orientation) or (height // 2 if orientation == 'horizontal' else width // 2)
            top_half, bottom_half = (section[:split_pos], section[split_pos:]) if orientation == 'horizontal' else (section[:, :split_pos], section[:, split_pos:])
            return split_section(top_half) + split_section(bottom_half)
        
        sections = split_section(image)
        if self.debug:
            for i, section in enumerate(sections):
                cv2.imwrite(str(self.debug_dir / f'split_section_{i}.png'), section)
                self.logger.debug(f"Section {i} size: {section.shape[:2]}")
        return sections

    def detect_sections(self, image: np.ndarray) -> Dict[str, List[Tuple]]:
        sections = {'text_blocks': [], 'visual_elements': []}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        initial_text_blocks = self._detect_text_blocks(gray)
        initial_visual_elements = self._detect_visual_elements(gray)
        for section_type, initial_sections in [('text_blocks', initial_text_blocks), ('visual_elements', initial_visual_elements)]:
            for x, y, w, h in initial_sections:
                if max(w, h) > self.MAX_IMAGE_LONG_EDGE:
                    section_image = image[y:y+h, x:x+w]
                    split_sections = self.split_large_image(section_image)
                    y_offset = y
                    for split in split_sections:
                        split_h, split_w = split.shape[:2]
                        sections[section_type].append((x, y_offset, split_w, split_h))
                        y_offset += split_h
                else:
                    sections[section_type].append((x, y, w, h))
        if self.debug:
            self._save_debug_image(image, sections, 'detected_sections.png')
            self.logger.debug(f"Detected {len(sections['text_blocks'])} text blocks and {len(sections['visual_elements'])} visual elements")
        return sections

    def save_sections(self, image: np.ndarray, sections: Dict[str, List[Tuple]], output_dir: str) -> List[str]:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        saved_paths = []
        for section_type in ['text_blocks', 'visual_elements']:
            for i, (x, y, w, h) in enumerate(sections[section_type]):
                section = image[y:y+h, x:x+w]
                section_rgb = cv2.cvtColor(section, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(section_rgb)
                output_file = output_path / f'{section_type[:-1]}_section_{i}.png'
                pil_image.save(str(output_file), 'PNG')
                saved_paths.append(str(output_file))
                if self.debug:
                    self.logger.debug(f"Saved {section_type[:-1]} section {i} to {output_file}")
        return saved_paths

    def _detect_text_blocks(self, gray_image: np.ndarray) -> List[Tuple]:
        preprocessed = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if self.debug:
            cv2.imwrite(str(self.debug_dir / 'preprocessed_text.png'), preprocessed)
        data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT)
        blocks = [(data['left'][i], data['top'][i], data['width'][i], data['height'][i]) for i in range(len(data['level'])) if int(data['level'][i]) == 4 and float(data['conf'][i]) > 30]
        return self._merge_overlapping_blocks(blocks)

    def _detect_visual_elements(self, gray_image: np.ndarray) -> List[Tuple]:
        return self._florence_region_proposal(gray_image)

    def _merge_overlapping_blocks(self, blocks: List[Tuple]) -> List[Tuple]:
        if not blocks:
            return []
        blocks = sorted(blocks, key=lambda b: b[0])
        merged = [list(blocks[0])]
        for block in blocks[1:]:
            if self._blocks_overlap(merged[-1], block):
                merged[-1][2] = max(merged[-1][0] + merged[-1][2], block[0] + block[2]) - merged[-1][0]
                merged[-1][3] = max(merged[-1][1] + merged[-1][3], block[1] + block[3]) - merged[-1][1]
            else:
                merged.append(list(block))
        return [tuple(m) for m in merged]

    def _blocks_overlap(self, block1: List[int], block2: Tuple[int, int, int, int]) -> bool:
        x1, y1, w1, h1 = block1
        x2, y2, w2, h2 = block2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def _save_debug_image(self, image: np.ndarray, sections: Dict[str, List[Tuple]], filename: str) -> None:
        if self.debug:
            debug_image = image.copy()
            for section_type, color in [('text_blocks', (255, 0, 0)), ('visual_elements', (0, 255, 0))]:
                for x, y, w, h in sections[section_type]:
                    cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 2)
            debug_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(debug_rgb).save(str(self.debug_dir / filename))

    def convert_pdf_to_continuous_image(self, pdf_path: str) -> np.ndarray:
        pages = convert_from_path(pdf_path)
        page_arrays = [np.array(page) for page in pages]
        max_width = max(page.shape[1] for page in page_arrays)
        standardized_pages = [np.pad(page, ((0, 0), (0, max_width - page.shape[1]), (0, 0)), 'constant', constant_values=255) for page in page_arrays]
        continuous_image = np.vstack(standardized_pages)
        if self.debug:
            cv2.imwrite(str(self.debug_dir / 'continuous_image.png'), cv2.cvtColor(continuous_image, cv2.COLOR_RGB2BGR))
        return continuous_image

    def _calculate_page_number(self, y_coordinate: int, page_heights: List[int]) -> int:
        cumulative_height = 0
        for page_num, height in enumerate(page_heights, 1):
            cumulative_height += height
            if y_coordinate < cumulative_height:
                return page_num
        return len(page_heights)

    def process_document(self, input_path: Union[str, np.ndarray], output_dir: str) -> List[Dict]:
        if isinstance(input_path, str) and input_path.lower().endswith('.pdf'):
            continuous_image = self.convert_pdf_to_continuous_image(input_path)
            page_heights = [page.shape[0] for page in convert_from_path(input_path)]
        else:
            continuous_image = input_path if isinstance(input_path, np.ndarray) else cv2.imread(input_path)
            page_heights = [continuous_image.shape[0]]
        sections = self.detect_sections(continuous_image)
        processed_sections = []
        section_number = 1
        for section_type in ['text_blocks', 'visual_elements']:
            for x, y, w, h in sections[section_type]:
                page_number = self._calculate_page_number(y, page_heights)
                section_image = continuous_image[y:y+h, x:x+w]
                output_file = Path(output_dir) / f'section_{section_number}_page_{page_number}.png'
                section_rgb = cv2.cvtColor(section_image, cv2.COLOR_BGR2RGB)
                Image.fromarray(section_rgb).save(str(output_file))
                section = {
                    'section_number': section_number,
                    'page_number': page_number,
                    'type': section_type.rstrip('s'),
                    'coordinates': {'x': x, 'y': y, 'width': w, 'height': h},
                    'file_path': str(output_file)
                }
                if section_type == 'text_blocks':
                    section['text'] = pytesseract.image_to_string(section_image)
                processed_sections.append(section)
                section_number += 1
                if self.debug:
                    self.logger.debug(f"Processed section {section_number} on page {page_number}")
        return processed_sections

    def segment_image(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return self._florence_region_proposal(image)

    def _florence_region_proposal(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return [(50, 50, 100, 100), (120, 120, 80, 80)]

    def merge_overlapping_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        merged_boxes = []
        for box in boxes:
            merged = False
            for i, mbox in enumerate(merged_boxes):
                if self._iou(box, mbox) > 0.5:
                    merged_boxes[i] = self._merge_boxes(box, mbox)
                    merged = True
                    break
            if not merged:
                merged_boxes.append(box)
        return merged_boxes

    def _iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area else 0

    def _merge_boxes(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x = min(x1, x2)
        y = min(y1, y2)
        w = max(x1 + w1, x2 + w2) - x
        h = max(y1 + h1, y2 + h2) - y
        return (x, y, w, h)

    def classify_and_extract(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> Dict[str, List[str]]:
        results = {'text': [], 'images': []}
        for box in boxes:
            x, y, w, h = box
            region = image[y:y+h, x:x+w]
            is_text = self._classify_region(region)
            if is_text:
                text = self._florence_ocr(region)
                results['text'].append(text)
            else:
                results['images'].append(region)
        return results

    def _classify_region(self, region: np.ndarray) -> bool:
        return True

    def _florence_ocr(self, region: np.ndarray) -> str:
        return "Extracted text from region"

# Example usage
processor = ImageSectionProcessor(debug=True)
image = cv2.imread('example_image.png')
regions = processor.segment_image(image)
merged_regions = processor.merge_overlapping_boxes(regions)
extracted_data = processor.classify_and_extract(image, merged_regions)
