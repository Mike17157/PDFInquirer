from typing import List, Dict
import pytesseract
import cv2
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import logging

class EmbeddingsGenerator:
    def __init__(self, debug: bool = False):
        """
        Initialize the embeddings generator.
        
        Args:
            debug (bool): Enable debug logging
        """
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Initialize Anthropic client for summaries
        self.anthropic = Anthropic()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _generate_section_summary(self, text: str, nearby_sections: List[str]) -> str:
        """
        Generate a summary of a text section with context from nearby sections.
        
        Args:
            text: Text to summarize
            nearby_sections: List of text from up to 5 nearby sections
        
        Returns:
            Generated summary
        """
        context = "\n".join([
            "Previous sections discuss:" if nearby_sections else "",
            *[f"- {s}" for s in nearby_sections[-5:]]
        ])
        
        prompt = f"""Please provide a concise summary of this text section. Focus on key points and how they relate to nearby content.

                Context:
                {context}

                Text to summarize:
                {text}

                Summary:"""

        response = self.anthropic.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

    def _get_nearby_sections(self, sections: List[Dict], current_idx: int) -> List[str]:
        """Get summaries from up to 5 nearby sections."""
        start_idx = max(0, current_idx - 2)
        end_idx = min(len(sections), current_idx + 3)
        
        nearby = []
        for i in range(start_idx, end_idx):
            if i != current_idx and 'summary' in sections[i]:
                nearby.append(sections[i]['summary'])
        
        return nearby

    def process_sections_with_context(self, sections: List[Dict]) -> List[Dict]:
        """
        Process sections with contextual summaries and embeddings.
        
        Args:
            sections: List of dictionaries containing section info
        
        Returns:
            List of dictionaries containing section info, summaries, and embeddings
        """
        processed_sections = []
        
        # First pass: Generate initial summaries without context
        for section in sections:
            if section['type'] == 'text':
                initial_summary = self._generate_section_summary(section['text'], [])
                section['summary'] = initial_summary
            processed_sections.append(section)
        
        # Second pass: Update summaries with context
        for i, section in enumerate(processed_sections):
            if section['type'] == 'text':
                # Get nearby section summaries
                nearby_summaries = self._get_nearby_sections(processed_sections, i)
                
                # Generate contextual summary
                section['summary'] = self._generate_section_summary(
                    section['text'], 
                    nearby_summaries
                )
                
                # Generate embedding from summary
                section['embedding'] = self.embedding_model.encode(
                    section['summary']
                ).tolist()
                
                if self.debug:
                    self.logger.debug(f"Processed section {i}:")
                    self.logger.debug(f"Summary: {section['summary'][:100]}...")
                    self.logger.debug(f"Context from {len(nearby_summaries)} nearby sections")
        
        return processed_sections
