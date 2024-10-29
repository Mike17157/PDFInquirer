from typing import List, Dict, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
import os
import json
from pathlib import Path

class SectionContextualizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
    def process_document(self, document_sections: List[Dict]) -> Dict:
        """Process document sections to add contextual descriptions"""
        # First pass: Generate embeddings for text sections
        text_sections = []
        visual_sections = []
        
        for section in document_sections:
            if section['type'] in ['Text', 'Title', 'List']:
                text_sections.append({
                    'text': section.get('text', ''),
                    'page': section['page_number'],
                    'section': section['section_number']
                })
            elif section['type'] in ['Figure', 'Table']:
                visual_sections.append(section)
        
        # Generate embeddings and BM25 index
        embeddings = self._generate_embeddings([s['text'] for s in text_sections])
        bm25 = BM25Okapi([s['text'].split() for s in text_sections])
        
        # Second pass: Add context to figures and tables
        enriched_sections = []
        
        for section in document_sections:
            if section['type'] in ['Figure', 'Table']:
                # Get relevant context
                context = self._get_relevant_context(
                    section,
                    text_sections,
                    embeddings,
                    bm25
                )
                
                # Generate description using Claude
                description = self._generate_description(section, context)
                
                # Add context and description to section
                section['context'] = context
                section['description'] = description
            
            enriched_sections.append(section)
        
        return enriched_sections
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text sections"""
        return self.embedding_model.encode(texts)
    
    def _get_relevant_context(
        self,
        visual_section: Dict,
        text_sections: List[Dict],
        embeddings: np.ndarray,
        bm25: BM25Okapi,
        top_k: int = 20
    ) -> List[Dict]:
        """Get relevant context using hybrid search"""
        query = visual_section.get('caption', {}).get('text', '')
        if not query:
            return []
            
        # BM25 scores
        bm25_scores = bm25.get_scores(query.split())
        
        # Vector similarity scores
        query_embedding = self.embedding_model.encode(query)
        vector_scores = np.dot(embeddings, query_embedding)
        
        # Combine scores (normalized)
        bm25_norm = bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else bm25_scores
        vector_norm = vector_scores / np.max(vector_scores) if np.max(vector_scores) > 0 else vector_scores
        combined_scores = 0.5 * bm25_norm + 0.5 * vector_norm
        
        # Get top-k results
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        return [text_sections[i] for i in top_indices]
    
    def _generate_description(self, section: Dict, context: List[Dict]) -> str:
        """Generate contextual description using Claude"""
        context_text = "\n".join(c['text'] for c in context)
        
        prompt = f"""Given a {section['type'].lower()} with the following caption:
                    {section.get('caption', {}).get('text', 'No caption available')}

                    And this surrounding context from the document:
                    {context_text}

                    Please provide a brief, contextual description of what this {section['type'].lower()} shows or represents. 
                    Focus on connecting it to the main ideas in the surrounding text."""

        response = self.anthropic.messages.create(
            model="claude-3-sonnet-20241022",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

    def save_enriched_metadata(
        self,
        document_id: str,
        enriched_sections: List[Dict],
        output_dir: Path
    ):
        """Save enriched metadata to file"""
        metadata_path = output_dir / 'metadata' / f"{document_id}_enriched.json"
        with open(metadata_path, 'w') as f:
            json.dump(enriched_sections, f, indent=2)
