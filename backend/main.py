import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from anthropic import Anthropic
from pathlib import Path
from .utils.document_processor import DocumentAnalyzer
import shutil
import json

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # Browser access
        "http://frontend:3000",      # Container access
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure processed_documents directory exists
DOCUMENTS_DIR = Path("processed_documents")
DOCUMENTS_DIR.mkdir(exist_ok=True)

# Initialize services
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
document_analyzer = DocumentAnalyzer(output_dir="processed_documents")

# Mount processed documents directory for static file access
app.mount("/documents", StaticFiles(directory="processed_documents"), name="documents")

@app.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    """Analyze document structure and content"""
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"temp_{file.filename}")
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Analyze document
            results = document_analyzer.analyze_document(temp_path)
            
            if not results['success']:
                raise HTTPException(status_code=400, detail=results['error'])
            
            return results
            
        finally:
            # Cleanup temporary file
            if temp_path.exists():
                temp_path.unlink()
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}")
async def get_document_metadata(document_id: str):
    """Get document metadata"""
    try:
        metadata_path = Path(f"processed_documents/metadata/{document_id}_metadata.json")
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
            
        with open(metadata_path, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}/sections")
async def get_document_sections(
    document_id: str, 
    page: int = None, 
    section_type: str = None
):
    """Get document sections with optional filtering"""
    try:
        # Load document analysis
        metadata_path = Path(f"processed_documents/metadata/{document_id}_metadata.json")
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load all sections
        sections = []
        for p in range(1, metadata['total_pages'] + 1):
            for s in range(metadata['sections_by_page'][str(p)]):
                section_path = Path(f"processed_documents/sections/{document_id}_p{p}_s{s}.txt")
                if section_path.exists():
                    with open(section_path, 'r') as f:
                        sections.append({
                            'page': p,
                            'section': s,
                            'content': f.read()
                        })
        
        # Apply filters
        if page is not None:
            sections = [s for s in sections if s['page'] == page]
        if section_type is not None:
            sections = [s for s in sections if s['type'] == section_type]
            
        return sections
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}/figures")
async def get_document_figures(document_id: str, page: int = None):
    """Get document figures with their captions"""
    try:
        figures_dir = Path(f"processed_documents/figures")
        figures = []
        
        for figure_file in figures_dir.glob(f"{document_id}_p*_s*.png"):
            # Parse page and section numbers from filename
            parts = figure_file.stem.split('_')
            page_num = int(parts[-2][1:])
            
            if page is None or page_num == page:
                figures.append({
                    'page': page_num,
                    'path': f"/documents/figures/{figure_file.name}",
                    'caption': None  # Caption is stored in metadata
                })
        
        return sorted(figures, key=lambda x: (x['page']))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}/tables")
async def get_document_tables(document_id: str, page: int = None):
    """Get document tables with their captions"""
    try:
        tables_dir = Path(f"processed_documents/tables")
        tables = []
        
        for table_file in tables_dir.glob(f"{document_id}_p*_s*.png"):
            parts = table_file.stem.split('_')
            page_num = int(parts[-2][1:])
            
            if page is None or page_num == page:
                tables.append({
                    'page': page_num,
                    'path': f"/documents/tables/{table_file.name}",
                    'caption': None  # Caption is stored in metadata
                })
        
        return sorted(tables, key=lambda x: (x['page']))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_document(document_id: str):
    """Generate a summary using Claude"""
    try:
        # Get document sections
        sections = await get_document_sections(document_id)
        
        # Combine text content
        text = "\n\n".join(section['content'] for section in sections)
        
        # Generate summary with Claude
        response = anthropic.completions.create(
            model="claude-2.0",
            max_tokens_to_sample=300,
            prompt=f"Here's the content of a document. Please summarize it briefly:\n\n{text[:2000]}..."
        )
        
        return {"summary": response.completion}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
