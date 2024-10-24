import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update this with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    # Read the PDF file
    pdf_reader = PdfReader(file.file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Process the text with Anthropic's Haiku model
    response = anthropic.completions.create(
        model="claude-2.0",
        max_tokens_to_sample=300,
        prompt=f"Here's the content of a PDF file. Please summarize it briefly:\n\n{text[:1000]}..."
    )

    return {"summary": response.completion}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
