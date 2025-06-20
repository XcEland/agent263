from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from mistralai import Mistral
import os
import tempfile
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Mistral OCR API",
    description="Unified API for document and image OCR using Mistral AI",
    version="2.0.0"
)

# Initialize Mistral client
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("MISTRAL_API_KEY environment variable not set")
client = Mistral(api_key=api_key)

def process_file(file: UploadFile, content_type: str) -> str:
    """Process file through Mistral OCR API"""
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = file.file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Upload to Mistral
        with open(tmp_path, "rb") as f:
            uploaded_file = client.files.upload(
                file={
                    "fileName": file.filename,
                    "content": f.read()
                },
                purpose="ocr"
            )
        
        # Get signed URL
        file_url = client.files.get_signed_url(file_id=uploaded_file.id)
        
        # Determine document type
        if content_type == "application/pdf":
            doc_type = "document_url"
        elif content_type in ["image/jpeg", "image/png"]:
            doc_type = "image_url"
        else:
            raise ValueError("Unsupported file type")
        
        # Process OCR
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": doc_type,
                doc_type: file_url.url
            },
            include_image_base64=False
        )
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Combine markdown from all pages
        return "\n\n".join([page.markdown for page in response.pages])
    
    except Exception as e:
        # Clean up temp file if exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.post("/ocr")
async def unified_ocr(file: UploadFile = File(...)):
    """
    Unified endpoint for OCR processing of both documents and images.
    
    Supports:
    - PDF documents (application/pdf)
    - JPEG images (image/jpeg)
    - PNG images (image/png)
    """
    # Validate content type
    valid_types = ["application/pdf", "image/jpeg", "image/png"]
    if file.content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {', '.join(valid_types)}"
        )
    
    try:
        markdown_content = process_file(file, file.content_type)
        return JSONResponse(content={
            "filename": file.filename,
            "content_type": file.content_type,
            "ocr_type": "document" if file.content_type == "application/pdf" else "image",
            "markdown": markdown_content,
            "pages": markdown_content.count('\n\n') + 1  # Estimate page count
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Keep separate endpoints for specific use cases
@app.post("/ocr/document")
async def ocr_document(file: UploadFile = File(...)):
    """Endpoint specifically for document OCR (PDF files)"""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF documents are supported")
    return await unified_ocr(file)

@app.post("/ocr/image")
async def ocr_image(file: UploadFile = File(...)):
    """Endpoint specifically for image OCR (JPEG/PNG)"""
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are supported")
    return await unified_ocr(file)

if __name__ == "__main__":
    uvicorn.run("ocr_api:app", host="0.0.0.0", port=8000, reload=True)