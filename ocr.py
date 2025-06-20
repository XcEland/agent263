from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from mistralai import Mistral
import google.generativeai as genai
import os
import tempfile
import uvicorn
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Document Verification API",
    description="OCR processing with Mistral and document verification with Google Gemini",
    version="3.0.0"
)

# Initialize Mistral client for OCR
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise RuntimeError("MISTRAL_API_KEY environment variable not set")
mistral_client = Mistral(api_key=mistral_api_key)

# Initialize Google Gemini client for verification
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set")
genai.configure(api_key=google_api_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Define valid categories
VALID_CATEGORIES = [
    "Proof of Identity",
    "Proof of Residence",
    "Proof of Income",
    "Employment Letter",
    "Application Form"
]

# Category-specific verification instructions
VERIFICATION_PROMPTS = {
    "Proof of Identity": """
    Verify if this is a valid government-issued identity document. 
    It MUST contain:
    - Full name of individual
    - Unique identification number
    - Date of birth
    - Photograph of individual
    - Issue date and/or expiration date
    - Issuing authority (e.g., government agency)
    
    Acceptable documents: National ID, Passport, Driver's License
    """,
    
    "Proof of Residence": """
    Verify if this is a valid proof of residence document. 
    It MUST contain:
    - Full name of individual
    - Complete physical address (street, city, postal code)
    - Date of issue (must be within last 3 months)
    - Issuing entity name and contact information
    
    Acceptable documents: Utility bill, Bank statement, Lease agreement
    """,
    
    "Proof of Income": """
    Verify if this is a valid proof of income document. 
    It MUST contain:
    - Full name of individual
    - Employer name
    - Income amount (monthly or annual)
    - Date range or pay period
    - Document date (within last 3 months)
    
    Acceptable documents: Payslip, Tax return, Bank statements showing salary deposits
    """,
    
    "Employment Letter": """
    Verify if this is a valid employment verification letter. 
    It MUST contain:
    - Company letterhead
    - Full name of employee
    - Employment start date
    - Job position/title
    - Salary information
    - Contact information of issuer
    - Signature of authorized representative
    """,
    
    "Application Form": """
    Verify if this is a completed application form. 
    It MUST contain:
    - Personal details section (name, contact info)
    - Financial information section
    - Signature and date fields completed
    - Relevant checkboxes selected
    """
}

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
            uploaded_file = mistral_client.files.upload(
                file={
                    "fileName": file.filename,
                    "content": f.read()
                },
                purpose="ocr"
            )
        
        # Get signed URL
        file_url = mistral_client.files.get_signed_url(file_id=uploaded_file.id)
        
        # Determine document type
        if content_type == "application/pdf":
            doc_type = "document_url"
        elif content_type in ["image/jpeg", "image/png"]:
            doc_type = "image_url"
        else:
            raise ValueError("Unsupported file type")
        
        # Process OCR
        response = mistral_client.ocr.process(
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

def verify_document_category(category: str, markdown_content: str) -> dict:
    """Verify if document content matches the specified category using Google Gemini"""
    try:
        # Get category-specific instructions
        instructions = VERIFICATION_PROMPTS.get(category, "")
        
        # Create verification prompt
        prompt = f"""
        ROLE: You are a bank branch consultant responsible for document verification.
        TASK: Analyze the document content below and determine if it matches a "{category}" document.
        
        VERIFICATION CRITERIA:
        {instructions}
        
        DOCUMENT CONTENT:
        {markdown_content[:15000]}  <!-- Truncated to 15k chars -->
        
        RESPONSE FORMAT: Return ONLY a JSON object with these keys:
        - "verified": boolean (true ONLY if document clearly matches all category requirements)
        - "confidence": integer (0-100, your confidence in verification)
        - "reason": string (brief explanation of verification decision)
        - "missing_fields": array of strings (any required fields missing)
        """
        
        # Get verification from Gemini
        response = gemini_model.generate_content(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in Gemini response")
        
        verification = json.loads(json_match.group())
        
        # Validate response structure
        required_keys = ["verified", "confidence", "reason", "missing_fields"]
        if not all(key in verification for key in required_keys):
            raise ValueError("Invalid verification response format")
            
        return verification
        
    except Exception as e:
        return {
            "verified": False,
            "confidence": 0,
            "reason": f"Verification failed: {str(e)}",
            "missing_fields": []
        }

@app.post("/verify-document")
async def verify_document(
    category: str = Form(..., description="Document category for verification"),
    file: UploadFile = File(...)
):
    """
    Endpoint for document verification with OCR processing
    
    Supports:
    - PDF documents (application/pdf)
    - JPEG images (image/jpeg)
    - PNG images (image/png)
    
    Categories:
    - Proof of Identity
    - Proof of Residence
    - Proof of Income
    - Employment Letter
    - Application Form
    """
    # Validate category
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Valid categories: {', '.join(VALID_CATEGORIES)}"
        )
    
    # Validate content type
    valid_types = ["application/pdf", "image/jpeg", "image/png"]
    if file.content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {', '.join(valid_types)}"
        )
    
    try:
        # Process file through OCR
        markdown_content = process_file(file, file.content_type)
        
        # Verify document category
        verification = verify_document_category(category, markdown_content)
        
        return JSONResponse(content={
            "category": category,
            "filename": file.filename,
            "content_type": file.content_type,
            "ocr_type": "document" if file.content_type == "application/pdf" else "image",
            "pages": markdown_content.count('\n\n') + 1,  # Estimate page count
            "verification": verification
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("ocr_api:app", host="0.0.0.0", port=8000, reload=True)