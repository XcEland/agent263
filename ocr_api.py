from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from mistralai import Mistral
import google.generativeai as genai
import os
import tempfile
import uvicorn
from dotenv import load_dotenv
import json
import re
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Mistral OCR API",
    description="Unified API for document OCR, image OCR, and document verification",
    version="3.0.0"
)

# Initialize Mistral client
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("MISTRAL_API_KEY environment variable not set")
client = Mistral(api_key=api_key)

# Initialize Google Gemini client for verification
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set")
genai.configure(api_key=google_api_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')   

# Field mapping instructions for the AI
FIELD_INSTRUCTIONS = {
    "personalDetails": {
        "countryOfBirth": "Look in Proof of Identity document (National ID) for country of birth",
        "citizenship": "Infer from countryOfBirth if not found (ZIMBABWE → ZIMBABWEAN)",
        "identificationType": "Determine from document type (NATIONAL ID or PASSPORT)",
        "idNumber": "Search all documents for ID number (especially Proof of Identity)",
        "dateOfBirth": "Search all documents for date of birth (format: YYYY-MM-DD)",
        "gender": "Infer from title if possible (MR/SIR → Male, MRS/MS → Female)",
        "title": "Search all documents for title (MR, MRS, MS, etc.)",
        "firstname": "Search all documents for first name",
        "lastname": "Search all documents for last name",
        "maritalStatus": "Only include if explicitly found (SINGLE, MARRIED, etc.)",
        "religion": "Only include if explicitly found",
        "race": "Only include if explicitly found",
        "numberOfDependents": "Only include if explicitly found",
        "highestLevelOfEducation": "Only include if explicitly found",
        "birthDistrict": "Search Proof of Identity for birth district"
    },
    "contactDetails": {
        "primaryMethodOfCommunication": "Only include if explicitly found",
        "email": "Search all documents for email address",
        "phoneNumber": "Search all documents for phone number",
        "telephoneNumber": "Search all documents for telephone number",
        "facebook": "Only include if explicitly found",
        "twitter": "Only include if explicitly found",
        "linkedin": "Only include if explicitly found",
        "skype": "Only include if explicitly found"
    },
    "addressDetails": {
        "addressType": "Default to RESIDENTIAL if found in Proof of Residence",
        "addressLine": "Extract from Proof of Residence",
        "street": "Extract from Proof of Residence",
        "suburb": "Extract from Proof of Residence",
        "city": "Extract from Proof of Residence",
        "country": "Infer from Proof of Identity if not found",
        "postalCode": "Only include if explicitly found",
        "periodOfResidenceInYears": "Only include if explicitly found",
        "periodOfResidenceInMonths": "Only include if explicitly found",
        "monthlyRentalAmount": "Only include if explicitly found",
        "homeOwnership": "Only include if explicitly found"
    },
    "employmentDetails": {
        "employerName": "Extract from Employment Letter",
        "phoneNumber": "Extract from Employment Letter",
        "telephoneNumber": "Extract from Employment Letter",
        "email": "Extract from Employment Letter",
        "address": "Extract from Employment Letter",
        "jobTitle": "Extract from Employment Letter",
        "industry": "Extract from Employment Letter",
        "monthlyGrossIncome": "Only include if explicitly found",
        "monthlyNetIncome": "Only include if explicitly found",
        "employmentType": "Only include if explicitly found",
        "employmentDate": "Only include if explicitly found (format: YYYY-MM-DD)",
        "employmentEndDate": "Only include if explicitly found (format: YYYY-MM-DD)",
        "sourceOfFunds": "Extract from Employment Letter or Proof of Income"
    }
}

def extract_details_from_documents(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract applicant details from document markdown using Google Gemini"""
    try:
        # Prepare document context
        document_context = "\n\n".join([
            f"DOCUMENT CATEGORY: {doc['category']}\nCONTENT:\n{doc['markdown'][:10000]}"
            for doc in documents
        ])
        
        # Create extraction prompt
        prompt = f"""
        ROLE: You are a bank branch consultant responsible for extracting applicant details from onboarding documents.
        TASK: Analyze the document content below and extract relevant information to populate the JSON structure.
        
        INSTRUCTIONS:
        1. BE STRICT: Only extract values that are EXPLICITLY stated in the documents. Do not guess or assume values.
        2. COMPARE DOCUMENTS: When the same field appears in multiple documents, verify consistency. If values conflict, use the value from the most authoritative document (Proof of Identity > Employment Letter > Proof of Residence).
        3. INFER ONLY WHEN LOGICAL: 
           - If countryOfBirth is "ZIMBABWE", set citizenship to "ZIMBABWEAN"
           - Map titles to gender: MR/SIR → Male, MRS/MS → Female
        4. FORMATTING:
           - All string values MUST BE IN UPPERCASE
           - Dates must be in YYYY-MM-DD format
           - Numbers should be in numerical format (not words)
        5. MISSING DATA: Leave fields blank if information is not found in any document.
        6. STRUCTURE: Return ONLY a JSON object with the exact structure specified below.
        
        DOCUMENT CONTENT:
        {document_context}
        
        FIELD MAPPING INSTRUCTIONS:
        {json.dumps(FIELD_INSTRUCTIONS, indent=2)}
        
        REQUIRED JSON STRUCTURE:
        {{
            "personalDetails": {{
                "countryOfBirth": "",
                "citizenship": "",
                "identificationType": "",
                "idNumber": "",
                "dateOfBirth": "",
                "gender": "",
                "title": "",
                "firstname": "",
                "lastname": "",
                "maritalStatus": "",
                "religion": "",
                "race": "",
                "numberOfDependents": 0,
                "highestLevelOfEducation": "",
                "birthDistrict": ""
            }},
            "contactDetails": {{
                "primaryMethodOfCommunication": "",
                "email": "",
                "phoneNumber": "",
                "telephoneNumber": "",
                "facebook": "",
                "twitter": "",
                "linkedin": "",
                "skype": ""
            }},
            "addressDetails": [
                {{
                    "addressType": "",
                    "addressLine": "",
                    "street": "",
                    "suburb": "",
                    "city": "",
                    "country": "",
                    "postalCode": "",
                    "periodOfResidenceInYears": "0",
                    "periodOfResidenceInMonths": "0",
                    "monthlyRentalAmount": "0",
                    "homeOwnership": ""
                }}
            ],
            "employmentDetails": {{
                "employerName": "",
                "phoneNumber": "",
                "telephoneNumber": "",
                "email": "",
                "address": "",
                "jobTitle": "",
                "industry": "",
                "monthlyGrossIncome": 0.0,
                "monthlyNetIncome": 0.0,
                "employmentType": "",
                "employmentDate": "",
                "employmentEndDate": "",
                "sourceOfFunds": [
                    {{
                        "source": "",
                        "currency": "",
                        "amount": 0.0
                    }}
                ]
            }}
        }}
        """
        
        # Get extraction from Gemini
        response = gemini_model.generate_content(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if not json_match:
            raise ValueError("No JSON found in Gemini response")
        
        extracted_data = json.loads(json_match.group())
        
        # Validate response structure
        required_sections = ["personalDetails", "contactDetails", 
                            "addressDetails", "employmentDetails"]
        if not all(section in extracted_data for section in required_sections):
            raise ValueError("Invalid extraction response format")
            
        return extracted_data
        
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {str(e)}")

@app.post("/extract-details")
async def extract_details(
    documents: List[Dict[str, Any]] = Body(..., description="List of OCR-processed documents")
):
    """
    Extract applicant details from OCR-processed documents
    
    Request Body:
    [
        {
            "category": "Proof of Identity",
            "markdown": "...",
            ... (other fields optional)
        },
        ...
    ]
    """
    try:
        # Validate input
        if not documents or not isinstance(documents, list):
            raise HTTPException(status_code=400, detail="Invalid documents format")
            
        # Extract details using AI
        extracted_data = extract_details_from_documents(documents)
        
        return JSONResponse(content=extracted_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 

# Define valid categories
VALID_CATEGORIES = [
    "Proof of Identity",
    "Proof of Residence",
    "Proof of Income",
    "Employment Letter",
    "Application Form"
]

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

def verify_document_category(category: str, markdown_content: str) -> dict:
    """Verify if document content matches the specified category"""
    try:
        # Truncate content to fit within token limits
        truncated_content = markdown_content[:15000]  # Keep first 15k characters
        
        # Create verification prompt
        prompt = f"""
        You are a bank branch consultant responsible for verifying that the provided document 
        matches the specified category. Analyze the document content and determine if it contains
        the required information for the category.
        
        Category: {category}
        Document Content:
        {truncated_content}
        
        Your verification should be strict. Only return a JSON response with these keys:
        - "verified": boolean (true only if document clearly matches the category)
        - "confidence": integer (0-100, confidence level in verification)
        - "reason": string (brief explanation of your decision)
        
        Category Requirements:
        - "Proof of Identity": Must contain government-issued ID details like full name, ID number, 
          date of birth, and photo identification. Examples: National ID, Passport.
        - "Proof of Residence": Must show name and physical address. Examples: utility bill,affidavit form,
          bank statement, lease agreement (must be recent - within 3 months).
        - "Proof of Income": Must show income details like salary amounts, pay periods, employer info. 
          Examples: payslips, tax returns, bank statements showing salary deposits.
        - "Employment Letter": Must be on company letterhead, contain employment details 
          (position, start date, salary), and be signed by employer.
        - "Application Form": Must be a filled application form with personal and financial details.
        """
        
        # Get verification from Mistral
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        verification = json.loads(response.choices[0].message.content)
        
        # Validate response structure
        if not all(key in verification for key in ["verified", "confidence", "reason"]):
            raise ValueError("Invalid verification response format")
            
        return verification
        
    except Exception as e:
        return {
            "verified": False,
            "confidence": 0,
            "reason": f"Verification failed: {str(e)}"
        }

@app.post("/ocr")
async def unified_ocr(
    category: str = Form(..., description="Document category for verification"),
    file: UploadFile = File(...)
):
    """
    Unified endpoint for OCR processing and verification
    
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
            "markdown": markdown_content,
            "pages": markdown_content.count('\n\n') + 1,  # Estimate page count
            "verification": verification
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Separate endpoints for backward compatibility
@app.post("/ocr/document")
async def ocr_document(
    category: str = Form(..., description="Document category for verification"),
    file: UploadFile = File(...)
):
    """Endpoint specifically for document OCR (PDF files) with verification"""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF documents are supported")
    return await unified_ocr(category, file)

@app.post("/ocr/image")
async def ocr_image(
    category: str = Form(..., description="Document category for verification"),
    file: UploadFile = File(...)
):
    """Endpoint specifically for image OCR (JPEG/PNG) with verification"""
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are supported")
    return await unified_ocr(category, file)

if __name__ == "__main__":
    uvicorn.run("ocr_api:app", host="0.0.0.0", port=8000, reload=True)