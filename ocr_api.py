from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from mistralai import Mistral
import google.generativeai as genai
import os
import tempfile
import uvicorn
from dotenv import load_dotenv
import json
import re
import httpx
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Mistral OCR API",
    description="Unified API for document OCR, image OCR, and document verification",
    version="3.0.0",
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    openapi_tags=[
        {
            "name": "OCR Processing",
            "description": "Endpoints for document OCR and verification",
        },
        {
            "name": "Document Management",
            "description": "Endpoints for file upload and retrieval",
        },
        {
            "name": "Data Extraction",
            "description": "Endpoints for extracting structured data from documents",
        },
        {
            "name": "System Health",
            "description": "Service health monitoring",
        },
    ]
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom error responses
    openapi_schema["components"]["schemas"]["HTTPError"] = {
        "type": "object",
        "properties": {
            "detail": {"type": "string", "example": "Error details"}
        }
    }
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


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
           - Zimbawean ID number must follow the format: XX-XXXXXXX [A-Z] XX or XX-XXXXXX [A-Z] XX
            Example: 08-123456 D 53 or 08-1234567 D 53 or 02-100482 G 03
            Components:
            XX: Two digits representing the year of registration
            XXXXXXX or XXXXXX: A serial number
            [A-Z]: A single letter, possibly indicating the place of registration
            XX: Two digits, possibly a check digit AA-AAAAAA A AA or AA-AAAAAA A AA  (12/13 digits), 
            Sanitize the ID number to remove any spaces or special characters. For example, "08-123456 D 53" should be stored/returned as "08123456D53".
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

# @app.post("/api/v1/agents/documents/extract-details")
@app.post(
    "/api/v1/agents/documents/extract-details",
    tags=["Data Extraction"],
    summary="Extract structured data from OCR-processed documents",
    description="""Processes OCR-processed documents through AI models to extract structured applicant information.
    
**Process Flow:**
1. Accepts multiple OCR-processed documents
2. Uses Google Gemini for data extraction
3. Returns structured JSON with validated fields
    
**Data Validation Rules:**
- All string values are UPPERCASE
- Dates in YYYY-MM-DD format
- ID numbers sanitized (remove spaces/dashes)
- Strict extraction (only explicit values)
    """,
    response_description="Structured applicant details",
    responses={
        200: {
            "description": "Successful extraction",
            "content": {
                "application/json": {
                    "example": {
                        "personalDetails": {
                            "countryOfBirth": "ZIMBABWE",
                            "citizenship": "ZIMBABWEAN",
                            # ... other fields ...
                        }
                    }
                }
            }
        },
        400: {"description": "Invalid request format"},
        500: {"description": "Processing error"}
    }
)
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

def process_file(file: UploadFile, content_type: str) -> tuple:
    """Process file through Mistral OCR API and return (markdown, file_id, file_url)"""
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
        
        # Store file ID for response
        file_id = uploaded_file.id
        
        # Get signed URL - NEW: Capture file URL
        file_url_obj = client.files.get_signed_url(file_id=file_id)
        file_url = file_url_obj.url
        
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
                doc_type: file_url  # Use the signed URL
            },
            include_image_base64=False
        )
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Combine markdown from all pages
        markdown_content = "\n\n".join([page.markdown for page in response.pages])
        
        return markdown_content, file_id, file_url  # Return file URL
    
    except Exception as e:
        # Clean up temp file if exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
    
# def verify_document_category(category: str, markdown_content: str) -> dict:
#     """Verify if document content matches the specified category"""
#     try:
#         # Truncate content to fit within token limits
#         truncated_content = markdown_content[:15000]  # Keep first 15k characters
        
#         # Create verification prompt
#         prompt = f"""
#         You are a bank branch consultant responsible for verifying that the provided document 
#         matches the specified category. Analyze the document content and determine if it contains
#         the required information for the category.
        
#         Category: {category}
#         Document Content:
#         {truncated_content}
        
#         Your verification should be strict. Only return a JSON response with these keys:
#         - "verified": boolean (true only if document clearly matches the category)
#         - "confidence": integer (0-100, confidence level in verification)
#         - "reason": string (brief explanation of your decision)
        
#         Category Requirements:
#         - "Proof of Identity": Must contain government-issued ID details like full name, ID number, 
#           date of birth, and photo identification. Examples: National ID, Passport.
#         - "Proof of Residence": Must show name and physical address. Examples: utility bill,affidavit form,
#           bank statement, lease agreement (must be recent - within 3 months).
#         - "Proof of Income": Must show income details like salary amounts, pay periods, employer info. 
#           Examples: payslips, tax returns, bank statements showing salary deposits.
#         - "Employment Letter": Must be on company letterhead, contain employment details 
#           (position, start date, salary), and be signed by employer.
#         - "Application Form": Must be a filled application form with personal and financial details.
#         """
        
#         # Get verification from Mistral
#         response = client.chat.complete(
#             model="mistral-large-latest",
#             messages=[{"role": "user", "content": prompt}],
#             response_format={"type": "json_object"}
#         )
        
#         # Parse JSON response
#         verification = json.loads(response.choices[0].message.content)
        
#         # Validate response structure
#         if not all(key in verification for key in ["verified", "confidence", "reason"]):
#             raise ValueError("Invalid verification response format")
            
#         return verification
        
#     except Exception as e:
#         return {
#             "verified": False,
#             "confidence": 0,
#             "reason": f"Verification failed: {str(e)}"
#         }

def verify_document_category(category: str, markdown_content: str) -> dict:
    """Verify if document content matches the specified category and suggest correct category"""
    try:
        # Truncate content to fit within token limits
        truncated_content = markdown_content[:15000]  # Keep first 15k characters
        
        # Create verification prompt with category suggestion
        prompt = f"""
        You are a bank branch consultant responsible for document verification. Perform these tasks:
        1. Analyze the document content and determine if it contains
        the required information for the category.
        
        Category: {category}
        Document Content: {truncated_content}

        2. Verify if the document matches the specified category: {category}
        3. If it doesn't match, determine the correct category from: {", ".join(VALID_CATEGORIES)}
        4. Provide a confidence score (0-100)
        5. Explain your reasoning
        6. if the document does not match the category, give the most appropriate reason for the mismatch and the reason must be consise and presentable to the client in a single to two sentences.
        
        
        Your verification should be strict. Only return a JSON response with these keys:
        - "verified": boolean (true only if document clearly matches the category)
        - "confidence": integer (0-100 confidence level)
        - "reason": string (brief explanation)
        - "correct_category": string (the most appropriate category)
        - "initial_category": string (the provided category by the client: {category})
        
        Category Requirements:
        - "Proof of Identity": Government-issued ID with full name, ID number, date of birth
        - "Proof of Residence": Shows name and physical address (utility bill, bank statement, lease, affidavit)
        - "Proof of Income": Shows income details (salary amounts, pay periods, employer info)
        - "Employment Letter": Company letterhead with employment details, signed by employer
        - "Application Form": Filled application form with personal/financial details
        
        Document Content:
        {truncated_content}
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
        required_keys = ["verified", "confidence", "reason", "correct_category"]
        if not all(key in verification for key in required_keys):
            raise ValueError("Invalid verification response format")
            
        # Ensure correct_category is valid
        if verification["correct_category"] not in VALID_CATEGORIES:
            verification["correct_category"] = category
            
        return verification
        
    except Exception as e:
        return {
            "verified": False,
            "confidence": 0,
            "reason": f"Verification failed: {str(e)}",
            "correct_category": category
        }
    
# @app.post("/api/v1/agents/ocr/verify-document")
@app.post(
    "/api/v1/agents/ocr/verify-document",
    tags=["OCR Processing"],
    summary="OCR processing with document verification",
    description="""Processes documents/images through OCR and verifies document category.
    
**Features:**
- PDF and image support (JPEG/PNG)
- Automatic category verification
- Content-based category correction
- Returns cleaned markdown content
    
**Categories:** 
- Proof of Identity
- Proof of Residence
- Proof of Income
- Employment Letter
- Application Form
    """,
    response_description="OCR results with verification status",
    responses={
        200: {
            "description": "OCR processed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "category": "Proof of Identity",
                        "filename": "id_document.pdf",
                        "content_type": "application/pdf",
                        "ocr_type": "document",
                        "pages": 1,
                        "verification": {
                            "verified": True,
                            "confidence": 95,
                            "reason": "Document contains required identity fields",
                            "correct_category": "Proof of Identity"
                        },
                        "file_id": "file-abc123",
                        "file_url": "https://signed.url/document",
                        "view_url": "/file-view/file-abc123"
                    }
                }
            }
        },
        400: {"description": "Invalid category or file type"},
        500: {"description": "Processing error"}
    }
)
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
        # Process file through OCR - now returns file_url
        markdown_content, file_id, file_url = process_file(file, file.content_type)
        
        # Verify document category
        verification = verify_document_category(category, markdown_content)
        
        # Determine the correct category to return
        corrected_category = verification.get("correct_category", category)

        return JSONResponse(content={
            "category": corrected_category,
            "filename": file.filename,
            "content_type": file.content_type,
            "ocr_type": "document" if file.content_type == "application/pdf" else "image",
            "markdown": markdown_content,
            "pages": markdown_content.count('\n\n') + 1,  # Estimate page count
            "verification": verification,
            "file_id": file_id,
            "file_url": file_url,  # Direct access URL
            "view_url": f"/file-view/{file_id}"  # Safe viewing URL
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Separate endpoints for backward compatibility
@app.post("/api/v1/agents/ocr/document")
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

# @app.get("/api/v1/agents/ocr/documents/{file_id}")
@app.get(
    "/api/v1/agents/ocr/documents/{file_id}",
    tags=["Document Management"],
    summary="Retrieve stored document",
    description="""Access document stored in Mistral system.
    
**Options:**
- Redirect to signed URL (default)
- Download content directly (with ?download=true)
    """,
    responses={
        307: {"description": "Redirect to signed document URL"},
        200: {
            "description": "Document content when download=true",
            "content": {"application/json": {}}
        },
        404: {"description": "Document not found"}
    }
)
async def get_document(
    file_id: str, 
    download: Optional[bool] = False
):
    """
    Access a document stored on Mistral by its file ID
    
    Parameters:
    - file_id: Mistral file identifier
    - download: Set to true to download content instead of redirecting
    
    Returns:
    - Redirect to signed URL (default)
    - File content if download=true
    """
    try:
        # Get signed URL from Mistral
        file_url = client.files.get_signed_url(file_id=file_id)
        
        if download:
            # Download file content
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(file_url.url)
                response.raise_for_status()
                
                return JSONResponse(
                    content={
                        "file_id": file_id,
                        "content": response.text,
                        "content_type": response.headers.get("Content-Type", "application/octet-stream")
                    }
                )
        else:
            # Redirect to signed URL
            return RedirectResponse(url=file_url.url)
            
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found or inaccessible: {str(e)}"
        )

# @app.get("/api/v1/agents/ocr/documents/{file_id}/info")
@app.get(
    "/api/v1/agents/ocr/documents/{file_id}/info",
    tags=["Document Management"],
    summary="Get document metadata",
    description="Retrieve metadata about stored document",
    responses={
        200: {
            "description": "Document metadata",
            "content": {
                "application/json": {
                    "example": {
                        "file_id": "file-abc123",
                        "filename": "document.pdf",
                        "purpose": "ocr",
                        "created_at": "2023-10-05T12:30:45Z",
                        "object": "file",
                        "status_details": "processed",
                        "size_bytes": 45678
                    }
                }
            }
        },
        404: {"description": "Document not found"}
    }
)
async def get_document_info(file_id: str):
    """
    Get metadata about a document stored on Mistral
    
    Parameters:
    - file_id: Mistral file identifier
    
    Returns:
    - Document metadata
    """
    try:
        # Retrieve file information
        file_info = client.files.retrieve(file_id=file_id)
        
        # Convert created_at to ISO format
        created_at = file_info.created_at
        if isinstance(created_at, int):
            created_at = datetime.utcfromtimestamp(created_at).isoformat() + "Z"
        elif hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat()
        else:
            created_at = str(created_at)
        
        # Build response with safe attribute access
        return JSONResponse(content={
            "file_id": file_info.id,
            "filename": file_info.filename,
            "purpose": file_info.purpose,
            "created_at": created_at,
            "object": file_info.object,
            # "status": file_info.status,
            "status_details": getattr(file_info, "status_details", None),
            # Size information might not be available in all API versions
            "size_bytes": getattr(file_info, "bytes", None),
        })
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {str(e)}"
        )

# @app.post("/api/v1/agents/documents/upload-file")
@app.post(
    "/api/v1/agents/documents/upload-file",
    tags=["Document Management"],
    summary="Upload file to Mistral",
    description="Store file in Mistral system and get access URLs",
    response_description="File metadata and access URLs",
    responses={
        200: {
            "description": "File uploaded successfully",
            "content": {
                "application/json": {
                    "example": {
                        "file_id": "file-xyz789",
                        "file_url": "https://signed.url/file",
                        "view_url": "/file-view/file-xyz789",
                        "content_type": "image/png",
                        "filename": "id_photo.png"
                    }
                }
            }
        },
        400: {"description": "Invalid file type"},
        500: {"description": "Upload failed"}
    }
)
async def upload_file(
    file: UploadFile = File(..., description="File to upload (PDF, JPEG, PNG)")
):
    """
    Upload a file to Mistral and return file metadata
    
    Supports:
    - PDF documents (application/pdf)
    - JPEG images (image/jpeg)
    - PNG images (image/png)
    
    Returns:
    - file_id: Mistral's file identifier
    - file_url: Temporary signed URL for direct access
    - view_url: URL for browser viewing
    - content_type: Detected file type
    - filename: Original filename
    """
    # Validate content type
    valid_types = ["application/pdf", "image/jpeg", "image/png"]
    if file.content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {', '.join(valid_types)}"
        )
    
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
        
        # Get file metadata
        file_id = uploaded_file.id
        
        # Get signed URL
        file_url_obj = client.files.get_signed_url(file_id=file_id)
        file_url = file_url_obj.url
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return JSONResponse(content={
            "file_id": file_id,
            "file_url": file_url,
            "view_url": f"/file-view/{file_id}",
            "content_type": file.content_type,
            "filename": file.filename
        })
    
    except Exception as e:
        # Clean up temp file if exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )
       
# @app.get("/api/v1/agents/documents/file-view/{file_id}")
@app.get(
    "/api/v1/agents/documents/file-view/{file_id}",
    tags=["Document Management"],
    summary="View file in browser",
    description="Render document/image in browser with inline content-disposition",
    responses={
        200: {
            "description": "File content",
            "content": {"image/png": {}, "application/pdf": {}}
        },
        404: {"description": "File not found"}
    }
)
async def view_file(file_id: str):
    """
    View a file in the browser without downloading
    
    Parameters:
    - file_id: Mistral file identifier
    
    Returns:
    - File content with inline Content-Disposition
    """
    try:
        # Get signed URL from Mistral
        file_url = client.files.get_signed_url(file_id=file_id)
        
        async with httpx.AsyncClient() as http_client:
            # Fetch file headers to determine content type
            head_response = await http_client.head(file_url.url)
            head_response.raise_for_status()
            
            content_type = head_response.headers.get("Content-Type", "application/octet-stream")
            
            # Fetch actual content
            response = await http_client.get(file_url.url)
            response.raise_for_status()
            
            # Return with inline content disposition
            return Response(
                content=response.content,
                media_type=content_type,
                headers={"Content-Disposition": "inline"}
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"File access failed: {str(e)}"
        )


# @app.get("/api/v1/agents/health-check")
@app.get(
    "/api/v1/agents/health-check",
    tags=["System Health"],
    summary="Service health check",
    description="Verify API service availability",
    response_description="Service status",
    responses={
        200: {
            "description": "Service status",
            "content": {
                "application/json": {
                    "example": {
                        "status": "active", 
                        "message": "services are running"
                    }
                }
            }
        }
    }
)
def health_check():
    return {"status": "active", "message": "automation-agents services are running"}

if __name__ == "__main__":
    uvicorn.run("ocr_api:app", host="0.0.0.0", port=8000, reload=True)