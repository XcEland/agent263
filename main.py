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
    
# Define valid modules
VALID_MODULES = [
    "Machine Learning", "Data Structures and Algorithms", 
    "Database Systems", "Operating Systems",
    "Computer Networks", "Software Engineering",
    "Artificial Intelligence", "Cloud Computing",
    "Web Development", "Cybersecurity Fundamentals"
]

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

def assess_submitted_assignment(module: str, markdown_content: str) -> dict:
    """Assess student assignment with consistent mark calculation"""
    try:
        # Truncate content to fit within token limits
        truncated_content = markdown_content[:15000]
        
        # Create assessment prompt with strict mark consistency requirement
        prompt = f"""
        ROLE: You are an experienced educator assessing student assignments.
        TASK: Evaluate the submitted assignment ensuring mark consistency.
        
        INSTRUCTIONS:
        1. Verify if the content matches the module: {module}
        2. If content doesn't match:
           - Identify actual module it fits
           - Set is_correct_module=False
           - Provide clear feedback about mismatch
        3. If content is illegible/unclear:
           - Set confidence_assessment_score=0
           - Ask student to redo and resubmit
        4. For valid submissions:
           - Identify and assess each question individually
           - Assign marks to each question (max per question specified)
           - Calculate total marks: SUM(question_marks)
           - Ensure total marks equals the final percentage score
           - For each question provide:
               * Feedback on correctness
               * Mark awarded (with max possible)
               * Specific improvement suggestions
        
        MARKING CONSISTENCY RULES:
        - Total marks MUST equal: SUM(question_marks)
        - Final percentage = (SUM(question_marks) / TOTAL_POSSIBLE_MARKS) * 100
        - Clearly specify max possible marks for each question
        
        ASSESSMENT CRITERIA:
        1. Content Accuracy (30%): Demonstrated understanding of concepts
        2. Critical Thinking (25%): Application of knowledge to solve problems
        3. Organization (20%): Logical structure and clarity of solutions
        4. Completeness (15%): All requirements addressed
        5. Presentation (10%): Readability and proper formatting
        
        MODULE: {module}
        ASSIGNMENT CONTENT:
        {truncated_content}
        
        RESPONSE FORMAT (JSON ONLY):
        {{
            "is_correct_module": boolean,
            "confidence_assessment_score": integer (0-100),
            "total_possible_marks": integer,
            "marks_achieved": integer,
            "marks_percentage": integer (0-100),
            "overall_feedback": string,
            "strengths": [string],
            "improvements": [string],
            "criteria": [
                {{
                    "criterion": string,
                    "score": integer,
                    "feedback": string
                }}
            ],
            "assessment_details": {{
                "question_1": {{
                    "max_marks": integer,
                    "awarded_marks": integer,
                    "feedback": string,
                    "improvement": string
                }},
                "question_2": {{
                    "max_marks": integer,
                    "awarded_marks": integer,
                    "feedback": string,
                    "improvement": string
                }},
                ...
            }},
            "detected_module": string (if mismatched),
            "mark_consistency_check": string
        }}
        """
        
        # Get assessment from Gemini
        response = gemini_model.generate_content(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if not json_match:
            raise ValueError("No JSON found in Gemini response")
        
        assessment = json.loads(json_match.group())
        
        # Validate response structure
        required_keys = ["is_correct_module", "confidence_assessment_score", 
                        "marks_percentage", "overall_feedback", "assessment_details"]
        if not all(key in assessment for key in required_keys):
            raise ValueError("Invalid assessment response format")
            
        # Perform server-side mark consistency check
        total_awarded = 0
        total_possible = 0
        
        for q, details in assessment.get("assessment_details", {}).items():
            total_awarded += details.get("awarded_marks", 0)
            total_possible += details.get("max_marks", 0)
        
        calculated_percentage = 0
        if total_possible > 0:
            calculated_percentage = round((total_awarded / total_possible) * 100)
        
        # Add consistency check note
        assessment["mark_consistency_check"] = (
            "Verified" if calculated_percentage == assessment.get("marks_percentage", 0)
            else f"Inconsistency detected: Calculated {calculated_percentage}% vs Reported {assessment.get('marks_percentage', 0)}%"
        )
            
        return assessment
        
    except Exception as e:
        return {
            "is_correct_module": False,
            "confidence_assessment_score": 0,
            "total_possible_marks": 0,
            "marks_achieved": 0,
            "marks_percentage": 0,
            "overall_feedback": f"Assessment failed: {str(e)}",
            "strengths": [],
            "improvements": ["Technical error occurred during assessment"],
            "criteria": [],
            "assessment_details": {},
            "mark_consistency_check": "Not performed due to error"
        }

@app.post(
    "/api/v1/agents/student/assessment",
    tags=["Student Assessment"],
    summary="Evaluate student assignments with mark consistency",
    description="""Assess student submissions through OCR with detailed evaluation.
    
**Features:**
- PDF and image support (JPEG/PNG)
- Module-matching verification
- Question-level feedback with mark consistency
- Server-side mark validation
    
**Modules:** 
Machine Learning, Data Structures and Algorithms, Database Systems, 
Operating Systems, Computer Networks, Software Engineering,
Artificial Intelligence, Cloud Computing, Web Development, 
Cybersecurity Fundamentals
    """,
    response_description="Detailed assignment evaluation",
    responses={
        200: {
            "description": "Assessment completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "module": "Data Structures and Algorithms",
                        "filename": "dsa_assignment.pdf",
                        "content_type": "application/pdf",
                        "ocr_type": "document",
                        "markdown": "# Assignment 3: Trees and Graphs\n\n1. Implement BFS...",
                        "pages": 4,
                        "assessment": {
                            "is_correct_module": True,
                            "confidence_assessment_score": 92,
                            "total_possible_marks": 50,
                            "marks_achieved": 42,
                            "marks_percentage": 84,
                            "overall_feedback": "Good implementation but needs optimization",
                            "strengths": ["Correct BFS implementation", "Proper graph representation"],
                            "improvements": ["Optimize time complexity", "Handle edge cases"],
                            "criteria": [
                                {
                                    "criterion": "Content Accuracy",
                                    "score": 28,
                                    "feedback": "Algorithms implemented correctly"
                                }
                            ],
                            "assessment_details": {
                                "question_1": {
                                    "max_marks": 15,
                                    "awarded_marks": 14,
                                    "feedback": "Correct BFS implementation",
                                    "improvement": "Add cycle detection"
                                },
                                "question_2": {
                                    "max_marks": 20,
                                    "awarded_marks": 18,
                                    "feedback": "Good Dijkstra implementation",
                                    "improvement": "Optimize priority queue usage"
                                },
                                "question_3": {
                                    "max_marks": 15,
                                    "awarded_marks": 10,
                                    "feedback": "Partial solution for topological sort",
                                    "improvement": "Handle disconnected graphs"
                                }
                            },
                            "mark_consistency_check": "Verified"
                        },
                        "file_id": "file-dsa456",
                        "file_url": "https://signed.url/document",
                        "view_url": "/file-view/file-dsa456"
                    }
                }
            }
        },
        400: {"description": "Invalid module or file type"},
        500: {"description": "Processing error"}
    }
)
async def assess_assignment(
    module: str = Form(..., description="Academic module for assessment"),
    file: UploadFile = File(...)
):
    """
    Evaluate student assignments with mark consistency validation
    
    Supports:
    - PDF documents (application/pdf)
    - JPEG images (image/jpeg)
    - PNG images (image/png)
    
    Modules:
    Machine Learning, Data Structures and Algorithms, Database Systems, 
    Operating Systems, Computer Networks, Software Engineering,
    Artificial Intelligence, Cloud Computing, Web Development, 
    Cybersecurity Fundamentals
    """
    # Validate module
    if module not in VALID_MODULES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid module. Valid modules: {', '.join(VALID_MODULES)}"
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
        markdown_content, file_id, file_url = process_file(file, file.content_type)
        
        # Assess the assignment
        assessment = assess_submitted_assignment(module, markdown_content)
        
        # Estimate page count
        page_count = markdown_content.count('\n\n') + 1
        
        return JSONResponse(content={
            "module": module,
            "filename": file.filename,
            "content_type": file.content_type,
            "ocr_type": "document" if file.content_type == "application/pdf" else "image",
            "markdown": markdown_content,
            "pages": page_count,
            "assessment": assessment,
            "file_id": file_id,
            "file_url": file_url,
            "view_url": f"/file-view/{file_id}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  
    

def assess_submitted_assignment(module: str, markdown_content: str, marking_scheme: Optional[Dict] = None) -> dict:
    """Assess student assignment with optional marking scheme"""
    try:
        # Truncate content to fit within token limits
        truncated_content = markdown_content[:15000]
        
        # Create assessment prompt with optional marking scheme
        prompt = f"""
        ROLE: You are an experienced educator assessing student assignments.
        TASK: Evaluate the submitted assignment using the provided marking scheme if available.
        
        INSTRUCTIONS:
        1. Verify if the content matches the module: {module}
        2. If content doesn't match:
           - Identify actual module it fits
           - Set is_correct_module=False
           - Provide clear feedback about mismatch
        3. If content is illegible/unclear:
           - Set confidence_assessment_score=0
           - Ask student to redo and resubmit
        4. For valid submissions:
           - Use the marking scheme if provided, otherwise use standard criteria
           - Identify and assess each question individually
           - Assign marks to each question (max per question specified)
           - Calculate total marks: SUM(question_marks)
           - Ensure total marks equals the final percentage score
           - For each question provide:
               * Feedback on correctness
               * Mark awarded (with max possible)
               * Specific improvement suggestions
        """
        
        # Add marking scheme to prompt if provided
        if marking_scheme:
            prompt += f"""
            
        MARKING SCHEME PROVIDED:
        {json.dumps(marking_scheme, indent=2)}
            """
        else:
            prompt += """
            
        STANDARD ASSESSMENT CRITERIA:
        1. Content Accuracy (30%): Demonstrated understanding of concepts
        2. Critical Thinking (25%): Application of knowledge to solve problems
        3. Organization (20%): Logical structure and clarity of solutions
        4. Completeness (15%): All requirements addressed
        5. Presentation (10%): Readability and proper formatting
            """
        
        # Add response format
        prompt += f"""
        
        MODULE: {module}
        ASSIGNMENT CONTENT:
        {truncated_content}
        
        RESPONSE FORMAT (JSON ONLY):
        {{
            "is_correct_module": boolean,
            "confidence_assessment_score": integer (0-100),
            "total_possible_marks": integer,
            "marks_achieved": integer,
            "marks_percentage": integer (0-100),
            "overall_feedback": string,
            "strengths": [string],
            "improvements": [string],
            "criteria": [
                {{
                    "criterion": string,
                    "score": integer,
                    "feedback": string
                }}
            ],
            "assessment_details": {{
                "question_1": {{
                    "max_marks": integer,
                    "awarded_marks": integer,
                    "feedback": string,
                    "improvement": string
                }},
                ...
            }},
            "detected_module": string (if mismatched),
            "mark_consistency_check": string,
            "marking_scheme_used": boolean
        }}
        """
        
        # Get assessment from Gemini
        response = gemini_model.generate_content(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if not json_match:
            raise ValueError("No JSON found in Gemini response")
        
        assessment = json.loads(json_match.group())
        
        # Validate response structure
        required_keys = ["is_correct_module", "confidence_assessment_score", 
                        "marks_percentage", "overall_feedback", "assessment_details"]
        if not all(key in assessment for key in required_keys):
            raise ValueError("Invalid assessment response format")
            
        # Perform server-side mark consistency check
        total_awarded = 0
        total_possible = 0
        
        for q, details in assessment.get("assessment_details", {}).items():
            total_awarded += details.get("awarded_marks", 0)
            total_possible += details.get("max_marks", 0)
        
        calculated_percentage = 0
        if total_possible > 0:
            calculated_percentage = round((total_awarded / total_possible) * 100)
        
        # Add consistency check note
        assessment["mark_consistency_check"] = (
            "Verified" if calculated_percentage == assessment.get("marks_percentage", 0)
            else f"Inconsistency detected: Calculated {calculated_percentage}% vs Reported {assessment.get('marks_percentage', 0)}%"
        )
        
        # Add flag indicating if marking scheme was used
        assessment["marking_scheme_used"] = marking_scheme is not None
            
        return assessment
        
    except Exception as e:
        return {
            "is_correct_module": False,
            "confidence_assessment_score": 0,
            "total_possible_marks": 0,
            "marks_achieved": 0,
            "marks_percentage": 0,
            "overall_feedback": f"Assessment failed: {str(e)}",
            "strengths": [],
            "improvements": ["Technical error occurred during assessment"],
            "criteria": [],
            "assessment_details": {},
            "mark_consistency_check": "Not performed due to error",
            "marking_scheme_used": False
        }

@app.post(
    "/api/v2/agents/student/assessment",
    tags=["Student Assessment"],
    summary="Evaluate assignments with custom marking schemes",
    description="""Assess student submissions with optional marking scheme.
    
**Features:**
- PDF and image support (JPEG/PNG)
- Module-matching verification
- Custom marking scheme support
- Question-level feedback with mark consistency
- Server-side mark validation
    
**Modules:** 
Machine Learning, Data Structures and Algorithms, Database Systems, 
Operating Systems, Computer Networks, Software Engineering,
Artificial Intelligence, Cloud Computing, Web Development, 
Cybersecurity Fundamentals
    """,
    response_description="Detailed assignment evaluation",
    responses={
        200: {
            "description": "Assessment completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "module": "Machine Learning",
                        "filename": "nn_assignment.pdf",
                        "content_type": "application/pdf",
                        "ocr_type": "document",
                        "markdown": "# Neural Networks Assignment\n\n1. Implement backpropagation...",
                        "pages": 5,
                        "marking_scheme": {
                            "question_1": {
                                "max_marks": 20,
                                "criteria": [
                                    {"name": "Correct implementation", "weight": 0.7},
                                    {"name": "Efficiency", "weight": 0.3}
                                ]
                            },
                            "question_2": {
                                "max_marks": 30,
                                "rubric": {
                                    "Excellent": {"min": 27, "description": "Full implementation with optimizations"},
                                    "Good": {"min": 21, "description": "Correct implementation"},
                                    "Partial": {"min": 15, "description": "Partial solution"}
                                }
                            }
                        },
                        "assessment": {
                            "is_correct_module": True,
                            "confidence_assessment_score": 88,
                            "total_possible_marks": 50,
                            "marks_achieved": 42,
                            "marks_percentage": 84,
                            "overall_feedback": "Good implementation but needs optimization",
                            "strengths": ["Correct backpropagation derivation", "Proper gradient checking"],
                            "improvements": ["Implement vectorized operations", "Add regularization"],
                            "criteria": [
                                {
                                    "criterion": "Content Accuracy",
                                    "score": 28,
                                    "feedback": "Algorithms implemented correctly"
                                }
                            ],
                            "assessment_details": {
                                "question_1": {
                                    "max_marks": 20,
                                    "awarded_marks": 18,
                                    "feedback": "Correct implementation but suboptimal efficiency",
                                    "improvement": "Use vectorized operations for efficiency"
                                },
                                "question_2": {
                                    "max_marks": 30,
                                    "awarded_marks": 24,
                                    "feedback": "Good solution but missing regularization",
                                    "improvement": "Add L2 regularization to prevent overfitting"
                                }
                            },
                            "mark_consistency_check": "Verified",
                            "marking_scheme_used": True
                        },
                        "file_id": "file-ml789",
                        "file_url": "https://signed.url/document",
                        "view_url": "/file-view/file-ml789"
                    }
                }
            }
        },
        400: {"description": "Invalid module or file type"},
        500: {"description": "Processing error"}
    }
)
async def assess_assignment(
    module: str = Form(..., description="Academic module for assessment"),
    file: UploadFile = File(...),
    marking_scheme: Optional[str] = Form(None, description="Optional marking scheme as JSON string")
):
    """
    Evaluate student assignments with optional marking scheme
    
    Supports:
    - PDF documents (application/pdf)
    - JPEG images (image/jpeg)
    - PNG images (image/png)
    
    Parameters:
    - module: Academic module being assessed
    - file: Student's assignment file
    - marking_scheme: Optional JSON string with custom marking scheme
    
    Modules:
    Machine Learning, Data Structures and Algorithms, Database Systems, 
    Operating Systems, Computer Networks, Software Engineering,
    Artificial Intelligence, Cloud Computing, Web Development, 
    Cybersecurity Fundamentals
    """
    # Validate module
    if module not in VALID_MODULES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid module. Valid modules: {', '.join(VALID_MODULES)}"
        )
    
    # Validate content type
    valid_types = ["application/pdf", "image/jpeg", "image/png"]
    if file.content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {', '.join(valid_types)}"
        )
    
    # Parse marking scheme if provided
    scheme_dict = None
    if marking_scheme:
        try:
            scheme_dict = json.loads(marking_scheme)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid marking scheme JSON: {str(e)}"
            )
    
    try:
        # Process file through OCR
        markdown_content, file_id, file_url = process_file(file, file.content_type)
        
        # Assess the assignment
        assessment = assess_submitted_assignment(module, markdown_content, scheme_dict)
        
        # Estimate page count
        page_count = markdown_content.count('\n\n') + 1
        
        # Prepare response
        response_data = {
            "module": module,
            "filename": file.filename,
            "content_type": file.content_type,
            "ocr_type": "document" if file.content_type == "application/pdf" else "image",
            "markdown": markdown_content,
            "pages": page_count,
            "assessment": assessment,
            "file_id": file_id,
            "file_url": file_url,
            "view_url": f"/file-view/{file_id}"
        }
        
        # Include marking scheme in response if provided
        if scheme_dict:
            response_data["marking_scheme"] = scheme_dict
        
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/api/v1/agents/ocr/general",
    tags=["OCR Processing"],
    summary="General-purpose OCR for documents and images",
    description="Perform OCR on multiple documents/images and return markdown content",
    response_description="OCR results for each document",
    responses={
        200: {
            "description": "OCR processed successfully",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "documentName": "document1.pdf",
                            "markdown": "# Document content...",
                            "file_id": "file-abc123",
                            "file_url": "https://signed.url/document",
                            "view_url": "/file-view/file-abc123"
                        },
                        {
                            "documentName": "image1.png",
                            "markdown": "Text from image...",
                            "file_id": "file-xyz456",
                            "file_url": "https://signed.url/image",
                            "view_url": "/file-view/file-xyz456"
                        }
                    ]
                }
            }
        },
        400: {"description": "Invalid file type"},
        500: {"description": "Processing error"}
    }
)
async def general_ocr(
    files: List[UploadFile] = File(..., description="Files to process (PDF, JPEG, PNG)")
):
    """
    Perform OCR on multiple documents/images
    
    Supports:
    - PDF documents (application/pdf)
    - JPEG images (image/jpeg)
    - PNG images (image/png)
    
    Returns OCR results for each file
    """
    # Validate content types
    valid_types = ["application/pdf", "image/jpeg", "image/png"]
    invalid_files = [f.filename for f in files if f.content_type not in valid_types]
    
    if invalid_files:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file types for: {', '.join(invalid_files)}. "
                   f"Supported types: {', '.join(valid_types)}"
        )
    
    try:
        results = []
        for file in files:
            # Process file through OCR
            markdown_content, file_id, file_url = process_file(file, file.content_type)
            
            results.append({
                "documentName": file.filename,
                "markdown": markdown_content,
                "file_id": file_id,
                "file_url": file_url,
                "view_url": f"/file-view/{file_id}"
            })
        
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.post(
    "/api/v1/agents/teacher/assessment-generation",
    tags=["Teacher Tools"],
    summary="Generate assessment questions",
    description="Create quiz/test questions based on specifications and reference materials",
    response_description="Generated questions in structured format",
    responses={
        200: {
            "description": "Questions generated successfully",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "text": "What is the time complexity of a binary search algorithm?",
                            "type": "multiple_choice",
                            "options": [
                                "O(1)",
                                "O(log n)",
                                "O(n)",
                                "O(n log n)"
                            ],
                            "correctAnswer": 1,
                            "explanation": "Binary search halves the search space each iteration, resulting in logarithmic time complexity.",
                            "difficulty": "easy",
                            "tags": ["algorithms", "complexity"]
                        },
                        {
                            "text": "Explain how a hash table handles collisions using open addressing.",
                            "type": "structured",
                            "correctAnswer": "When a collision occurs, open addressing searches for the next available slot in the hash table using a probing sequence.",
                            "explanation": "Open addressing resolves collisions by finding alternative slots within the same table rather than chaining.",
                            "difficulty": "medium",
                            "tags": ["data structures", "hashing"]
                        }
                    ]
                }
            }
        },
        400: {"description": "Invalid input parameters"},
        500: {"description": "Generation failed"}
    }
)
async def generate_assessment_questions(
    request_data: dict = Body(..., example={
        "difficulty": "medium",
        "questionTypes": "mixed",
        "numberOfQuestions": 10,
        "attributes": {
            "course": "Data Structures",
            "topic": "Trees and Graphs",
            "learningObjectives": ["BFS", "DFS", "Shortest Path"]
        },
        "referenceDocuments": [
            {
                "documentName": "lecture_notes.pdf",
                "markdown": "# Graph Algorithms\n\nBreadth-First Search (BFS) explores nodes level by level...",
            }
        ],
        "tags": ["graphs", "traversal", "algorithms"]
    })
):
    """
    Generate assessment questions based on specifications
    
    Parameters:
    - difficulty: easy/medium/hard (default: medium)
    - questionTypes: multiple_choice/structured/mixed (default: multiple_choice)
    - numberOfQuestions: Number of questions to generate (default: 5)
    - attributes: Key-value pairs describing assessment context
    - referenceDocuments: List of documents with markdown content
    - tags: Keywords to guide question generation
    
    Returns:
    - JSON array of question objects
    """
    try:
        # Extract and validate parameters
        difficulty = request_data.get("difficulty", "medium")
        if difficulty not in ["easy", "medium", "hard"]:
            raise HTTPException(status_code=400, detail="Invalid difficulty level")
        
        question_types = request_data.get("questionTypes", "multiple_choice")
        if question_types not in ["multiple_choice", "structured", "mixed"]:
            raise HTTPException(status_code=400, detail="Invalid question type")
        
        num_questions = request_data.get("numberOfQuestions", 5)
        if not isinstance(num_questions, int) or num_questions < 1 or num_questions > 50:
            raise HTTPException(status_code=400, detail="Number of questions must be 1-50")
        
        attributes = request_data.get("attributes", {})
        references = request_data.get("referenceDocuments", [])
        tags = request_data.get("tags", [])
        
        # Build context from reference documents
        reference_context = "\n\n".join(
            [f"DOCUMENT: {doc['documentName']}\nCONTENT:\n{doc['markdown'][:5000]}" 
             for doc in references]
        )
        
        # Create generation prompt
        prompt = f"""
        ROLE: Expert academic assessment designer
        TASK: Create {num_questions} high-quality assessment questions
        
        REQUIREMENTS:
        1. Difficulty: {difficulty}
        2. Question Types: {question_types}
        3. Key Attributes: {json.dumps(attributes)}
        4. Tags: {', '.join(tags)}
        
        INSTRUCTIONS:
        - Generate questions that assess understanding of key concepts
        - Vary question types appropriately for mixed requests
        - Ensure questions are unambiguous and test-relevant
        - Provide clear correct answers and explanations
        - Reference source materials when applicable
        - Format response as JSON array only
        
        CONTEXT MATERIALS:
        {reference_context}
        
        RESPONSE FORMAT:
        [
          {{
            "text": "Question text",
            "type": "multiple_choice|structured",
            "options": ["Option1", "Option2", ...],  // Only for multiple choice
            "correctAnswer": "Correct answer text or option index",
            "explanation": "Detailed explanation of answer",
            "difficulty": "easy|medium|hard",
            "tags": ["tag1", "tag2", ...]
          }},
          ...
        ]
        """
        
        # Get questions from Gemini
        response = gemini_model.generate_content(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response.text)
        if not json_match:
            raise ValueError("No valid JSON array found in response")
        
        questions = json.loads(json_match.group())
        
        # Validate response structure
        required_keys = ["text", "type", "correctAnswer", "explanation"]
        for i, q in enumerate(questions):
            if not all(key in q for key in required_keys):
                raise ValueError(f"Question {i+1} missing required fields")
            if q["type"] == "multiple_choice" and "options" not in q:
                raise ValueError(f"Multiple choice question {i+1} missing options")
        
        return JSONResponse(content=questions)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


@app.post(
    "/api/v2/agents/teacher/assessment-generation",
    tags=["Teacher Tools"],
    summary="Generate assessment questions with custom context",
    description="Create quiz/test questions based on specifications, reference materials, and custom context",
    response_description="Generated questions in structured format",
    responses={
        200: {
            "description": "Questions generated successfully",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "text": "What is the time complexity of a binary search algorithm?",
                            "type": "multiple_choice",
                            "options": [
                                "O(1)",
                                "O(log n)",
                                "O(n)",
                                "O(n log n)"
                            ],
                            "correctAnswer": 1,
                            "explanation": "Binary search halves the search space each iteration, resulting in logarithmic time complexity.",
                            "difficulty": "easy",
                            "tags": ["algorithms", "complexity"]
                        }
                    ]
                }
            }
        },
        400: {"description": "Invalid input parameters"},
        500: {"description": "Generation failed"}
    }
)
async def generate_assessment_questions(
    request_data: dict = Body(..., example={
        "context": "Focus on practical implementation questions with code examples",
        "difficulty": "medium",
        "questionTypes": "mixed",
        "numberOfQuestions": 5,
        "attributes": {
            "course": "Data Structures",
            "topic": "Trees and Graphs",
            "learningObjectives": ["BFS", "DFS", "Shortest Path"]
        },
        "referenceDocuments": [
            {
                "documentName": "lecture_notes.pdf",
                "markdown": "# Graph Algorithms\n\nBreadth-First Search (BFS) explores nodes level by level...",
            }
        ],
        "tags": ["graphs", "traversal", "algorithms"]
    })
):
    """
    Generate assessment questions based on specifications and custom context
    
    Parameters:
    - context: Additional instructions for question generation (optional)
    - difficulty: easy/medium/hard (default: medium)
    - questionTypes: multiple_choice/structured/mixed (default: multiple_choice)
    - numberOfQuestions: Number of questions to generate (default: 5)
    - attributes: Key-value pairs describing assessment context
    - referenceDocuments: List of documents with markdown content
    - tags: Keywords to guide question generation
    
    Returns:
    - JSON array of question objects
    """
    try:
        # Extract and validate parameters
        context = request_data.get("context", "")
        difficulty = request_data.get("difficulty", "medium")
        if difficulty not in ["easy", "medium", "hard"]:
            raise HTTPException(status_code=400, detail="Invalid difficulty level")
        
        question_types = request_data.get("questionTypes", "multiple_choice")
        if question_types not in ["multiple_choice", "structured", "mixed"]:
            raise HTTPException(status_code=400, detail="Invalid question type")
        
        num_questions = request_data.get("numberOfQuestions", 5)
        if not isinstance(num_questions, int) or num_questions < 1 or num_questions > 50:
            raise HTTPException(status_code=400, detail="Number of questions must be 1-50")
        
        attributes = request_data.get("attributes", {})
        references = request_data.get("referenceDocuments", [])
        tags = request_data.get("tags", [])
        
        # Build context from reference documents
        reference_context = "\n\n".join(
            [f"DOCUMENT: {doc['documentName']}\nCONTENT:\n{doc['markdown'][:5000]}" 
             for doc in references]
        )
        
        # Create generation prompt with custom context
        prompt = f"""
        ROLE: Expert academic assessment designer
        TASK: Create {num_questions} high-quality assessment questions
        
        REQUIREMENTS:
        1. Difficulty: {difficulty}
        2. Question Types: {question_types}
        3. Key Attributes: {json.dumps(attributes)}
        4. Tags: {', '.join(tags)}
        
        {"5. ADDITIONAL CONTEXT: " + context if context else ""}
        
        INSTRUCTIONS:
        - Generate questions that assess understanding of key concepts
        - Vary question types appropriately for mixed requests
        - Ensure questions are unambiguous and test-relevant
        - Provide clear correct answers and explanations
        - Reference source materials when applicable
        - Follow any specific instructions in the context
        - Format response as JSON array only
        
        CONTEXT MATERIALS:
        {reference_context}
        
        RESPONSE FORMAT:
        [
          {{
            "text": "Question text",
            "type": "multiple_choice|structured",
            "options": ["Option1", "Option2", ...],  // Only for multiple choice
            "correctAnswer": "Correct answer text or option index",
            "explanation": "Detailed explanation of answer",
            "difficulty": "easy|medium|hard",
            "tags": ["tag1", "tag2", ...]
          }},
          ...
        ]
        """
        
        # Get questions from Gemini
        response = gemini_model.generate_content(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response.text)
        if not json_match:
            raise ValueError("No valid JSON array found in response")
        
        questions = json.loads(json_match.group())
        
        # Validate response structure
        required_keys = ["text", "type", "correctAnswer", "explanation"]
        for i, q in enumerate(questions):
            if not all(key in q for key in required_keys):
                raise ValueError(f"Question {i+1} missing required fields")
            if q["type"] == "multiple_choice" and "options" not in q:
                raise ValueError(f"Multiple choice question {i+1} missing options")
        
        return JSONResponse(content=questions)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

@app.post(
    "/api/v1/agents/teacher/content-generation",
    tags=["Teacher Tools"],
    summary="Generate educational content for a topic",
    description="Create comprehensive learning materials based on specifications",
    response_description="Structured educational content",
    responses={
        200: {
            "description": "Content generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "topic": "Sorting Algorithms",
                        "content": "## Sorting Algorithms\n\nSorting algorithms arrange elements in a specific order...",
                        "further_research": [
                            "Explore hybrid sorting algorithms like Timsort",
                            "Research parallel sorting techniques for large datasets"
                        ],
                        "key_concepts": ["Bubble Sort", "Quick Sort", "Time Complexity"],
                        "learning_path": [
                            "1. Start with comparison-based sorts",
                            "2. Study divide-and-conquer approaches",
                            "3. Analyze time-space tradeoffs"
                        ]
                    }
                }
            }
        },
        500: {"description": "Content generation failed"}
    }
)
async def generate_educational_content(
    request_data: dict = Body(..., example={
        "topic": "Sorting Algorithms",
        "context": "Focus on practical implementations with code examples. Compare time complexity.",
        "attributes": {
            "audience": "Computer Science undergraduates",
            "depth": "Intermediate",
            "duration": "2-week module"
        },
        "referenceDocuments": [
            {
                "documentName": "algorithms_textbook.pdf",
                "markdown": "# Sorting\n\nBubble Sort: O(n²) time complexity..."
            }
        ],
        "tags": ["algorithms", "sorting", "complexity"]
    })
):
    """
    Generate comprehensive educational content for a specific topic
    
    Parameters:
    - topic: Subject matter to cover
    - context: Special instructions for content generation
    - attributes: Key-value pairs describing content requirements
    - referenceDocuments: Supporting materials
    - tags: Keywords for content focus
    
    Returns:
    - Structured educational content with further research suggestions
    """
    try:
        # Extract parameters
        topic = request_data.get("topic", "Unspecified Topic")
        context = request_data.get("context", "")
        attributes = request_data.get("attributes", {})
        references = request_data.get("referenceDocuments", [])
        tags = request_data.get("tags", [])
        
        # Build context from reference documents
        reference_context = "\n\n".join(
            [f"REFERENCE: {doc['documentName']}\n{doc['markdown'][:5000]}" 
             for doc in references]
        )
        
        # Create content generation prompt
        prompt = f"""
        ROLE: Expert educator and curriculum designer
        TASK: Create comprehensive educational content about: {topic}
        
        REQUIREMENTS:
        1. Audience: {attributes.get('audience', 'General audience')}
        2. Depth: {attributes.get('depth', 'Introductory')}
        3. Duration: {attributes.get('duration', 'Self-paced')}
        4. Tags: {', '.join(tags)}
        
        {f"SPECIAL INSTRUCTIONS: {context}" if context else ""}
        
        CONTENT STRUCTURE:
        - Engaging introduction to the topic
        - Clear explanations of key concepts
        - Practical examples and applications
        - Visual aids suggestions (diagrams, charts, etc.)
        - Code implementations where applicable
        - Real-world use cases
        - Summary of key takeaways
        
        FURTHER RESEARCH:
        - Provide 3-5 suggestions for deeper exploration
        - Include research topics and learning resources
        
        KEY CONCEPTS:
        - Identify 3-5 core concepts to highlight
        
        LEARNING PATH:
        - Suggest a structured learning sequence
        
        FORMAT REQUIREMENTS:
        - Use Markdown for formatting
        - Include section headings
        - Use bullet points for lists
        - Add code blocks where appropriate
        - Return JSON format specified below
        
        REFERENCE MATERIALS:
        {reference_context}
        
        RESPONSE FORMAT (JSON):
        {{
            "topic": "{topic}",
            "content": "Markdown formatted educational content",
            "further_research": [
                "Suggestion 1",
                "Suggestion 2",
                ...
            ],
            "key_concepts": ["Concept1", "Concept2", ...],
            "learning_path": ["Step 1", "Step 2", ...]
        }}
        """
        
        # Get content from Gemini
        response = gemini_model.generate_content(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if not json_match:
            raise ValueError("No valid JSON found in response")
        
        content = json.loads(json_match.group())
        
        # Validate response structure
        required_keys = ["topic", "content", "further_research", "key_concepts"]
        if not all(key in content for key in required_keys):
            raise ValueError("Content missing required fields")
            
        return JSONResponse(content=content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content generation failed: {str(e)}")

@app.post(
    "/api/v1/agents/teacher/plan-generation",
    tags=["Teacher Tools"],
    summary="Generate personalized study plan",
    description="Create a customized learning plan to improve student performance",
    response_description="Structured development plan",
    responses={
        200: {
            "description": "Plan generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "name": "Data Structures Mastery Plan",
                        "description": "Targeted plan to improve algorithm implementation and data structure concepts",
                        "progress": 0,
                        "potentialOverall": 95,
                        "eta": 12,
                        "performance": "Improving",
                        "skills": [
                            {
                                "name": "Data Structures",
                                "score": 60.8,
                                "subskills": [
                                    {
                                        "name": "Tree Traversal",
                                        "score": 90,
                                        "color": "yellow"
                                    },
                                    {
                                        "name": "Hash Tables",
                                        "score": 90,
                                        "color": "blue"
                                    }
                                ]
                            }
                        ],
                        "courseId": "CS301",
                        "steps": [
                            {
                                "title": "Complete Binary Tree Practice",
                                "type": "assignment",
                                "link": "https://example.com/binary-tree-practice",
                                "order": 1
                            }
                        ],
                        "milestones": [
                            "Week 1: Master basic data structures",
                            "Week 2: Implement complex algorithms"
                        ],
                        "resources": [
                            "Data Structures and Algorithms in Python (ebook)",
                            "Algorithm Visualization Tool"
                        ]
                    }
                }
            }
        },
        500: {"description": "Plan generation failed"}
    }
)
async def generate_study_plan(
    request_data: dict = Body(..., example={
        "firstName": "John",
        "lastName": "Doe",
        "courseName": "Advanced Algorithms",
        "courseID": "CS401",
        "currentOverallScore": "75%",
        "potentialOverallScore": "92%",
        "targetScore": "90%",
        "overallPerformance": "Below Average",
        "overallEngagement": "Medium",
        "attributeDetails": [
            {
                "name": "Dynamic Programming",
                "currentScore": "60%",
                "potentialScore": "85%",
                "targetScore": "90%",
                "gap": "30%",
                "weight": "40%"
            }
        ],
        "context": "Focus on practical coding exercises",
        "referenceDocuments": [
            {
                "documentName": "Algo_Reference.pdf",
                "markdown": "# Dynamic Programming\n\nKey concepts: Memoization, Tabulation..."
            }
        ]
    })
):
    """
    Generate personalized study plan for student improvement
    
    Parameters:
    - firstName: Student first name
    - lastName: Student last name
    - courseName: Course title
    - courseID: Course identifier
    - currentOverallScore: Current overall percentage
    - potentialOverallScore: Potential achievable score
    - targetScore: Target goal score
    - overallPerformance: Current performance level
    - overallEngagement: Student engagement level
    - attributeDetails: List of skill attributes with scores
    - context: Optional instructions for plan customization
    - referenceDocuments: Optional reference materials
    
    Returns:
    - Structured development plan with actionable steps
    """
    try:
        # Extract parameters
        first_name = request_data["firstName"]
        last_name = request_data["lastName"]
        course_name = request_data["courseName"]
        course_id = request_data["courseID"]
        current_score = float(request_data["currentOverallScore"].strip('%'))
        target_score = float(request_data["targetScore"].strip('%'))
        performance = request_data["overallPerformance"]
        engagement = request_data["overallEngagement"]
        attributes = request_data["attributeDetails"]
        context = request_data.get("context", "")
        references = request_data.get("referenceDocuments", [])
        
        # Preprocess attributes
        for attr in attributes:
            for field in ["currentScore", "potentialScore", "targetScore", "gap"]:
                if isinstance(attr[field], str) and '%' in attr[field]:
                    attr[field] = float(attr[field].strip('%'))
        
        # Sort attributes by gap (descending)
        attributes_sorted = sorted(
            attributes, 
            key=lambda x: x["gap"], 
            reverse=True
        )
        
        # Select top 3-5 attributes based on gap size
        focus_attributes = attributes_sorted[:min(5, len(attributes))]
        
        # Build attributes prompt
        attributes_prompt = "\n".join([
            f"- {attr['name']}: Current {attr['currentScore']}% → Target {attr['targetScore']}% "
            f"(Gap: {attr['gap']}%, Weight: {attr['weight']})"
            for attr in focus_attributes
        ])
        
        # Build reference context
        reference_context = "\n\n".join([
            f"REFERENCE: {doc['documentName']}\n{doc['markdown'][:2000]}"
            for doc in references
        ])
        
        # Create study plan prompt
        prompt = f"""
        ROLE: Expert educational planner for {course_name}
        TASK: Create personalized development plan for {first_name} {last_name}
        
        STUDENT PROFILE:
        - Current Overall: {current_score}%
        - Target Overall: {target_score}%
        - Performance Level: {performance}
        - Engagement Level: {engagement}
        
        FOCUS AREAS (Prioritized by Improvement Need):
        {attributes_prompt}
        
        {"SPECIAL INSTRUCTIONS: " + context if context else ""}
        
        PLAN REQUIREMENTS:
        1. Create a precise, relevant plan name reflecting the primary improvement area
        2. Write a concise description of the plan's objectives
        3. Focus on improving the top 3-5 attributes with the highest gaps
        4. Include 5-7 actionable steps with resource links
        5. Vary activity types (videos, readings, assignments, quizzes)
        6. Estimate realistic time commitment (ETA in weeks)
        7. Set achievable potential overall score
        8. Provide 2-3 milestones
        9. Recommend 3-5 key resources
        10. For each focus attribute:
            - Create 2-3 specific subskills
            - Set target scores for subskills
            - Assign colors: red(urgent), yellow(important), blue(valuable), green(achieved)
        
        REFERENCE MATERIALS:
        {reference_context}
        
        RESPONSE FORMAT (JSON ONLY):
        {{
            "name": "Plan name (e.g., 'Data Structures Mastery')",
            "description": "Brief plan description",
            "progress": 0,
            "potentialOverall": {min(100, target_score + 5)},
            "eta": 8,  // Estimated weeks to complete
            "performance": "Current performance status",
            "skills": [
                {{
                    "name": "Attribute name",
                    "score": current_score,
                    "subskills": [
                        {{
                            "name": "Specific subskill",
                            "score": target_score,
                            "color": "red|yellow|blue|green"
                        }}
                    ]
                }}
            ],
            "courseId": "{course_id}",
            "steps": [
                {{
                    "title": "Actionable step title",
                    "type": "video|document|assignment|quiz",
                    "link": "https://resource-link.com",
                    "order": 1
                }}
            ],
            "milestones": [
                "Milestone 1 description",
                "Milestone 2 description"
            ],
            "resources": [
                "Resource 1",
                "Resource 2"
            ]
        }}
        """
        
        # Get plan from Gemini
        response = gemini_model.generate_content(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if not json_match:
            raise ValueError("No valid JSON found in response")
        
        plan = json.loads(json_match.group())
        
        # Validate response structure
        required_keys = ["name", "description", "steps", "skills", "eta"]
        if not all(key in plan for key in required_keys):
            raise ValueError("Plan missing required fields")
            
        return JSONResponse(content=plan)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Plan generation failed: {str(e)}"
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)