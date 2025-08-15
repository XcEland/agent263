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
    uvicorn.run("ocr_api:app", host="0.0.0.0", port=8000, reload=True)