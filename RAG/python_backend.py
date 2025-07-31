# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from typing import Dict, Any
import logging
from datetime import datetime

from services.document_processor import DocumentProcessor
from services.rule_extractor import RuleExtractor
from services.bigquery_service import BigQueryService
from models.schemas import ExtractedRules, ProcessingResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document Processing API",
    description="API for extracting business rules from BRD documents and stored procedures",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor()
rule_extractor = RuleExtractor()
bigquery_service = BigQueryService()

@app.on_startup
async def startup_event():
    """Initialize database connection and create tables if needed"""
    try:
        await bigquery_service.initialize_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

@app.post("/api/extract-rules", response_model=Dict[str, Any])
async def extract_rules(
    brd_document: UploadFile = File(..., description="BRD PDF document"),
    stored_procedure: UploadFile = File(..., description="Stored procedure file")
):
    """Extract business rules from uploaded BRD document and stored procedure"""
    
    # Validate file types
    if not brd_document.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="BRD document must be a PDF file")
    
    if not stored_procedure.filename.lower().endswith(('.sql', '.txt')):
        raise HTTPException(status_code=400, detail="Stored procedure must be a SQL or TXT file")
    
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as brd_temp:
            brd_content = await brd_document.read()
            brd_temp.write(brd_content)
            brd_temp_path = brd_temp.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.sql') as sp_temp:
            sp_content = await stored_procedure.read()
            sp_temp.write(sp_content)
            sp_temp_path = sp_temp.name
        
        # Process documents
        logger.info("Processing BRD document...")
        brd_text = document_processor.extract_text_from_pdf(brd_temp_path)
        
        logger.info("Processing stored procedure...")
        sp_text = document_processor.extract_text_from_file(sp_temp_path)
        
        # Extract rules using RAG
        logger.info("Extracting rules using RAG...")
        extracted_rules = await rule_extractor.extract_rules(brd_text, sp_text)
        
        # Store in BigQuery
        processing_result = ProcessingResult(
            id=f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            brd_filename=brd_document.filename,
            sp_filename=stored_procedure.filename,
            extracted_rules=extracted_rules,
            processing_timestamp=datetime.now(),
            status="completed"
        )
        
        await bigquery_service.store_processing_result(processing_result)
        
        # Cleanup temporary files
        os.unlink(brd_temp_path)
        os.unlink(sp_temp_path)
        
        return extracted_rules.dict()
        
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        # Cleanup temporary files on error
        try:
            os.unlink(brd_temp_path)
            os.unlink(sp_temp_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/processing-history")
async def get_processing_history():
    """Get processing history from BigQuery"""
    try:
        history = await bigquery_service.get_processing_history()
        return {"history": history}
    except Exception as e:
        logger.error(f"Error fetching processing history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "services": {
            "document_processor": "ok",
            "rule_extractor": "ok",
            "bigquery": "ok"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)