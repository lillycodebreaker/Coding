# services/document_processor.py
import PyPDF2
import re
import logging
from typing import Dict, List, Any
import chardet

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Service for processing and extracting text from documents"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.sql', '.txt'}
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
            
            # Clean and normalize text
            text = self._clean_text(text)
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise Exception(f"Failed to process PDF file: {str(e)}")
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text content from text/SQL files"""
        try:
            # Detect file encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding_result = chardet.detect(raw_data)
                encoding = encoding_result['encoding'] or 'utf-8'
            
            # Read file with detected encoding
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
            
            # Clean and normalize text
            text = self._clean_text(text)
            logger.info(f"Successfully extracted {len(text)} characters from file")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from file: {e}")
            raise Exception(f"Failed to process file: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\-.,;:()\[\]{}\'\"=<>!@#$%^&*+/\\|`~]', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_sections(self, text: str, document_type: str) -> Dict[str, List[str]]:
        """Extract specific sections from document based on type"""
        sections = {}
        
        if document_type.lower() == 'brd':
            sections = self._extract_brd_sections(text)
        elif document_type.lower() in ['sql', 'sp']:
            sections = self._extract_sql_sections(text)
        
        return sections
    
    def _extract_brd_sections(self, text: str) -> Dict[str, List[str]]:
        """Extract sections specific to BRD documents"""
        sections = {
            'business_rules': [],
            'requirements': [],
            'validations': [],
            'conditions': [],
            'processes': []
        }
        
        # Common BRD section patterns
        patterns = {
            'business_rules': [
                r'business\s+rules?.*?(?=\n\n|\n[A-Z]|\Z)',
                r'rules?\s+and\s+regulations.*?(?=\n\n|\n[A-Z]|\Z)',
                r'rule\s*:\s*.*?(?=\n\n|\n[A-Z]|\Z)'
            ],
            'requirements': [
                r'requirements?.*?(?=\n\n|\n[A-Z]|\Z)',
                r'functional\s+requirements?.*?(?=\n\n|\n[A-Z]|\Z)',
                r'req\s*\d+.*?(?=\n\n|\n[A-Z]|\Z)'
            ],
            'validations': [
                r'validation.*?(?=\n\n|\n[A-Z]|\Z)',
                r'verify.*?(?=\n\n|\n[A-Z]|\Z)',
                r'check.*?(?=\n\n|\n[A-Z]|\Z)'
            ],
            'conditions': [
                r'if\s+.*?then.*?(?=\n\n|\n[A-Z]|\Z)',
                r'when\s+.*?(?=\n\n|\n[A-Z]|\Z)',
                r'condition.*?(?=\n\n|\n[A-Z]|\Z)'
            ]
        }
        
        for section, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                sections[section].extend(matches)
        
        return sections
    
    def _extract_sql_sections(self, text: str) -> Dict[str, List[str]]:
        """Extract sections specific to SQL/stored procedures"""
        sections = {
            'procedures': [],
            'functions': [],
            'conditions': [],
            'validations': [],
            'business_logic': []
        }
        
        # SQL-specific patterns
        patterns = {
            'procedures': [
                r'CREATE\s+PROCEDURE.*?(?=CREATE|ALTER|\Z)',
                r'ALTER\s+PROCEDURE.*?(?=CREATE|ALTER|\Z)'
            ],
            'functions': [
                r'CREATE\s+FUNCTION.*?(?=CREATE|ALTER|\Z)',
                r'ALTER\s+FUNCTION.*?(?=CREATE|ALTER|\Z)'
            ],
            'conditions': [
                r'IF\s+.*?(?=END\s+IF|ELSE|\Z)',
                r'CASE\s+WHEN.*?(?=END\s+CASE|\Z)',
                r'WHERE\s+.*?(?=ORDER|GROUP|HAVING|\Z)'
            ],
            'validations': [
                r'CHECK\s*\(.*?\)',
                r'CONSTRAINT.*?(?=,|\)|\Z)',
                r'VALIDATE.*?(?=;|\Z)'
            ]
        }
        
        for section, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                sections[section].extend(matches)
        
        return sections
    
    def preprocess_for_rag(self, text: str, document_type: str) -> Dict[str, Any]:
        """Preprocess document for RAG processing"""
        # Extract sections
        sections = self.extract_sections(text, document_type)
        
        # Create chunks for better RAG processing
        chunks = self._create_text_chunks(text)
        
        # Extract keywords and entities
        keywords = self._extract_keywords(text)
        
        return {
            'full_text': text,
            'sections': sections,
            'chunks': chunks,
            'keywords': keywords,
            'document_type': document_type,
            'metadata': {
                'character_count': len(text),
                'section_count': len(sections),
                'chunk_count': len(chunks)
            }
        }
    
    def _create_text_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Create overlapping text chunks for RAG processing"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Common business rule keywords
        business_keywords = [
            'rule', 'requirement', 'validation', 'condition', 'process',
            'business', 'logic', 'constraint', 'mandatory', 'optional',
            'must', 'should', 'shall', 'if', 'then', 'when', 'where'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in business_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        # Extract domain