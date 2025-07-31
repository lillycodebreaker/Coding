# .env
# Google Cloud Configuration
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
BQ_DATASET_ID=rag_document_processing

# OpenAI Configuration (optional, for enhanced RAG)
OPENAI_API_KEY=your-openai-api-key

# FastAPI Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# File Upload Configuration
MAX_FILE_SIZE=50MB
ALLOWED_EXTENSIONS=[".pdf", ".sql", ".txt"]
UPLOAD_TEMP_DIR=./temp_uploads

# RAG Configuration
CONFIDENCE_THRESHOLD=0.7
MAX_RULES_PER_DOCUMENT=1000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# BigQuery Configuration
BQ_LOCATION=US
BQ_JOB_TIMEOUT=300