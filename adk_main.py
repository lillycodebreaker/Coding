import logging
import os
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

logging.basicConfig(level=logging.INFO)

# Get the directory where main.py is located
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_IP = os.getenv("DB_IP")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = "3306"
# Example allowed origins for CORS
ALLOWED_ORIGINS = ["http://localhost", "http://localhost:8080", "*"]
# Set web=True if you intend to serve a web interface, False otherwise
SERVE_WEB_INTERFACE = True


def get_google_cloud_sql_db_url():
    username = DB_USER
    password = DB_PASSWORD
    db_ip = DB_IP  # This will come from Kubernetes
    db_port = DB_PORT  # Fixed the typo from your original code (was DP_PORT)
    database_name = DB_NAME
    # This is the simple, direct URL format that the adk library will understand.
    return f"mysql+pymysql://{username}:{password}@{db_ip}:{db_port}/{database_name}"


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri="sqlite:///./run_task_agent_test_sessions.db",#get_google_cloud_sql_db_url(),
    #session_service_kwargs={
    #    "pool_pre_ping": True,
    #    "pool_recycle": 3600,  # Recycle connections every hour
    #},
    allow_origins=ALLOWED_ORIGINS,
    web=SERVE_WEB_INTERFACE,
)

# Remove the "/" route if it exists
app.router.routes = [
    r
    for r in app.router.routes
    if not (getattr(r, "path", None) == "/" and getattr(r, "methods", set()) == {"GET"})
]


# You can add more FastAPI routes or configurations below if needed
# Example:
@app.get("/health")
async def health_check():
    return {
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "endpoint": "/health",
        "status": "operational",
    }


@app.get("/")
async def read_root():
    return {
        "api_name": "Run Task Agent API",
        "version": "1.0.0",
        "available_endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Root endpoint - API information",
            },
            {"path": "/run", "method": "POST", "description": "Run a task for a user"},
            {
                "path": "/apps/opera_agent/users/{{user_id}}/sessions/{{session_id}}",
                "method": "POST",
                "description": "Create a new session for a user",
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint",
            },
            {"path": "/docs", "method": "GET", "description": "Documentation endpoint"},
        ],
    }


if __name__ == "__main__":
    # Use the PORT environment variable provided by Cloud Run, defaulting to 8080
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
