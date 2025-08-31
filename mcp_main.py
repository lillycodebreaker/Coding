"""
FastAPI application and MCP Server for running the Fee Schedule Build "Agent" Task Automation.
This is a *deterministic* implementation, without any AI/ML components.
"""

# ~ Imports

import logging
from datetime import datetime

import opera_common_utils.logger
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastmcp import FastMCP
from starlette.responses import JSONResponse
from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Optional


from app import schemas, utils
from app.configs import config
from app.middleware.logging_middleware import LoggingMiddleware

import google.auth
from google.auth.transport.requests import AuthorizedSession
import requests
from requests.adapters import HTTPAdapter
from google.cloud import bigquery
import asyncio

# ~ Setup
#logging.basicConfig(
#    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
#)
#logger = logging.getLogger(__name__)

# ~ Placeholder Config
class Config:
    class Settings:
        SERVICE_NAME = "FeeScheduleBuildService"
        API_TITLE = "Fee Schedule MCP API"
        API_DESCRIPTION = "API for Fee Schedule Build MCP Agent"
        API_VERSION = "1.0.0"
    settings = Settings()

config = Config()

# ~ Placeholder Middleware
#class LoggingMiddleware:
#    def __init__(self, app):
#        self.app = app

# ~ Placeholder Database Connector
#class AsyncDatabaseConnector:
#    def __init__(self):
#        self.table = "project.dataset.table"
#        self.client = None  # Replace with actual BigQuery client

# Global database connector instance
#db_connector = AsyncDatabaseConnector()
#specialty_data = SpecialtyData(db_connector=db_connector)

mcp: FastMCP = FastMCP(
    config.settings.SERVICE_NAME,
    stateless_http=True
)
mcp_app = mcp.http_app(transport="streamable-http", path="/mcp")

app = FastAPI(
    title=config.settings.API_TITLE,
    description=config.settings.API_DESCRIPTION,
    version=config.settings.API_VERSION,
    lifespan=mcp_app.router.lifespan_context,
)

app.mount("/FeeScheduleBuild", mcp_app)

# ~ Middleware
#app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-correlation-id"],
)

# ~ API Routes
@app.get("/")
@app.get("/favicon.ico", include_in_schema=False)
async def hello_world(request: Request):
    return JSONResponse(
        content={"message": f"<p>Welcome to the {config.settings.API_TITLE}!</p>"},
        status_code=200,
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

@app.get("/health")
def health_check():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "status": "healthy",
        "timestamp": current_time,
        "version": config.settings.API_VERSION,
        "endpoints": [
            {"endpoint": "/api/FeeScheduleBuild", "status": "operational"},
            {"endpoint": "/health", "status": "operational"},
            {"endpoint": "/FeeScheduleBuild/sse", "status": "operational"},
            {"endpoint": "/FeeScheduleBuild/messages", "status": "operational"},
        ],
    }

@app.get("/api/FeeScheduleBuild/health")
async def healthcheck_endpoint():
    return Response(status_code=200, content="ok")

@app.post("/api/FeeScheduleBuild")
async def FeeScheduleBuildApi(feeSchedule: str, buildDate: str = "", sessionId: str = ""):
    # Placeholder response
    return JSONResponse(content={"feeSchedule": feeSchedule, "buildDate": buildDate, "Success": True, "sessionId": sessionId}, status_code=200)
    
# ~ MCP Server
@mcp.tool()
async def FeeScheduleBuild(feeSchedule: str="", buildDate: str = "", sessionId: str = "") -> dict:
    return {"feeSchedule": feeSchedule, "buildDate": buildDate}

