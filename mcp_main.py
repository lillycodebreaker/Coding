"""
FastAPI application and MCP Server for running the Fee Schedule Build "Agent" Task Automation.
This is a *deterministic* implementation, without any AI/ML components.
"""

# ~ Imports

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastmcp import FastMCP
from starlette.responses import JSONResponse


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

app.mount("/TestFunction", mcp_app)

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
            {"endpoint": "/api/TestFunction", "status": "operational"},
            {"endpoint": "/health", "status": "operational"},
            {"endpoint": "/TestFunction/sse", "status": "operational"},
            {"endpoint": "/TestFunction/messages", "status": "operational"},
        ],
    }

@app.get("/api/TestFunction/health")
async def healthcheck_endpoint():
    return Response(status_code=200, content="ok")

@app.post("/api/TestFunction")
async def TestFunctionApi(input: str="", sessionId: str = ""):
    # Placeholder response
    return JSONResponse(content={"input": input, "Success": True, "sessionId": sessionId}, status_code=200)
    
# ~ MCP Server
@mcp.tool()
async def TestFunction(input: str="", sessionId: str = "") -> dict:
    return {"input": input}


