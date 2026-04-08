"""
FastAPI server for the OpenEnv environment.
Exposes /health endpoint and runs on port 7860.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="OpenEnv: Advanced Data Engineering Pipeline")


@app.get("/health")
async def health():
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "openenv-data-engineering-pipeline",
            "version": "2.0.0",
        },
    )


@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={
            "message": "OpenEnv: Advanced Multi-Table Data Engineering Pipeline",
            "docs": "/docs",
            "health": "/health",
        },
    )
