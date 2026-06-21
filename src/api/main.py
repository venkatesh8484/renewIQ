import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import settings
from src.api.routes import chat, contracts, reports

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RenewIQ API starting up")
    logger.info(f"LLM backend: {settings.llm_backend}")
    logger.info(f"Mock mode: {settings.use_mock_endpoint}")
    yield
    logger.info("RenewIQ API shutting down")


app = FastAPI(
    title="RenewIQ API",
    description=(
        "Renewable Energy PPA Intelligence Copilot — "
        "multi-agent LLM system for PPA contract risk analysis"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(contracts.router, prefix="/contracts", tags=["contracts"])
app.include_router(reports.router, prefix="/reports", tags=["reports"])


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness check — used by Docker healthcheck and Kubernetes probes."""
    return {
        "status": "ok",
        "llm_backend": settings.llm_backend,
        "mock_mode": settings.use_mock_endpoint,
    }


@app.get("/", tags=["ops"])
def root() -> dict:
    return {"message": "RenewIQ API — see /docs for Swagger UI"}
