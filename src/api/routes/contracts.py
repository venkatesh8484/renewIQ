"""
POST /contracts/upload — trigger PPA contract ingestion pipeline
GET  /contracts        — list all loaded contracts
GET  /contracts/{id}   — get contract metadata

Phase 1: stubbed with correct response shapes.
Phase 3: wires in real ingestion pipeline.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory store for Phase 1 — replaced by Delta table lookup in Phase 3
_contracts: dict[str, dict] = {}


class ContractMetadata(BaseModel):
    contract_id: str
    filename: str
    ppa_type: str                   # "physical" | "virtual" | "sleeved"
    counterparty: Optional[str] = None
    strike_price_eur: Optional[float] = None
    volume_mw: Optional[float] = None
    tenor_years: Optional[int] = None
    delivery_point: Optional[str] = None
    status: str = "ingested"        # "uploading" | "ingesting" | "ingested" | "failed"
    chunk_count: Optional[int] = None


class UploadResponse(BaseModel):
    contract_id: str
    status: str
    message: str


@router.post("/upload", response_model=UploadResponse)
async def upload_contract(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a PPA PDF contract and trigger the ingestion pipeline.

    The pipeline: PDF → PyMuPDF chunker → risk tagger → Silver Delta table
    → Vector Search index sync (auto, via Delta Sync index).
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Phase 1 stub — log the upload, return accepted status
    # Phase 3: write to ADLS Blob, trigger Lakeflow pipeline job
    contract_id = file.filename.replace(".pdf", "").lower().replace(" ", "-")
    logger.info(f"Contract upload received: {file.filename} → contract_id={contract_id}")

    _contracts[contract_id] = {
        "contract_id": contract_id,
        "filename": file.filename,
        "ppa_type": "unknown",
        "status": "ingesting",
    }

    return UploadResponse(
        contract_id=contract_id,
        status="accepted",
        message=(
            f"Contract '{file.filename}' accepted. "
            "Ingestion pipeline triggered (Phase 1 stub — real pipeline in Phase 3)."
        ),
    )


@router.get("", response_model=list[ContractMetadata])
def list_contracts() -> list[ContractMetadata]:
    """List all contracts currently loaded in the system."""
    return [ContractMetadata(**c) for c in _contracts.values()]


@router.get("/{contract_id}", response_model=ContractMetadata)
def get_contract(contract_id: str) -> ContractMetadata:
    """Get metadata for a specific contract."""
    if contract_id not in _contracts:
        raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found")
    return ContractMetadata(**_contracts[contract_id])
