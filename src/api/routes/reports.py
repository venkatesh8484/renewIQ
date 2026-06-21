"""
GET /reports/{report_id}      — get report metadata + Markdown content
GET /reports/{report_id}/pdf  — download report as PDF

Phase 1: stubbed. Phase 5: serves real WeasyPrint-generated PDFs from Delta table.
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory store for Phase 1
_reports: dict[str, dict] = {}


class ReportSummary(BaseModel):
    report_id: str
    contract_ids: list[str]
    generated_at: datetime
    risk_flags_count: int
    highest_severity: str
    total_exposure_eur: float


class ReportDetail(ReportSummary):
    markdown_content: str


@router.get("", response_model=list[ReportSummary])
def list_reports() -> list[ReportSummary]:
    """List all generated risk reports."""
    return [ReportSummary(**r) for r in _reports.values()]


@router.get("/{report_id}", response_model=ReportDetail)
def get_report(report_id: str) -> ReportDetail:
    """Get full report content as Markdown."""
    if report_id not in _reports:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")
    return ReportDetail(**_reports[report_id])


@router.get("/{report_id}/pdf")
def download_report_pdf(report_id: str) -> Response:
    """
    Download report as a styled PDF.
    Phase 1: returns a placeholder PDF stub.
    Phase 5: serves real WeasyPrint-rendered PDFs.
    """
    if report_id not in _reports:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")

    # Phase 1 stub — return plaintext as application/pdf placeholder
    content = f"[Phase 1 stub] PDF for report {report_id} — WeasyPrint rendering wired in Phase 5."
    return Response(
        content=content.encode(),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={report_id}.pdf"},
    )
