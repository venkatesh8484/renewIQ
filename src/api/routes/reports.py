"""
GET /reports            — list all generated risk reports
GET /reports/{id}       — get report metadata + narrative
GET /reports/{id}/pdf   — download report as PDF (WeasyPrint)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from src.api.routes import _report_store

router = APIRouter()
logger = logging.getLogger(__name__)


class RiskFlagSummary(BaseModel):
    flag_id: str
    contract_id: str
    clause_id: str
    risk_category: str
    severity: str
    description: str
    exposure_eur: Optional[float] = None


class ReportSummary(BaseModel):
    report_id: str
    query: str
    contract_ids: list[str]
    risk_flags_count: int
    highest_severity: Optional[str]
    total_exposure_eur: Optional[float]
    negative_hours_90d: int


class ReportDetail(ReportSummary):
    risk_flags: list[RiskFlagSummary]
    narrative: str
    sources: list[str]


@router.get("", response_model=list[ReportSummary])
def list_reports() -> list[ReportSummary]:
    """List all generated risk reports (most recent first)."""
    return [_to_summary(r) for r in _report_store.values()]


@router.get("/{report_id}", response_model=ReportDetail)
def get_report(report_id: str) -> ReportDetail:
    """Get full report including narrative and all risk flags."""
    report = _report_store.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")
    return _to_detail(report)


@router.get("/{report_id}/pdf")
def download_report_pdf(report_id: str) -> Response:
    """
    Download report as a styled PDF.
    Renders the narrative + risk flags table via WeasyPrint.
    Falls back to a minimal HTML→bytes conversion when WeasyPrint is unavailable.
    """
    report = _report_store.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")

    html = _render_html(report)

    try:
        from weasyprint import HTML
        pdf_bytes = HTML(string=html).write_pdf()
        logger.info(f"[/reports] WeasyPrint PDF generated for {report_id}")
    except ImportError:
        # WeasyPrint not installed in this environment — return HTML as fallback
        logger.warning("[/reports] WeasyPrint not available, returning HTML")
        return Response(
            content=html.encode("utf-8"),
            media_type="text/html",
            headers={"Content-Disposition": f"inline; filename={report_id}.html"},
        )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={report_id}.pdf"},
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _severity_order(s: Optional[str]) -> int:
    return {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}.get(s or "", 9)


def _highest_severity(flags: list[dict]) -> Optional[str]:
    if not flags:
        return None
    return min(flags, key=lambda f: _severity_order(f.get("severity")))["severity"]


def _to_summary(r: dict) -> ReportSummary:
    flags = r.get("risk_flags", [])
    return ReportSummary(
        report_id=r.get("session_id", "unknown"),
        query=r.get("query", ""),
        contract_ids=r.get("contract_ids", []),
        risk_flags_count=len(flags),
        highest_severity=_highest_severity(flags),
        total_exposure_eur=r.get("total_exposure_eur"),
        negative_hours_90d=r.get("negative_hours_90d", 0),
    )


def _to_detail(r: dict) -> ReportDetail:
    flags = r.get("risk_flags", [])
    return ReportDetail(
        report_id=r.get("session_id", "unknown"),
        query=r.get("query", ""),
        contract_ids=r.get("contract_ids", []),
        risk_flags_count=len(flags),
        highest_severity=_highest_severity(flags),
        total_exposure_eur=r.get("total_exposure_eur"),
        negative_hours_90d=r.get("negative_hours_90d", 0),
        risk_flags=[RiskFlagSummary(**f) for f in flags],
        narrative=r.get("narrative", ""),
        sources=r.get("sources", []),
    )


def _render_html(report: dict) -> str:
    flags = report.get("risk_flags", [])
    flag_rows = ""
    for f in flags:
        sev = f.get("severity", "")
        colour = {"HIGH": "#c0392b", "MEDIUM": "#e67e22", "LOW": "#27ae60"}.get(sev, "#555")
        exp = f"€{f['exposure_eur']:,.0f}" if f.get("exposure_eur") else "—"
        flag_rows += (
            f"<tr>"
            f"<td style='color:{colour};font-weight:bold'>{sev}</td>"
            f"<td>{f.get('contract_id','')}</td>"
            f"<td>{f.get('clause_id','')}</td>"
            f"<td>{f.get('risk_category','')}</td>"
            f"<td>{exp}</td>"
            f"<td>{f.get('description','')}</td>"
            f"</tr>"
        )

    narrative_html = "<br>".join(report.get("narrative", "").split("\n"))
    total = report.get("total_exposure_eur")
    total_str = f"€{total:,.0f}" if total else "—"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RenewIQ Risk Report — {report.get('session_id','')}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
  h1 {{ color: #1a5276; }} h2 {{ color: #2c3e50; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }}
  th {{ background: #2c3e50; color: white; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .summary {{ background: #eaf4fb; padding: 12px; border-radius: 4px; margin-bottom: 16px; }}
</style>
</head>
<body>
<h1>RenewIQ PPA Risk Report</h1>
<div class="summary">
  <strong>Query:</strong> {report.get('query','')}<br>
  <strong>Contracts:</strong> {', '.join(report.get('contract_ids', []))}<br>
  <strong>Total EUR Exposure:</strong> {total_str}<br>
  <strong>Negative Price Hours (90d):</strong> {report.get('negative_hours_90d', 0)}
</div>

<h2>Risk Flags</h2>
<table>
  <tr><th>Severity</th><th>Contract</th><th>Clause</th><th>Category</th><th>Exposure</th><th>Description</th></tr>
  {flag_rows if flag_rows else '<tr><td colspan="6">No risk flags identified.</td></tr>'}
</table>

<h2>Narrative</h2>
<p>{narrative_html}</p>

<h2>Source Clauses</h2>
<p>{', '.join(report.get('sources', [])) or '—'}</p>

<p style="color:#aaa;font-size:11px;margin-top:40px">
  Generated by RenewIQ · Session {report.get('session_id','')}
</p>
</body>
</html>"""
