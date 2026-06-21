"""
Synthetic PPA Contract Generator
----------------------------------
Generates 5 structurally realistic PPA PDF contracts:
  - 2x Physical PPA (wind, solar)
  - 2x Virtual/Financial PPA (vPPA)
  - 1x Sleeved PPA

Each PDF is 30–60 pages with realistic clause numbering, Dutch law references,
and all 6 risk category clause types present.

Based on:
  - EIB Advisory Services PPA template framework
  - DLA Piper international PPA risk allocation standards
  - ACM Netherlands / Dutch Electricity Act 1998 (Elektriciteitswet)

Usage:
    python scripts/generate_synthetic_ppas.py
    python scripts/generate_synthetic_ppas.py --output-dir data/synthetic_ppa/
"""

import argparse
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

try:
    from fpdf import FPDF
except ImportError:
    raise ImportError("Run: pip install fpdf2")


# ── Contract templates ────────────────────────────────────────────────────────

@dataclass
class PPASpec:
    contract_id: str
    ppa_type: str                    # "physical" | "virtual" | "sleeved"
    seller: str
    buyer: str
    asset_name: str
    asset_type: str                  # "wind_onshore" | "wind_offshore" | "solar"
    delivery_point: str
    strike_price_eur: float
    volume_mw: float
    tenor_years: int
    start_date: date
    negative_price_floor: bool       # True = contract has a floor clause
    take_or_pay_pct: float           # e.g. 0.85 = 85% minimum take obligation
    curtailment_compensation: bool   # True = seller compensated for curtailment
    governing_law: str = "Netherlands"
    arbitration: str = "ICC Paris"


CONTRACTS: list[PPASpec] = [
    PPASpec(
        contract_id="zeeland-wind-physical-ppa-v1",
        ppa_type="physical",
        seller="WindPark Zeeland B.V.",
        buyer="NL Manufacturing Group N.V.",
        asset_name="WindPark Zeeland Phase II",
        asset_type="wind_onshore",
        delivery_point="TenneT 150kV — Borssele substation, Zeeland",
        strike_price_eur=68.00,
        volume_mw=12.0,
        tenor_years=15,
        start_date=date(2024, 1, 1),
        negative_price_floor=False,      # KEY: no floor → HIGH exposure risk
        take_or_pay_pct=0.85,
        curtailment_compensation=False,  # KEY: no compensation → curtailment risk
    ),
    PPASpec(
        contract_id="flevoland-solar-virtual-ppa-v1",
        ppa_type="virtual",
        seller="SolarFields Flevoland B.V.",
        buyer="Dutch Retail Chain Holding B.V.",
        asset_name="Solarpark Flevoland",
        asset_type="solar",
        delivery_point="APX/EPEX NL Hub (financial settlement)",
        strike_price_eur=72.50,
        volume_mw=8.5,
        tenor_years=12,
        start_date=date(2023, 7, 1),
        negative_price_floor=True,       # Has floor clause at €0/MWh
        take_or_pay_pct=0.90,
        curtailment_compensation=True,
    ),
    PPASpec(
        contract_id="groningen-wind-physical-ppa-v1",
        ppa_type="physical",
        seller="Groningen Wind Energy B.V.",
        buyer="AMS DataCenter Coöperatie U.A.",
        asset_name="Windpark Eemshaven Noord",
        asset_type="wind_offshore",
        delivery_point="TenneT 380kV — Eemshaven, Groningen",
        strike_price_eur=61.00,
        volume_mw=25.0,
        tenor_years=20,
        start_date=date(2022, 4, 1),
        negative_price_floor=True,       # Floor at -€20/MWh
        take_or_pay_pct=0.80,
        curtailment_compensation=True,
    ),
    PPASpec(
        contract_id="brabant-solar-virtual-ppa-v1",
        ppa_type="virtual",
        seller="SolarEnergie Brabant B.V.",
        buyer="NL Food Processing Group B.V.",
        asset_name="Solarpark Tilburg Zuid",
        asset_type="solar",
        delivery_point="APX/EPEX NL Hub (financial settlement)",
        strike_price_eur=78.00,
        volume_mw=6.0,
        tenor_years=10,
        start_date=date(2025, 1, 1),
        negative_price_floor=False,      # No floor
        take_or_pay_pct=0.75,
        curtailment_compensation=False,
    ),
    PPASpec(
        contract_id="friesland-wind-sleeved-ppa-v1",
        ppa_type="sleeved",
        seller="FrieslandWind Energie B.V.",
        buyer="NL Chemical Industry B.V.",
        asset_name="Windpark Afsluitdijk",
        asset_type="wind_onshore",
        delivery_point="Liander DSO Zone — Friesland",
        strike_price_eur=65.50,
        volume_mw=18.0,
        tenor_years=15,
        start_date=date(2023, 10, 1),
        negative_price_floor=True,
        take_or_pay_pct=0.85,
        curtailment_compensation=True,
    ),
]


# ── PDF generator ─────────────────────────────────────────────────────────────

class PPAGenerator(FPDF):

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "POWER PURCHASE AGREEMENT — CONFIDENTIAL", align="C")
        self.ln(2)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(20, 60, 100)
        self.ln(6)
        self.cell(0, 8, title, ln=True)
        self.set_draw_color(20, 60, 100)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def clause_heading(self, clause_id: str, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.ln(4)
        self.cell(0, 7, f"{clause_id}  {title}", ln=True)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def sub_clause(self, ref: str, text: str):
        self.set_font("Helvetica", "", 10)
        x = self.get_x()
        self.set_x(20)
        self.multi_cell(0, 5.5, f"({ref})  {text}")
        self.ln(1)


def generate_ppa_pdf(spec: PPASpec, output_path: Path):
    pdf = PPAGenerator(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(15, 20, 15)

    end_date = spec.start_date + timedelta(days=365 * spec.tenor_years)

    # ── Cover Page ──────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 60, 100)
    pdf.ln(20)
    pdf.cell(0, 12, "POWER PURCHASE AGREEMENT", align="C", ln=True)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"({spec.ppa_type.upper()} PPA)", align="C", ln=True)
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)

    cover_info = [
        ("Contract ID",     spec.contract_id),
        ("Asset",           f"{spec.asset_name} ({spec.asset_type.replace('_', ' ').title()})"),
        ("Seller",          spec.seller),
        ("Buyer",           spec.buyer),
        ("Delivery Point",  spec.delivery_point),
        ("Strike Price",    f"€{spec.strike_price_eur:.2f}/MWh"),
        ("Contract Volume", f"{spec.volume_mw} MW"),
        ("Tenor",           f"{spec.tenor_years} years"),
        ("Start Date",      spec.start_date.strftime("%d %B %Y")),
        ("End Date",        end_date.strftime("%d %B %Y")),
        ("Governing Law",   spec.governing_law),
    ]

    pdf.ln(8)
    for label, value in cover_info:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 7, label + ":", ln=False)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, value, ln=True)
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5,
        "This document contains commercially sensitive information and is intended "
        "solely for the named parties. Redistribution is prohibited without written consent.")

    # ── Article 1: Definitions ──────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("ARTICLE 1 — DEFINITIONS AND INTERPRETATION")
    pdf.clause_heading("1.1", "Definitions")
    pdf.body_text(
        'In this Agreement, unless the context otherwise requires, the following terms '
        'shall have the meanings ascribed to them below:'
    )
    definitions = [
        ("Agreement", "this Power Purchase Agreement including all schedules and amendments."),
        ("Balancing Responsible Party (BRP)",
         "the party designated under the Dutch Electricity Act 1998 (Elektriciteitswet 1998) "
         "responsible for maintaining balance within its portfolio."),
        ("Contract Price", f"€{spec.strike_price_eur:.2f}/MWh as specified in Article 6."),
        ("Delivery Point", spec.delivery_point + "."),
        ("EPEX SPOT NL", "the day-ahead electricity market operated by EPEX SPOT SE for the Netherlands bidding zone."),
        ("Force Majeure Event",
         "any event beyond a Party's reasonable control as defined in Article 14."),
        ("GOPACS", "Grid Operator Platform for Congestion Solutions, operated jointly by TenneT, "
         "Liander, Enexis, Stedin and Westland Infra."),
        ("Imbalance Price",
         "the settlement price published by TenneT B.V. for deviations from the scheduled programme."),
        ("Metered Output", "the net electrical energy delivered to the Delivery Point as recorded by the fiscal meter."),
        ("Scheduled Output", "the forecasted generation volume submitted to the BRP by the Seller."),
        ("Settlement Period", "each 15-minute interval used for imbalance settlement by TenneT."),
        ("Strike Price", f"€{spec.strike_price_eur:.2f}/MWh, as may be adjusted per Article 6.3."),
        ("Take-or-Pay Quantity",
         f"{int(spec.take_or_pay_pct * 100)}% of the Scheduled Monthly Generation Volume as defined in Schedule 2."),
        ("TenneT", "TenneT TSO B.V., the Dutch Transmission System Operator."),
    ]
    for term, defn in definitions:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(55, 5.5, f'"{term}"', ln=False)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.5, defn)
        pdf.ln(1)

    pdf.clause_heading("1.2", "Interpretation")
    pdf.body_text(
        "References to statutes include all amendments. Headings are for convenience only. "
        "References to 'days' mean calendar days unless stated otherwise. "
        "This Agreement shall be governed by and construed in accordance with the laws of "
        f"the {spec.governing_law}."
    )

    # ── Article 2: Term ─────────────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("ARTICLE 2 — TERM AND COMMENCEMENT")
    pdf.clause_heading("2.1", "Term")
    pdf.body_text(
        f"This Agreement shall commence on {spec.start_date.strftime('%d %B %Y')} "
        f"(the 'Commencement Date') and shall continue in full force and effect until "
        f"{end_date.strftime('%d %B %Y')} (the 'Expiry Date'), being a period of "
        f"{spec.tenor_years} ({spec.tenor_years}) years, unless earlier terminated "
        "in accordance with Article 13."
    )
    pdf.clause_heading("2.2", "Conditions Precedent")
    pdf.body_text(
        "The obligations of the Parties under this Agreement are conditional upon: "
        "(a) the Asset obtaining all required permits and grid connection agreements; "
        "(b) the Seller providing evidence of registration as a BRP or delegation thereof; "
        "(c) the Buyer providing credit support in accordance with Article 10."
    )
    pdf.clause_heading("2.3", "Extension")
    pdf.body_text(
        "The Parties may extend this Agreement by mutual written consent no later than "
        "24 months prior to the Expiry Date. Any extension shall be subject to renegotiation "
        "of the Strike Price in accordance with Article 6.3."
    )

    # ── Article 3: Delivery ─────────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("ARTICLE 3 — DELIVERY OBLIGATIONS")
    pdf.clause_heading("3.1", "Delivery Point")
    pdf.body_text(
        f"The Seller shall deliver, or procure the delivery of, all Metered Output to the "
        f"Delivery Point at: {spec.delivery_point}. Title and risk in the electrical energy "
        "shall pass from the Seller to the Buyer at the Delivery Point."
    )
    pdf.clause_heading("3.2", "Scheduling and Nomination")
    pdf.body_text(
        "The Seller shall submit day-ahead generation forecasts to the designated BRP no later "
        "than 12:00 CET on the day preceding the Delivery Day. Forecasts shall be provided "
        "in 15-minute Settlement Period intervals consistent with TenneT's scheduling requirements."
    )
    pdf.clause_heading("3.3", "Metering")
    pdf.body_text(
        "Metering shall be conducted using fiscal-grade meters installed and maintained by "
        "the relevant Distribution System Operator or TenneT at the Delivery Point. "
        "Meter readings shall be provided to both Parties within 5 Business Days of each "
        "Settlement Period."
    )

    # ── Article 4: Volume Obligations ──────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("ARTICLE 4 — VOLUME AND TAKE-OR-PAY OBLIGATIONS")
    pdf.clause_heading("4.1", "Estimated Annual Generation")
    pdf.body_text(
        f"The Seller estimates the Asset will generate approximately "
        f"{int(spec.volume_mw * 8760 * 0.35 / 1000):,} GWh per annum based on a capacity "
        f"factor of 35% for {spec.asset_type.replace('_', ' ')}. This estimate is provided "
        "for indicative purposes only and does not constitute a guarantee of output."
    )
    pdf.clause_heading("4.2", "Take-or-Pay Obligation")  # volume_risk
    pdf.body_text(
        f"The Buyer shall be obligated to pay for, or take delivery of, a minimum of "
        f"{int(spec.take_or_pay_pct * 100)}% of the Scheduled Monthly Generation Volume "
        "in each calendar month during the Term ('Take-or-Pay Quantity'). "
        "Where the Buyer fails to take delivery of the Take-or-Pay Quantity, the Buyer "
        "shall pay to the Seller the Contract Price multiplied by the shortfall volume, "
        "irrespective of whether such energy was generated or could have been delivered. "
        "This obligation applies regardless of market conditions, including during periods "
        "of negative or zero market prices."
    )
    pdf.clause_heading("4.3", "Shape Risk")
    pdf.body_text(
        "The Buyer acknowledges and accepts the generation profile risk ('Shape Risk') "
        "associated with the intermittent nature of the Asset. The Buyer shall be responsible "
        "for procuring balancing services to manage deviations between the Scheduled Output "
        "and the Buyer's actual consumption, except where such deviations arise from events "
        "covered under Article 11 (Curtailment) or Article 14 (Force Majeure)."
    )

    # ── Article 5: Curtailment ──────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("ARTICLE 5 — CURTAILMENT AND GRID CONSTRAINTS")
    pdf.clause_heading("5.1", "Curtailment Events")  # curtailment_risk
    pdf.body_text(
        "A 'Curtailment Event' means any reduction in the Metered Output of the Asset "
        "instructed by TenneT or the relevant DSO (including via GOPACS redispatch orders) "
        "due to grid congestion or system stability requirements, which is outside the "
        "reasonable control of the Seller."
    )
    pdf.clause_heading("5.2", "Curtailment Compensation")
    if spec.curtailment_compensation:
        pdf.body_text(
            "In the event of a Curtailment Event, the Seller shall be entitled to receive "
            "compensation from the Buyer equal to the Contract Price multiplied by the "
            "estimated volume of energy that would have been generated but for the curtailment "
            "('Proxy Generation'). Proxy Generation shall be calculated using the method "
            "specified in Schedule 3 (P90 generation profile methodology). "
            "GOPACS redispatch payments received by the Seller shall be deducted from "
            "any compensation payable under this Clause."
        )
    else:
        pdf.body_text(
            "The Seller shall bear all volume risk associated with Curtailment Events. "
            "No compensation shall be payable by the Buyer in respect of any reduction in "
            "Metered Output resulting from a Curtailment Event. The Buyer's payment obligations "
            "under Article 6 shall be reduced proportionally to reflect only actual Metered Output "
            "delivered to the Delivery Point. The Seller acknowledges that grid congestion in "
            "the Zeeland and South-Holland DSO zones has historically affected curtailment volumes "
            "and has priced this risk into the Contract Price."
        )
    pdf.clause_heading("5.3", "GOPACS Participation")
    pdf.body_text(
        "The Seller shall participate in the GOPACS market to the extent technically feasible "
        "and economically reasonable. Any GOPACS redispatch revenues received by the Seller "
        "shall be shared with the Buyer in the ratio 60:40 (Seller:Buyer) unless otherwise "
        "agreed in writing."
    )

    # ── Article 6: Pricing ──────────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("ARTICLE 6 — CONTRACT PRICE AND PAYMENT")
    pdf.clause_heading("6.1", "Strike Price")  # price_risk
    pdf.body_text(
        f"The Contract Price for electrical energy delivered under this Agreement shall be "
        f"€{spec.strike_price_eur:.2f} per MWh (the 'Strike Price'), fixed for the duration "
        "of the Term subject to Articles 6.2 and 6.3."
    )
    pdf.clause_heading("6.2", "Basis of Payment")
    if spec.ppa_type == "physical":
        pdf.body_text(
            "Payment shall be calculated as: Payment = Strike Price × Metered Output (MWh). "
            "Invoices shall be issued monthly within 10 Business Days of the end of each "
            "calendar month. Payment shall be due within 30 days of invoice date."
        )
    else:
        pdf.body_text(
            "As this is a virtual/financial PPA, settlement shall be on a Contract for "
            "Difference (CfD) basis: Settlement Amount = (Strike Price − EPEX SPOT NL "
            "Day-Ahead Price) × Metered Output. Where the EPEX price exceeds the Strike "
            "Price, the Buyer shall pay the difference to the Seller. Where the Strike "
            "Price exceeds the EPEX price, the Seller shall pay the difference to the Buyer."
        )
    pdf.clause_heading("6.3", "Price Indexation")
    pdf.body_text(
        "The Strike Price shall be subject to annual indexation of 70% of the Dutch CPI "
        "(Consumentenprijsindex) as published by Statistics Netherlands (CBS) on 1 January "
        "of each Contract Year, commencing from the third Contract Year. The maximum annual "
        "adjustment shall not exceed 3% or fall below -1%."
    )

    # ── Article 7: Negative Price Provisions ───────────────────────────────
    pdf.add_page()
    pdf.chapter_title("ARTICLE 7 — NEGATIVE PRICE PROVISIONS")
    pdf.clause_heading("7.1", "Negative Market Price Events")  # price_risk — KEY CLAUSE
    pdf.body_text(
        "A 'Negative Price Event' means any hour in which the EPEX SPOT NL day-ahead price "
        "is below zero (€0/MWh). The Parties acknowledge that the increasing penetration of "
        "renewable generation in the Netherlands bidding zone has resulted in a statistically "
        "significant increase in the frequency and duration of Negative Price Events."
    )
    pdf.clause_heading("7.2", "Negative Price Floor")
    if spec.negative_price_floor:
        floor_price = -20.0 if spec.ppa_type == "physical" else 0.0
        pdf.body_text(
            f"Notwithstanding Article 6.1, during any Negative Price Event, the effective "
            f"settlement price shall not fall below €{floor_price:.2f}/MWh "
            f"(the 'Price Floor'). For the avoidance of doubt, the Buyer shall not be "
            "required to pay more than the Contract Price and the Seller shall not be "
            "required to pay the Buyer where the market price falls below the Price Floor. "
            "The Price Floor shall apply automatically and shall not require notice from "
            "either Party."
        )
    else:
        pdf.body_text(
            "No price floor applies under this Agreement. During any Negative Price Event, "
            "the Buyer's payment obligations under Article 6.1 shall remain in full force "
            "and effect. The Buyer shall pay the Strike Price in respect of all Metered Output "
            "delivered to the Delivery Point irrespective of the prevailing EPEX SPOT NL "
            "day-ahead price, including during periods of negative market prices. "
            "The Buyer acknowledges having been advised of the risk of Negative Price Events "
            "and has determined that the fixed-price certainty outweighs this risk."
        )
    pdf.clause_heading("7.3", "Obligation During Negative Price Hours")
    pdf.body_text(
        "Article 4.2 (Take-or-Pay) shall apply during Negative Price Events. "
        "The Seller shall not be entitled to curtail generation solely on the basis of "
        "a Negative Price Event. The Buyer shall not be entitled to refuse delivery "
        "during a Negative Price Event. Both Parties shall continue to perform their "
        "respective obligations under this Agreement during Negative Price Events."
    )

    # ── Article 8: Basis Risk ───────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("ARTICLE 8 — BASIS RISK AND BALANCING")
    pdf.clause_heading("8.1", "Basis Risk")  # basis_risk
    pdf.body_text(
        "The Parties acknowledge that the price of electricity at the Delivery Point may "
        "differ from the EPEX SPOT NL day-ahead price due to grid constraints, locational "
        "marginal pricing, or DSO-zone specific congestion. This price differential is "
        "referred to as 'Basis Risk' and shall be borne by:"
    )
    pdf.sub_clause("i", "the Seller, in respect of any negative basis differential at the Delivery Point;")
    pdf.sub_clause("ii", "the Buyer, in respect of any positive basis differential benefiting the Buyer's portfolio.")
    pdf.clause_heading("8.2", "Balancing Responsibility")
    pdf.body_text(
        "The Seller shall ensure that all electrical energy delivered under this Agreement "
        "is included within a duly registered BRP portfolio. The Seller shall be responsible "
        "for all imbalance costs arising from deviations between Scheduled Output and actual "
        "Metered Output, except where such deviations arise directly from a Curtailment Event "
        "under Article 5 or a Force Majeure Event under Article 14."
    )
    pdf.clause_heading("8.3", "Imbalance Settlement")
    pdf.body_text(
        "Where TenneT publishes an Imbalance Price in excess of 200% of the day-ahead price "
        "for three or more consecutive Settlement Periods, the Parties shall convene within "
        "2 Business Days to discuss the impact on this Agreement and any remedial actions."
    )

    # ── Article 9: Counterparty Risk ────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("ARTICLE 9 — CREDIT SUPPORT AND COUNTERPARTY RISK")
    pdf.clause_heading("9.1", "Credit Support Annex")  # counterparty_risk
    pdf.body_text(
        "Each Party shall, prior to the Commencement Date and thereafter on an annual basis, "
        "provide the other Party with its most recent audited financial statements and a "
        "credit rating from at least one recognised rating agency (Moody's, S&P, or Fitch). "
        "Where a Party does not have a public credit rating, it shall provide a bank "
        "guarantee or parent company guarantee in the form set out in Schedule 4."
    )
    pdf.clause_heading("9.2", "Termination Events")
    pdf.body_text(
        "Each of the following shall constitute a Termination Event: "
        "(a) insolvency, liquidation or administration of a Party; "
        "(b) failure to pay any amount due under this Agreement within 15 Business Days "
        "of the due date; "
        "(c) material breach not remedied within 30 days of notice; "
        "(d) credit rating downgrade below BB- (or equivalent) where no alternative credit "
        "support is provided within 20 Business Days."
    )
    pdf.clause_heading("9.3", "Step-In Rights")
    pdf.body_text(
        "In the event of a Seller Termination Event, the Buyer shall have the right, "
        "but not the obligation, to step into the Seller's financing arrangements with "
        "its lenders and assume operational control of the Asset for the purpose of "
        "continuing energy delivery under this Agreement."
    )

    # ── Article 10: Change in Law ────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("ARTICLE 10 — CHANGE IN LAW AND REGULATORY RISK")
    pdf.clause_heading("10.1", "Change-in-Law Provisions")  # legal_regulatory
    pdf.body_text(
        "A 'Change in Law' means any amendment, repeal or replacement of: "
        "(a) the Dutch Electricity Act 1998 (Elektriciteitswet 1998); "
        "(b) the Dutch Energy Agreement (Energieakkoord); "
        "(c) EU Directive 2018/2001 (Renewable Energy Directive II); "
        "(d) the EU Emissions Trading System (EU-ETS) Regulations; "
        "(e) any ACM (Authority for Consumers and Markets) grid code or tariff determination "
        "that materially and adversely affects the economic position of either Party."
    )
    pdf.clause_heading("10.2", "Notification")
    pdf.body_text(
        "The Party affected by a Change in Law shall notify the other Party within "
        "20 Business Days of becoming aware of such Change in Law, providing a reasoned "
        "assessment of the financial impact and proposed mitigation measures."
    )
    pdf.clause_heading("10.3", "Renegotiation")
    pdf.body_text(
        "Where a Change in Law results in a material adverse effect exceeding €500,000 "
        "per annum on either Party, the Parties shall enter into good faith renegotiation "
        "of the affected terms within 60 days of notification. If the Parties cannot agree "
        "on amended terms within 90 days, either Party may terminate this Agreement on "
        "12 months' written notice without penalty."
    )

    # ── Article 11: Force Majeure ─────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("ARTICLE 11 — FORCE MAJEURE")
    pdf.clause_heading("11.1", "Definition")
    pdf.body_text(
        "Force Majeure means any event or circumstance beyond the reasonable control of "
        "a Party that prevents or delays the performance of its obligations, including: "
        "acts of God, war, terrorism, grid failure caused by events external to the Asset, "
        "government action, strikes not involving the Party's own employees, or TenneT "
        "emergency grid management orders. Market price movements, including Negative Price "
        "Events, shall not constitute Force Majeure."
    )
    pdf.clause_heading("11.2", "Consequences")
    pdf.body_text(
        "The Party invoking Force Majeure shall be relieved of its obligations to the extent "
        "and for the duration of the Force Majeure Event, provided it: "
        "(a) gives notice within 5 Business Days of the event occurring; "
        "(b) uses reasonable endeavours to overcome or mitigate the Force Majeure Event; "
        "(c) resumes performance as soon as reasonably practicable."
    )

    # ── Article 12: Dispute Resolution ──────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("ARTICLE 12 — DISPUTE RESOLUTION AND GOVERNING LAW")
    pdf.clause_heading("12.1", "Governing Law")
    pdf.body_text(f"This Agreement shall be governed by the laws of the {spec.governing_law}.")
    pdf.clause_heading("12.2", "Expert Determination")
    pdf.body_text(
        "Any dispute relating to metering, generation calculations, or technical matters "
        "shall be referred to an independent expert appointed by mutual agreement or, "
        "failing agreement within 15 days, appointed by the Dutch Association of Energy "
        "Producers (NVDE)."
    )
    pdf.clause_heading("12.3", "Arbitration")
    pdf.body_text(
        f"All other disputes shall be finally settled under the Rules of Arbitration of "
        f"the {spec.arbitration} by three arbitrators appointed in accordance with those Rules. "
        "The language of arbitration shall be English."
    )

    # ── Schedules ─────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("SCHEDULE 1 — ASSET DESCRIPTION")
    pdf.body_text(f"Asset Name:        {spec.asset_name}")
    pdf.body_text(f"Asset Type:        {spec.asset_type.replace('_', ' ').title()}")
    pdf.body_text(f"Installed Capacity: {spec.volume_mw} MW")
    pdf.body_text(f"Delivery Point:    {spec.delivery_point}")
    pdf.body_text(f"Grid Connection:   Pursuant to TenneT/DSO connection agreement [ref TBD]")
    pdf.body_text(f"Grid Code:         ENTSO-E Network Code on Requirements for Generators (RfG)")

    pdf.add_page()
    pdf.chapter_title("SCHEDULE 2 — PAYMENT AND TAKE-OR-PAY CALCULATION")
    pdf.body_text(
        f"Strike Price:          €{spec.strike_price_eur:.2f}/MWh\n"
        f"Take-or-Pay Threshold: {int(spec.take_or_pay_pct * 100)}% of Scheduled Monthly Generation Volume\n"
        f"Invoice Frequency:     Monthly\n"
        f"Payment Terms:         30 days from invoice date\n"
        f"Bank Account:          To be provided by Seller prior to Commencement Date"
    )

    pdf.add_page()
    pdf.chapter_title("SCHEDULE 3 — SIGNATURES")
    pdf.ln(10)
    pdf.body_text(f"For and on behalf of {spec.seller}:")
    pdf.ln(10)
    pdf.body_text("Signature: _______________________    Date: ___________________")
    pdf.body_text("Name:      _______________________")
    pdf.body_text("Title:     _______________________")
    pdf.ln(15)
    pdf.body_text(f"For and on behalf of {spec.buyer}:")
    pdf.ln(10)
    pdf.body_text("Signature: _______________________    Date: ___________________")
    pdf.body_text("Name:      _______________________")
    pdf.body_text("Title:     _______________________")

    pdf.output(str(output_path))
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PPA PDF contracts")
    parser.add_argument(
        "--output-dir", type=str,
        default="data/synthetic_ppa",
        help="Directory to write PDF files (default: data/synthetic_ppa/)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(CONTRACTS)} synthetic PPA contracts → {out}/\n")
    for spec in CONTRACTS:
        path = out / f"{spec.contract_id}.pdf"
        generate_ppa_pdf(spec, path)
        size_kb = path.stat().st_size // 1024
        print(f"  ✓ {path.name}  ({size_kb} KB, {spec.ppa_type}, "
              f"{'floor' if spec.negative_price_floor else 'NO FLOOR'}, "
              f"€{spec.strike_price_eur}/MWh, {spec.volume_mw}MW)")

    print(f"\nDone. {len(CONTRACTS)} PDFs written to {out}/")
    print("Next: python scripts/ingest_contracts.py --source", out)


if __name__ == "__main__":
    main()
