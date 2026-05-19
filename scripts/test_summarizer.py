#!/usr/bin/env python3
"""
Test the Mistral-backed IPD summarizer — two modes:

  1. Direct (no server, no MongoDB — fastest, tests the Mistral call itself):
       python3 scripts/test_summarizer.py

  2. Full IPD endpoint (server must be running, MongoDB needed):
       python3 scripts/test_summarizer.py --live
       python3 scripts/test_summarizer.py --live --port 8080

The direct mode exercises the exact same code path that
POST /process_pdcm_message uses: summarize_agent() -> Mistral Large.
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Realistic IPD payload — mirrors what /process_pdcm_message expects.
# discharge_summary is raw HTML (exactly what comes from the EHR/CIS).
# dates are in dd/mm/yyyy as the orchestrator parses them that way.
# ---------------------------------------------------------------------------
SAMPLE_IPD_PAYLOAD = {
    "prompt_code": "openai_gen_ip_1",
    "data_payload": {
        "patient_id": "TEST-IPD-001",
        "gender": "Male",
        "age": "58",
        "visits": [
            {
                "date": "10/05/2026",
                "discharge_date": "17/05/2026",
                "visit_type": "IP",
                "is_processed": False,
                "tests": [
                    {"test": "HbA1c", "value": "8.2"},
                    {"test": "Creatinine", "value": "1.4"},
                ],
                "medications": "Carvedilol 6.25mg BD, Furosemide 40mg OD, Sacubitril/Valsartan 49/51mg BD, Metformin 500mg BD",
                "discharge_summary": """
<html><body>
<h2>Discharge Summary</h2>
<p><b>Admission:</b> 10-May-2026 &nbsp; <b>Discharge:</b> 17-May-2026</p>
<p><b>Diagnosis:</b> Acute Decompensated Heart Failure (NYHA Class III), Type 2 Diabetes Mellitus</p>
<p><b>Chief Complaint:</b> Progressive breathlessness on exertion for 7 days,
bilateral pedal oedema, orthopnoea (2-pillow).</p>
<p><b>History:</b> Known dilated cardiomyopathy (EF 35%) on Carvedilol, Furosemide,
Sacubitril/Valsartan. Recent dietary non-compliance (high sodium).</p>
<p><b>Key Investigations:</b> BNP 1840 pg/mL, Creatinine 1.4 mg/dL, HbA1c 8.2%,
Echo: EF 30%, moderate mitral regurgitation.</p>
<p><b>Treatment:</b> IV Furosemide 80mg BD for 3 days, fluid restriction 1.5L/day,
Metformin held, sliding-scale insulin initiated. Carvedilol halved during acute phase.</p>
<p><b>Procedures:</b> 2D Echocardiogram on admission.</p>
<p><b>Discharge Condition:</b> Stable, SpO2 98% on room air, no pedal oedema.</p>
<p><b>Follow-up:</b> Cardiology OPD in 1 week. BNP + renal panel in 2 weeks.
Diabetes educator referral arranged.</p>
</body></html>
""",
            }
        ],
    },
}

# Standalone discharge-only payload (for /summaries/discharge)
DISCHARGE_TEXT = SAMPLE_IPD_PAYLOAD["data_payload"]["visits"][0]["discharge_summary"]
DISCHARGE_DATE = "17-May-2026"


# ---------------------------------------------------------------------------
# Mode 1 — direct Python (no server, no MongoDB)
# ---------------------------------------------------------------------------
def test_direct():
    print("=" * 60)
    print("MODE: Direct Python — no server or MongoDB needed")
    print(f"  SUMMARIZER_PROVIDER : {os.getenv('SUMMARIZER_PROVIDER', 'mistral')}")
    print(f"  SUMMARIZER_MODEL    : {os.getenv('SUMMARIZER_MODEL', 'mistral-large-latest')}")
    print("=" * 60)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from docstribe_summarizer import summarize_agent

    visit = SAMPLE_IPD_PAYLOAD["data_payload"]["visits"][0]
    raw_discharge = visit["discharge_summary"]
    discharge_date = "17-May-2026"  # already formatted as the orchestrator would format it

    print("\n[TEST] summarize_agent() — the exact call made inside /process_pdcm_message")
    print(f"  Input length : {len(raw_discharge)} chars (HTML discharge note)")
    print(f"  Discharge date: {discharge_date}")
    print("  Calling Mistral Large ... ", end="", flush=True)

    try:
        result = summarize_agent(raw_discharge, discharge_date)
        print("OK")
        print("\n--- Mistral Large output ---")
        print(json.dumps(result, indent=2))
        print("\nKeys returned:", list(result.keys()))
    except Exception as exc:
        print(f"\nFAIL — {exc}")
        sys.exit(1)

    print("\nDirect test passed. Mistral Large is working correctly.")


# ---------------------------------------------------------------------------
# Mode 2 — full IPD endpoint (needs live server + MongoDB)
# ---------------------------------------------------------------------------
def test_live(port: int):
    import requests

    base = f"http://localhost:{port}"
    print("=" * 60)
    print(f"MODE: Live server at {base}")
    print("  This hits POST /process_pdcm_message — full IPD pipeline")
    print("  (requires server + MongoDB to be running)")
    print("=" * 60)

    # Step 1: ingest IPD record (triggers Mistral summarization internally)
    print("\n[1/2] POST /process_pdcm_message — ingest IPD record ...")
    try:
        resp = requests.post(
            f"{base}/process_pdcm_message",
            json=SAMPLE_IPD_PAYLOAD,
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"  HTTP {resp.status_code} — PASS")
        visits = data.get("visits", [])
        if visits:
            ds = visits[0].get("discharge_summary", {})
            print("  Discharge summary keys from Mistral:", list(ds.keys()) if isinstance(ds, dict) else type(ds))
            print(json.dumps(visits[0], indent=2))
    except requests.exceptions.ConnectionError:
        print(f"  FAIL — server not reachable at {base}")
        print("  Start it with: python3 ai_orchestrator.py")
        sys.exit(1)
    except Exception as exc:
        print(f"  FAIL — {exc}")
        sys.exit(1)

    # Step 2: also hit /summaries/discharge directly for a focused check
    print("\n[2/2] POST /summaries/discharge — standalone discharge summarizer ...")
    try:
        resp = requests.post(
            f"{base}/summaries/discharge",
            json={"medical_text": DISCHARGE_TEXT, "discharge_date": DISCHARGE_DATE},
            timeout=60,
        )
        resp.raise_for_status()
        print(f"  HTTP {resp.status_code} — PASS")
        print(json.dumps(resp.json(), indent=2))
    except Exception as exc:
        print(f"  FAIL — {exc}")
        sys.exit(1)

    print("\nAll live IPD tests passed.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Mistral-backed IPD summarizer")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Hit the live Flask server (needs python3 ai_orchestrator.py running)",
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.live:
        test_live(args.port)
    else:
        test_direct()
