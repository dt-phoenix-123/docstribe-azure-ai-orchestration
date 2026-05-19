#!/usr/bin/env python3
"""
Test the IP Admission Advice workflow — 2-step flow:

  Step 1: POST /process_ip_advise_patient
          → OPD processing + lab summarization (Mistral Large)
          → Saved to ip_recommendation_patient_collection with status: pending

  Step 2: POST /collect_ip_advise_pending_request
          → GPT-4.1 structured output → PatientCarePlan
          → Status → completed

Usage (server must be running: python3 ai_orchestrator.py):
  python3 scripts/test_ip_advise.py
  python3 scripts/test_ip_advise.py --port 8080
"""

import argparse
import json
import sys
import time
import requests

# ---------------------------------------------------------------------------
# Realistic OPD payload for a patient being advised IP admission.
# ip_advised="Yes" is the key flag that makes this an admission-advice case.
# dates in dd/mm/yyyy — orchestrator parses them this way.
# ---------------------------------------------------------------------------
PATIENT_ID = "TEST-IP-ADVISE-001"

IP_ADVISE_PAYLOAD = {
    "prompt_code": "openai_gen_op_1",
    "data_payload": {
        "patient_id": PATIENT_ID,
        "gender": "Male",
        "age": "62",
        "visits": [
            {
                "date": "19/05/2026",
                "visit_type": "OP",
                "is_processed": False,
                "ip_advised": "Yes",
                "follow_up_date": "26/05/2026",
                "clinical_note": (
                    "Patient presenting with chest pain on exertion (CCS Grade 3), "
                    "progressive dyspnoea (NYHA Class III), and two episodes of pre-syncope "
                    "in the past week. Known case of severe aortic stenosis (valve area 0.7 cm²) "
                    "on Echo. Referred for urgent TAVR/SAVR evaluation. "
                    "Advised immediate IP admission for pre-operative workup and cardiac clearance."
                ),
                "medications": (
                    "Tab. Furosemide 40mg OD, Tab. Carvedilol 6.25mg BD, "
                    "Tab. Aspirin 75mg OD, Tab. Atorvastatin 40mg HS"
                ),
                "diagnosis_advised_list": (
                    "Severe Aortic Stenosis, Stable Ischemic Heart Disease, "
                    "Hypertension Grade 2"
                ),
                "investigations": (
                    "2D Echo, Coronary Angiogram, Chest X-Ray PA view, "
                    "CBC, RFT, LFT, Coagulation profile, Blood grouping"
                ),
                "tests": [
                    {"test": "HbA1c",       "value": "7.4"},
                    {"test": "Creatinine",  "value": "1.3"},
                    {"test": "Hemoglobin",  "value": "10.2"},
                    {"test": "FBS",         "value": "138"},
                ],
            }
        ],
    },
}


def pretty(data: dict) -> str:
    return json.dumps(data, indent=2)


def run(base: str):
    session = requests.Session()

    # -----------------------------------------------------------------------
    # Step 1 — ingest the OPD record as an IP-advise candidate
    # -----------------------------------------------------------------------
    print("=" * 62)
    print("STEP 1 — POST /process_ip_advise_patient")
    print(f"  Patient ID : {PATIENT_ID}")
    print(f"  ip_advised : Yes  |  diagnosis: Severe Aortic Stenosis")
    print("=" * 62)

    resp1 = session.post(
        f"{base}/process_ip_advise_patient",
        json=IP_ADVISE_PAYLOAD,
        timeout=90,
    )
    resp1.raise_for_status()
    step1 = resp1.json()

    status = step1.get("status")
    rec_id = step1.get("ip_recommendation_patient_id", "")
    print(f"  HTTP {resp1.status_code} — status: {status}")
    print(f"  ip_recommendation_patient_id : {rec_id}")

    if status != "pending":
        print("\n  [!] Expected status=pending. Full response:")
        print(pretty(step1))
        print("\n  Tip: ensure ip_advised=Yes and tests/diagnosis are present.")
        sys.exit(1)

    print("\n  clinical_assessment keys :", list((step1.get("clinical_assessment") or {}).keys()))
    print("  agentic_summary keys     :", list((step1.get("agentic_summary") or {}).keys()))
    print("\n  Step 1 passed — record is pending in MongoDB.")

    # -----------------------------------------------------------------------
    # Step 2 — generate the IP admission care plan via GPT-4.1
    # -----------------------------------------------------------------------
    print()
    print("=" * 62)
    print("STEP 2 — POST /collect_ip_advise_pending_request")
    print("  Generating PatientCarePlan via GPT-4.1 structured output ...")
    print("=" * 62)

    resp2 = session.post(
        f"{base}/collect_ip_advise_pending_request",
        json={"patient_ids": [PATIENT_ID]},
        timeout=120,
    )
    resp2.raise_for_status()
    step2 = resp2.json()

    overall = step2.get("status")
    processed = step2.get("processed", 0)
    failed = step2.get("failed", 0)
    print(f"  HTTP {resp2.status_code} — status: {overall}  |  processed: {processed}  |  failed: {failed}")

    if failed:
        print("\n  Failures:")
        for f in step2.get("failures", []):
            print(f"    patient_id={f.get('patient_id')}  error={f.get('error')}")
        sys.exit(1)

    responses = step2.get("responses", [])
    if not responses:
        print("\n  No responses returned. Is the patient still pending?")
        sys.exit(1)

    rec = responses[0].get("recommendation", {})
    print("\n  --- PatientCarePlan ---")

    # Talking points
    tp = rec.get("talking_points", {})
    print(f"\n  talking_points.important_pointers:")
    for p in (tp.get("important_pointers") or []):
        print(f"    • {p}")

    # Actionables
    act = (tp.get("actionables") or {})
    print(f"\n  actionables.procedure_criticiality : {act.get('procedure_criticiality')}")
    print(f"  actionables.surgery_advised        : {act.get('surgery_advised')}")
    print(f"  actionables.admission_advised_date : {act.get('admission_advised_date')}")
    deferred = act.get("deferred_timeline") or {}
    print(f"  actionables.deferred_timeline      : {deferred.get('time')}  red_flags: {deferred.get('red_flags')}")

    # IP advised meta
    meta = rec.get("ip_advised_meta") or {}
    print(f"\n  ip_advised_meta.admission_type    : {meta.get('admission_type')}")
    print(f"  ip_advised_meta.procedure_name    : {meta.get('procedure_name')}")
    print(f"  ip_advised_meta.procedure_setting : {meta.get('procedure_setting')}")
    print(f"  ip_advised_meta.procedure_cost    : ₹{meta.get('procedure_cost')}")
    print(f"  ip_advised_meta.procedure_type    : {meta.get('procedure_type')}")
    print(f"  ip_advised_meta.is_robotic        : {meta.get('is_robotic')}")

    # Surgery info
    si = rec.get("surgery_info") or {}
    print(f"\n  surgery_info.surgery_advised      : {si.get('surgery_advised')}")
    print(f"  surgery_info.simple_name          : {si.get('simple_name')}")
    print(f"  surgery_info.admission_advised_date: {si.get('admission_advised_date')}")

    print(f"\n  follow_up_date : {rec.get('follow_up_date')}")
    print(f"  doctor_name    : {rec.get('doctor_name')}")
    print(f"  department     : {rec.get('department')}")

    print("\n  Full recommendation JSON:")
    print(pretty(rec))
    print("\nAll steps passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test IP Admission Advice workflow")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    base = f"http://localhost:{args.port}"
    print(f"Target: {base}\n")

    try:
        run(base)
    except requests.exceptions.ConnectionError:
        print(f"ERROR — server not reachable at {base}")
        print("Start it with: python3 ai_orchestrator.py")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP ERROR — {e}")
        try:
            print(e.response.json())
        except Exception:
            print(e.response.text)
        sys.exit(1)
