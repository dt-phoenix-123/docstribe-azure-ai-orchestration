import pandas as pd
from pymongo import MongoClient
import requests
from datetime import datetime
import time


def change_appointment_date(payload):
    data_payload = payload.get("data_payload")
    visits = data_payload.get("visits", [])
    for visit in visits:
        follow_up_date = visit.get("follow_up_date", "")
        if follow_up_date:
            try:
                wrong_date = datetime.strptime(follow_up_date, "%m/%d/%Y")
            except:
                wrong_date = datetime.strptime(follow_up_date, "%d-%m-%Y")
            corrected_date = wrong_date.strftime("%d/%m/%Y")
            visit["follow_up_date"] = corrected_date
    return payload


def replace_nan_with_empty_string(obj):
    if isinstance(obj, dict):
        return {k: replace_nan_with_empty_string(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_empty_string(item) for item in obj]
    elif isinstance(obj, float) and str(obj) == "nan":
        return ""
    else:
        return obj


def call_api(payload, endpoint_url):
    headers = {"Content-Type": "application/json"}
    response = requests.post(endpoint_url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


# MongoDB Connection
uri = "mongodb+srv://dtadmin:docstribe%40123@dt-redcliffe-instance-1.teoq61l.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, tlsAllowInvalidCertificates=True)
db = client["batch_processing"]

# Collections
pdcm_log = db["pdcm_workflow_log"]
opd_log = db["opd_workflow_log"]

# Endpoints
endpoints = {
    "pdcm": "https://docstribe-wockhardt-careflux-877719534636.asia-south2.run.app/process_pdcm_message",
    "opd":  "https://docstribe-wockhardt-careflux-877719534636.asia-south2.run.app/process_opd_message"
}

# Patient list
patient_list = [
    '687dd19c3e22076700a7a461',
    '687dd19d3e22076700a7a46a',
    '687dd19f3e22076700a7a4aa',
    '6881dbdf34b5c97e1175a6ad',
    '6881dbe634b5c97e1175a763',
    '6881dbf134b5c97e1175a861',
    '688e83ae5f0ea220723196ca'
]

cnt = 0
error_cnt = 0
error_patient_ids = []

# Iterate through each patient
for patient_id in patient_list:
    print(f"\n🔎 Checking patient_id: {patient_id}")

    # Try PDCM first
    pdcm_doc = pdcm_log.find_one({"data_payload.patient_id": patient_id})

    if pdcm_doc:
        source = "pdcm"
        payload = pdcm_doc
        print(f"✅ Found in PDCM workflow log: {patient_id}")
    else:
        # Try OPD if not found in PDCM
        opd_doc = opd_log.find_one({"data_payload.patient_id": patient_id})
        if opd_doc:
            source = "opd"
            payload = opd_doc
            print(f"✅ Found in OPD workflow log: {patient_id}")
        else:
            print(f"⚠️ Patient {patient_id} not found in either log.")
            continue

    # Clean and prepare payload
    payload.pop("_id", None)
    payload = replace_nan_with_empty_string(payload)
    # payload = change_appointment_date(payload)  # optional

    try:
        print(f"📤 Sending to {source.upper()} API endpoint...")
        resp = call_api(payload, endpoints[source])
        cnt += 1
        print(f"✅ Successfully pushed {patient_id} → {source.upper()} API")
        print("Response:", resp)
    except Exception as e:
        print(f"❌ Error pushing {patient_id} ({source}): {e}")
        error_cnt += 1
        error_patient_ids.append(patient_id)
        time.sleep(5)
        continue

print("\n==============================")
print(f"✅ Total Successful: {cnt}")
print(f"❌ Total Failed: {error_cnt}")
if error_patient_ids:
    print("Failed patient IDs:", error_patient_ids)
print("==============================")