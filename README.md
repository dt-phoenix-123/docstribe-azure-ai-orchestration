# Docstribe Orchestrator API

This service coordinates OPD/IPD processing pipelines using Azure OpenAI batches,
MongoDB persistence (or a local state store), and pydantic validation. Below is a
reference for the available HTTP endpoints and the expected payloads.

Base URL defaults to `http://localhost:8080` unless `PORT` is overridden.

## Environment Requirements

At minimum set the following in `.env` (or export them before launching):

```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_DEPLOYMENT=Docstribe-o3
AZURE_GROK_BASE_URL=https://<resource>.services.ai.azure.com/openai/v1/
AZURE_GROK_DEPLOYMENT=Docstribe-grok-3
DOCSTRIBE_MONGODB_URI=mongodb+srv://...
LLM_ORCHESTRATOR_DB=max_azure_check
```

Install dependencies: `pip install -r requirements.txt`

Start the app: `python3 ai_orchestrator.py`

## Endpoints

### `POST /process_opd_message`
Ingests a new OPD request, normalises visit dates/tests, summarises clinical data,
and stores the case with `status` set to `pending`.

```json
{
  "prompt_code": "openai_gen_op_1",
  "data_payload": {
    "patient_id": "67cecdfa...",
    "gender": "male",
    "age": "41",
    "visits": [
      {
        "date": "02/01/2025",
        "tests": [{"test": "No Test", "value": "-"}],
        "visit_type": "OP",
        "is_processed": false,
        "clinical_note": "...",
        "medications": "...",
        "ip_advised": "No"
      }
    ]
  }
}
```

### `POST /collect_opd_pending_requests`
Generates batch-ready JSONL entries for OPD cases still marked `pending` and
updates their status to `processing`.

```json
{
  "responses": [
    {
      "patient_details": {"patient_id": "67cecdfa...", "age": "41", "gender": "Male"},
      "abnormalities_identified": [...],
      "clinical_assessment": {...},
      "agentic_summary": {...},
      "radiology_findings": {}
    }
  ]
}
```

### `POST /upload_batch`
Uploads a JSONL file to Azure OpenAI, creates a batch job, and records the batch
metadata. `type` must be `"PDCM"` or `"OPD"`. `file_url` is a local path to the
JSONL file generated earlier.

```json
{
  "type": "OPD",
  "file_url": "/Users/<user>/Max_AI_Orchestration/data/jsonl/continental_opd_jsonl_file.jsonl"
}
```

### `POST /check_batch_status`
Checks the status of an Azure OpenAI batch job.

```json
{ "batch_id": "batch_12345" }
```

### `POST /retrieve_results`
Downloads the completed batch output file from Azure OpenAI, updates MongoDB with
the model responses, and removes the temporary file. `file_type` chooses the
collection (`"PDCM"` or `"OPD"`). `file_id` is the `output_file_id` returned by
Azure when the batch completed.

```json
{
  "file_type": "OPD",
  "file_id": "file-67890"
}
```

### `POST /view_opd_output`
Fetches and validates completed OPD cases. The response contains an array of
pydantic-validated records (or `status="failed"` if validation fails).

```json
{ "patient_ids": ["67cecdfa..."] }
```

### `POST /process_pdcm_message`
Equivalent to `process_opd_message` but for IPD/PDCM flows (discharge summaries,
lab tests, etc.). Payload mirrors the IPD data structure.

### `POST /collect_pending_requests`
Collects pending IPD cases, writes batch JSONL (care plan prompts), and marks
them as `processing`.

### `POST /upload_opd_records`
Uploads an Excel file containing a `json_output` column and publishes each entry
to the OPD Pub/Sub topic.

### `POST /upload_pdcm_records`
Same pattern as above but for IPD records.

### `POST /process_op` / `POST /process_ip`
Directly publishes raw OPD/IPD payloads to Pub/Sub topics and stores the original
payload in the corresponding `<type>_workflow_log` collection.

## Summariser Endpoints

### `POST /summaries/discharge`
Summarises discharge text using the Azure Grok deployment.

```json
{
  "medical_text": "<html or plain text>",
  "discharge_date": "12-Sep-2025",
  "model": "groq"  // optional; default is groq
}
```

### `POST /summaries/opd`
Triggers the OPD summariser (lab data + clinical notes).

```json
{
  "patient_data": {
    "visits": [...]
  }
}
```

### `POST /summaries/diagnostics`
Summarises diagnostic PDFs.

```json
{ "pdf_url": "https://.../report.pdf" }
```

## OPD JSON Validator

The `opd_pydantic_validator.py` module powers OPD response validation. It honours
`DOCSTRIBE_MONGODB_URI` / `LLM_ORCHESTRATOR_DB` for cohort enrichment but falls
back to a no-op if MongoDB is unavailable. Validation failures are printed to the
console for debugging.

## Automation Workflow (Redis-backed)

The `automation/` package provides a configurable orchestrator that automates the
full OPD batch lifecycle:

1. **Queue intake**: enqueue payloads on the Redis list `opd:incoming` to simulate
   `/process_op` requests.
2. **Incoming worker**: calls `/process_opd_message` for each queued job.
3. **Pending collector**: polls MongoDB for `status="pending"` and calls
   `/collect_opd_pending_requests` when documents are available.
4. **Batch submitter**: monitors the JSONL file and triggers `/upload_batch` once
   the configured threshold is reached.
5. **Batch poller**: checks `/check_batch_status` every `AUTOMATION_POLL_INTERVAL`
   seconds; once completed, it queues the output file for retrieval.
6. **Result worker**: invokes `/retrieve_results` for completed batches.

### Enabling the orchestrator

```
export AUTOMATION_ENABLED=true
export AUTOMATION_BASE_URL=http://localhost:8080
export AUTOMATION_REDIS_URL=redis://localhost:6379/0
export AUTOMATION_BATCH_THRESHOLD=100
export AUTOMATION_PDCM_QUEUE=pdcm:incoming
python3 -m automation.orchestrator
```

To enqueue jobs programmatically:

```python
from automation.redis_queue import RedisQueue

queue = RedisQueue("opd:incoming")
queue.push({
    "prompt_code": "openai_gen_op_1",
    "data_payload": {...}
})
```

All workers run as daemon threads. Stop them with `Ctrl+C`.

## Troubleshooting

- Status remains `pending` after `/collect_opd_pending_requests`: ensure the
  payload wraps each record under `responses` and the `patient_details.patient_id`
  matches the document inserted by `/process_opd_message`.
- Azure batch upload fails with `url_mismatch`: make sure your JSONL lines use
  the `/v1/chat/completions` endpoint (the orchestrator’s helpers do this by
  default).
- Validation errors during `/view_opd_output`: check the console logs for the
  pydantic error details; if cohort enrichment is needed, verify MongoDB access.
