README.md
# Workforce IDQS – Intelligent Data Querying System (HR Analytics)

This project is a prototype of an **Intelligent Data Querying System (IDQS)** designed for **HR analytics**. 
 
It demonstrates how **HR business questions** (e.g., *“Show headcount by division and employment status”*) can be translated into **PySpark queries** using:

- **CSN metadata catalogs** (to define schema, glossary, and joins),
- **Hugging Face NLP** (to interpret HR questions into structured query plans),
- **PySpark** (for distributed execution on Workforce and Assignment datasets),
- **FastAPI** (to expose analytics endpoints and NLP query interface).

---

##  HR Context

HR organizations often face challenges such as:

- Tracking **headcount, attrition, and tenure** across divisions/departments,
- Understanding **assignment relationships** (e.g., primary vs. multiple active assignments),
- Monitoring **managerial span of control** (direct reports),
- Integrating business rules (e.g., active employees, full-time definitions) consistently,
- Ensuring that **out-of-scope queries** (e.g., “payroll by shoe size”) fail gracefully.

This prototype addresses those by combining **HR domain glossary + NLP + Spark**.

---

## Architecture Flow



User (HR Query)
│
▼
FastAPI (/nlp/query, /analytics/*)
│
▼
WorkforceNLPProcessor (HF zero-shot)
├─ Glossary/Schema guard (reject unknown terms)
├─ Override (headcount+division+status)
└─ Plan JSON
│
▼
hr_query_engine (PySpark)
├─ Load parquet datasets
├─ Canonicalize columns (via CSN mapping)
├─ Apply filters, joins, aggregations
└─ Return JSON rows
│
▼
FastAPI JSON Response


---

##  Project Structure


workforce-idqs/
├─ docker-compose.yml
├─ Dockerfile.api
├─ requirements.txt
├─ .env.example
├─ README.md
├─ demo_hr_analytics.ipynb
├─ data/
│  ├─ workforce_data.parquet/
│  ├─ assignment_data.parquet/
│  ├─ workforce_csn.json
│  └─ assignment_csn.json
└─ src/
   ├─ hr_metadata_service.py     # Loads CSN metadata + glossary
   ├─ hr_query_engine.py         # Executes Spark queries from plan
   ├─ workforce_nlp_processor.py # NL → Plan via Hugging Face + glossary guard
   └─ workforce_api.py  


---

##  Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)  
- Hugging Face API token ([https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

---

##  Setup Instructions

1. **Unzip** the project
2. copy .env.example to .env and set your Hugging Face token:

   ```env
   HF_TOKEN=hf_****************************
   HF_MIN_SCORE=0.6

## Run
docker compose up -d --build


Wait until logs show:

Uvicorn running on http://0.0.0.0:8000
Application startup complete.

## Test
Health Check
curl http://localhost:8000/health

Swagger Docs

http://localhost:8000/docs

Valid Query
$body = @{ query = "Show headcount by division and employment status" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/nlp/query" -Method Post -Body $body -ContentType "application/json"

Invalid Query (graceful failure)
$body = @{ query = "Give me payroll totals by shoe size" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/nlp/query" -Method Post -Body $body -ContentType "application/json"


Expected response:

{
  "intent": "unknown",
  "error": "Query references terms not found in the HR glossary/schema.",
  "unknown_terms": ["payroll","shoe","size"]
}

# Key Endpoints

GET /health → service check

GET /analytics/headcount_by_division

GET /analytics/headcount_by_division_status

GET /analytics/hired_last_months_with_active_assignments?months=6

GET /analytics/avg_tenure_by_department_full_time

GET /analytics/managers_direct_reports_gt?threshold=5

POST /nlp/query → natural language → Spark query

POST /plan/execute → run a raw plan JSON

# Stop
docker compose down