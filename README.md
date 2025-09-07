# Workforce IDQS — Intelligent HR Analytics (PySpark + FastAPI)

A minimal end-to-end Workforce Analytics prototype:
- **PySpark** data processing over Parquet (workforce + assignment)
- **CSN/JSON metadata** → logical→physical mappings + HR glossary
- **Rule-based HR NLP** → normalized query plans
- **FastAPI** endpoints + **Jupyter** notebook demo
- One-command startup with **docker compose**

---

## 1) Quick Start

```bash
# from repo root
docker compose up -d
