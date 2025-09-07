from __future__ import annotations
from typing import Dict, Any
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.sql import SparkSession

from hr_metadata_service import HRMetadataService
from workforce_nlp_processor import WorkforceNLPProcessor
from hr_query_engine import execute_plan

APP_VERSION = "0.1.0"

# Initialize Spark + metadata + NLP once
spark = SparkSession.builder.appName("workforce-idqs-api").getOrCreate()
meta = HRMetadataService(data_dir=os.environ.get("DATA_DIR","data")).load()
nlp = WorkforceNLPProcessor(meta)

app = FastAPI(
    title="Workforce IDQS",
    version=APP_VERSION,
    description="Intelligent HR analytics over Workforce & Assignment (PySpark + HF NLP + CSN glossary).",
    openapi_version="3.1.0"
)

class NLQ(BaseModel):
    query: str

class Plan(BaseModel):
    plan: Dict[str, Any]

@app.get("/health", summary="Health")
def health():
    return {"status":"ok","engine":"spark","spark_version": spark.version}

@app.post("/nlp/query", summary="Nlp Query")
def nlp_query(payload: NLQ):
    plan = nlp.parse(payload.query)
    if plan.get("intent") == "unknown":
        raise HTTPException(status_code=400, detail=plan.get("error","Unable to parse query"))
    try:
        res = execute_plan(spark, meta, plan)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {e}")

@app.post("/plan/execute", summary="Execute Raw Plan")
def execute_raw_plan(payload: Plan):
    try:
        return execute_plan(spark, meta, payload.plan)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------- Demo analytics endpoints ----------
@app.get("/analytics/headcount_by_division", summary="Headcount By Division")
def headcount_by_division():
    plan = {
        "intent":"organizational_analysis","tables":["workforce"],
        "select":["division"],"group_by":["division"],
        "aggregations":[{"function":"count_distinct","column":"personIdExternal","alias":"headcount"}],
        "order_by":[{"column":"headcount","direction":"desc"}],
        "limit":1000,"glossary_refs":["headcount"],
        "business_context":"Headcount grouped by division"
    }
    return execute_plan(spark, meta, plan)

@app.get("/analytics/headcount_by_division_status", summary="Headcount By Division Status")
def headcount_by_division_status():
    plan = {
        "intent":"organizational_analysis","tables":["workforce"],
        "select":["division","status"],"group_by":["division","status"],
        "aggregations":[{"function":"count_distinct","column":"personIdExternal","alias":"headcount"}],
        "order_by":[{"column":"headcount","direction":"desc"}],
        "limit":1000,"glossary_refs":["headcount"],
        "business_context":"Headcount grouped by division and employment status"
    }
    return execute_plan(spark, meta, plan)

@app.get("/analytics/hired_last_months_with_active_assignments", summary="Hired Last Months With Active Assignments")
def hired_last_months_with_active_assignments(months: int = 6):
    from datetime import date, timedelta
    since = (date.today() - timedelta(days=30*months)).isoformat()
    plan = {
        "intent":"cross_table_analysis",
        "tables":["workforce","assignment"],
        "joins":[{"left_table":"workforce","right_table":"assignment","on":"personIdExternal"}],
        "where":[
            {"table":"workforce","column":"startDate","operator":">=","value":since},
            {"table":"assignment","column":"assignmentStatus","operator":"=","value":"A"}
        ],
        "select":["personIdExternal","startDate","division","department","assignmentType","assignmentStatus"],
        "order_by":[{"column":"startDate","direction":"desc"}],
        "limit":1000,
        "glossary_refs":["active_assignment"],
        "business_context":f"Hired since {since} with active assignments"
    }
    return execute_plan(spark, meta, plan)

@app.get("/analytics/avg_tenure_by_department_full_time", summary="Avg Tenure By Department Full Time")
def avg_tenure_by_department_full_time():
    plan = {
        "intent":"workforce_analytics",
        "tables":["workforce","assignment"],
        "joins":[{"left_table":"workforce","right_table":"assignment","on":"personIdExternal"}],
        "where":[
            {"table":"assignment","column":"assignmentType","operator":"in","value":["FULL-TIME","FULL TIME","FT"]},
            {"table":"workforce","column":"status","operator":"=","value":"A"}
        ],
        "select":["department"],
        "group_by":["department"],
        "aggregations":[{"function":"avg_tenure_years","column":"startDate","alias":"avg_tenure_years"}],
        "order_by":[{"column":"avg_tenure_years","direction":"desc"}],
        "limit":1000,"glossary_refs":["tenure_years","full_time","active_employees"],
        "business_context":"Average tenure by department for full-time active employees"
    }
    return execute_plan(spark, meta, plan)

@app.get("/analytics/managers_direct_reports_gt", summary="Managers Direct Reports Gt")
def managers_direct_reports_gt(threshold: int = 5):
    plan = {
        "intent":"management_hierarchy","tables":["workforce"],
        "select":["managerId","department"],"group_by":["managerId","department"],
        "aggregations":[{"function":"count","column":"personIdExternal","alias":"direct_reports"}],
        "having":[{"column":"direct_reports","operator":">","value":threshold}],
        "order_by":[{"column":"direct_reports","direction":"desc"}],
        "limit":1000,"glossary_refs":[],
        "business_context":f"Managers with >{threshold} direct reports by department"
    }
    return execute_plan(spark, meta, plan)
