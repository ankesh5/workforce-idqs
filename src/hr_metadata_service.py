from __future__ import annotations
from typing import Dict, Any
import json
import os

DEFAULT_ORG_HIERARCHY = ["division", "department", "location"]
DEFAULT_JOIN_KEY = "personIdExternal"

# Minimal default glossary terms used by the engine
DEFAULT_GLOSSARY = {
    "headcount": {
        "aggregation": {"function": "count_distinct", "column": "personIdExternal", "alias": "headcount"},
        "description": "count of unique employees"
    },
    "active_employees": {
        "where": [{"table": "workforce", "column": "status", "operator": "=", "value": "A"}],
        "description": "workforce.status = 'A'"
    },
    "active_assignment": {
        "where": [{"table": "assignment", "column": "assignmentStatus", "operator": "=", "value": "A"}],
        "description": "assignment.assignmentStatus = 'A'"
    },
    "primary_assignment": {
        "where": [{"table": "assignment", "column": "isPrimary", "operator": "=", "value": True}],
        "description": "assignment.isPrimary = true"
    },
    "full_time": {
        "where": [{"table":"assignment","column":"assignmentType","operator":"in","value":["FULL-TIME","FULL TIME","FT"]}],
        "description": "assignment.assignmentType in ('FULL-TIME','FULL TIME','FT')"
    },
    "tenure_years": {
        "formula": "datediff(coalesce(endDate,today), startDate) / 365.25",
        "description": "Tenure computed from startDate to current date (or endDate if present)"
    }
}

class HRMetadataService:
    """
    Loads CSN JSON files from data_dir and exposes:
      - tables: workforce / assignment mappings
      - business_terms (glossary)
      - join_key
      - org_hierarchy
      - data_paths
    """
    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = data_dir
        self._meta: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        wf_csn = os.path.join(self.data_dir, "workforce_csn.json")
        asg_csn = os.path.join(self.data_dir, "assignment_csn.json")

        wf = self._load_one(wf_csn, fallback_mapping={
            "personIdExternal": "personIdExternal",
            "startDate": "startDate",
            "status": "status",
            "division": "division",
            "department": "department",
            "location": "location",
            # optional but useful
            "managerId": "manager",
            "endDate": None
        })

        asg = self._load_one(asg_csn, fallback_mapping={
            "personIdExternal": "personIdExternal",
            "assignmentStatus": "assignmentStatus",
            "isPrimary": "isPrimary",
            "assignmentType": "assignmentType",
            "standardHours": "standardHours",
            "division": "division",
            "department": "department",
            "location": "location",
            "startDate": "startDate",
            "endDate": "endDate"
        })

        glossary = self._merge_glossary(wf.get("business_terms"), asg.get("business_terms"))

        self._meta = {
            "tables": {
                "workforce": {"mapping": wf["mapping"]},
                "assignment": {"mapping": asg["mapping"]}
            },
            "business_terms": glossary,
            "join_key": wf.get("join_key") or asg.get("join_key") or DEFAULT_JOIN_KEY,
            "org_hierarchy": wf.get("organizational_hierarchy") or DEFAULT_ORG_HIERARCHY,
            "data_paths": {
                "workforce": os.path.join(self.data_dir, "workforce_data.parquet"),
                "assignment": os.path.join(self.data_dir, "assignment_data.parquet")
            }
        }
        return self._meta

    # ---------- helpers ----------
    def _load_one(self, path: str, fallback_mapping: Dict[str, Any]) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {"mapping": fallback_mapping}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mapping = data.get("mapping") or fallback_mapping
        # sanitize: ensure optional keys exist even if None
        for k in fallback_mapping:
            mapping.setdefault(k, fallback_mapping[k])

        out = {
            "mapping": mapping,
            "business_terms": data.get("business_terms") or {},
            "organizational_hierarchy": data.get("organizational_hierarchy") or DEFAULT_ORG_HIERARCHY,
            "join_key": data.get("join_key") or DEFAULT_JOIN_KEY
        }
        return out

    def _merge_glossary(self, wf_terms, asg_terms) -> Dict[str, Any]:
        g = dict(DEFAULT_GLOSSARY)
        for src in (wf_terms or {}, asg_terms or {}):
             # If a tuple/list sneaks in, coerce it to {} to avoid .items() crash
             if not isinstance(src, dict):
                continue
             for k, v in src.items():
                 g[k] = v
        return g


    # convenience getters
    def meta(self) -> Dict[str, Any]:
        return self._meta

    def business_term(self, key: str) -> Dict[str, Any] | None:
        return (self._meta.get("business_terms") or {}).get(key)

    def table_mapping(self, table: str) -> Dict[str, str]:
        return ((self._meta.get("tables") or {}).get(table) or {}).get("mapping") or {}
