from __future__ import annotations
from typing import Dict, Any, List
import re
from datetime import date, timedelta

class WorkforceNLPProcessor:
    """
    Lightweight, rule-based NLP → normalized HR query plans.

    Supported intents:
      - organizational_analysis  (e.g., "headcount by division and employment status")
      - cross_table_analysis     (e.g., "hired in the last 6 months with active assignments")
      - workforce_analytics      (e.g., "average tenure by department for full-time employees")
      - management_hierarchy     (e.g., "managers with more than 5 direct reports")

    Returned plan schema matches what hr_query_engine.execute_plan expects.
    """

    INTENT_WORKFORCE   = "workforce_analytics"
    INTENT_ASSIGNMENT  = "assignment_analysis"
    INTENT_ORG         = "organizational_analysis"
    INTENT_MGMT        = "management_hierarchy"
    INTENT_CROSSTABLE  = "cross_table_analysis"

    # -----------------------
    # Public API
    # -----------------------
    def parse(self, text: str) -> Dict[str, Any]:
        q = (text or "").strip().lower()

        # --- 1) Headcount by X (division/department/location) (+ optional employment status)
        if "headcount" in q and ("division" in q or "department" in q or "location" in q):
            dims = self._detect_dims(q, default=["division"])
            by_status = bool(re.search(r"(employment\s*)?status|active|inactive", q))
            select = dims + (["status"] if by_status else [])
            group_by = select.copy()
            return {
                "intent": self.INTENT_ORG,
                "tables": ["workforce"],
                "select": select,
                "group_by": group_by,
                "aggregations": [{"function": "count_distinct", "column": "personIdExternal", "alias": "headcount"}],
                "order_by": [{"column": "headcount", "direction": "desc"}],
                "limit": 1000,
                "business_context": f"Headcount grouped by {', '.join(group_by)}"
            }

        # --- 2) Hired in the last N months with active assignments
        m = re.search(r"hired\s+in\s+the\s+last\s+(\d+)\s*month", q)
        if m and ("active assignment" in q or "active assignments" in q):
            months = int(m.group(1))
            since = (date.today() - timedelta(days=30*months)).isoformat()
            return {
                "intent": self.INTENT_CROSSTABLE,
                "tables": ["workforce", "assignment"],
                "joins": [{"left_table": "workforce", "right_table": "assignment", "on": "personIdExternal"}],
                "where": [
                    {"table": "workforce", "column": "startDate", "operator": ">=", "value": since},
                    {"table": "assignment", "column": "assignmentStatus", "operator": "=", "value": "A"},
                ],
                "select": ["personIdExternal", "startDate", "division", "department", "assignmentType", "assignmentStatus"],
                "order_by": [{"column": "startDate", "direction": "desc"}],
                "limit": 1000,
                "business_context": f"Employees hired since {since} who have active assignments"
            }

        # --- 3) Average tenure by department for full-time employees
        if ("average tenure" in q or "avg tenure" in q) and ("department" in q) and self._mentions_full_time(q):
            return {
                "intent": self.INTENT_WORKFORCE,
                "tables": ["workforce", "assignment"],
                "joins": [{"left_table": "workforce", "right_table": "assignment", "on": "personIdExternal"}],
                "where": [
                    {"table": "assignment", "column": "assignmentType", "operator": "in", "value": ["FULL-TIME","FULL TIME","FT"]},
                    {"table": "workforce", "column": "status", "operator": "=", "value": "A"},
                ],
                "select": ["department"],
                "group_by": ["department"],
                "aggregations": [{"function": "avg_tenure_years", "column": "startDate", "alias": "avg_tenure_years"}],
                "order_by": [{"column": "avg_tenure_years", "direction": "desc"}],
                "limit": 1000,
                "business_context": "Average tenure by department for full-time active employees"
            }

        # --- 4) Managers with more than N direct reports (default N=5)
        if re.search(r"managers?\s+.*(more than|>\s*\d+|over\s*\d+).*direct reports", q) or "direct reports" in q:
            threshold = self._extract_threshold(q, default=5)
            dims = ["managerId", "department"] if "department" in q else ["managerId"]
            return {
                "intent": self.INTENT_MGMT,
                "tables": ["workforce"],
                "select": dims,
                "group_by": dims,
                "aggregations": [{"function": "count", "column": "personIdExternal", "alias": "direct_reports"}],
                "having": [{"column": "direct_reports", "operator": ">", "value": threshold}],
                "order_by": [{"column": "direct_reports", "direction": "desc"}],
                "limit": 1000,
                "business_context": f"Managers with >{threshold} direct reports" + (" by department" if "department" in q else "")
            }

        # --- 5) Active employees in Technology with primary assignments
        if "active" in q and "primary assignment" in q and ("technology" in q or "tech" in q):
            return {
                "intent": self.INTENT_CROSSTABLE,
                "tables": ["workforce", "assignment"],
                "joins": [{"left_table": "workforce", "right_table": "assignment", "on": "personIdExternal"}],
                "where": [
                    {"table": "workforce", "column": "status", "operator": "=", "value": "A"},
                    {"table": "workforce", "column": "division", "operator": "=", "value": "Technology"},
                    {"table": "assignment", "column": "isPrimary", "operator": "=", "value": True},
                    {"table": "assignment", "column": "assignmentStatus", "operator": "=", "value": "A"},
                ],
                "select": ["personIdExternal", "division", "department", "assignmentType"],
                "limit": 1000,
                "business_context": "Active workforce in Technology with primary active assignments"
            }

        # --- default: unsupported
        return {
            "intent": "unknown",
            "error": ("Unsupported or ambiguous HR query. Try one of: "
                      "'Show headcount by division and employment status', "
                      "'Find employees hired in the last 6 months with active assignments', "
                      "'Average tenure by department for full-time employees', "
                      "'List managers with more than 5 direct reports'.")
        }

    # -----------------------
    # Helpers
    # -----------------------
    def _detect_dims(self, q: str, default: List[str]) -> List[str]:
        dims = []
        if "division" in q:   dims.append("division")
        if "department" in q: dims.append("department")
        if "location" in q:   dims.append("location")
        return dims or default

    def _mentions_full_time(self, q: str) -> bool:
        return any(x in q for x in ["full-time","full time","ft"])

    def _extract_threshold(self, q: str, default: int = 5) -> int:
        m = re.search(r"(?:more than|over|>\s*)(\d+)", q)
        if m:
            try:
                return int(m.group(1))
            except:
                return default
        return default