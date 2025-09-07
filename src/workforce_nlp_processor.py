# src/workforce_nlp_processor.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Set
import os, re, requests, time
from datetime import date, timedelta


class WorkforceNLPProcessor:
    """
    HF-only NLP with glossary+schema validation.

    Flow:
      0) Build allowed vocabulary from:
         - meta['glossary'] keys (business terms)
         - canonical column names from workforce/assignment mapping
         - a small whitelist of HR terms (employees, headcount, tenure, etc.)
      1) Vocabulary screen: tokenize query -> remove stopwords -> any leftover non-allowed tokens?
         - If yes -> graceful failure (intent="unknown", explain unknown tokens)
      2) (Guard A) Explicit override for headcount+division+status phrasing
      3) Zero-shot classify against curated labels (one per required template + out_of_scope)
      4) Non-HR / low-confidence guard -> graceful failure
      5) Map label -> plan for Spark engine.

    Env:
      HF_TOKEN     (required)
      HF_ZS_MODEL  (optional, default: "facebook/bart-large-mnli")
      HF_MIN_SCORE (optional float, default: 0.35; consider 0.6 for stricter)
    """

    # Canonical engine intents
    INTENT_ORG        = "organizational_analysis"
    INTENT_WORKFORCE  = "workforce_analytics"
    INTENT_ASSIGNMENT = "assignment_analysis"
    INTENT_CROSS      = "cross_table_analysis"
    INTENT_MGMT       = "management_hierarchy"

    # Curated label set (exactly your HR requirements)
    LABELS: List[Dict[str, str]] = [
        {"key": "org_headcount_division_status",
         "text": "organizational: headcount by division and employment status"},
        {"key": "org_top5_departments_headcount",
         "text": "organizational: top 5 departments by headcount"},
        {"key": "wf_active_employee_count",
         "text": "workforce: active employee count"},
        {"key": "wf_avg_tenure_by_department_full_time",
         "text": "workforce: average tenure by department for full-time active employees"},
        {"key": "asg_active_by_type",
         "text": "assignment: active assignments by employment type"},
        {"key": "asg_multiple_active_assignments",
         "text": "assignment: employees with multiple active assignments"},
        {"key": "cross_hired_last_n_months_active_asg",
         "text": "cross-table: employees hired in the last N months with active assignments"},
        {"key": "mgmt_managers_gt_n_reports",
         "text": "management: managers with more than N direct reports"},
        {"key": "out_of_scope",
         "text": "not HR analytics / out of scope"},
    ]

    # Lightweight stopwords (keep it short; we only want to filter obvious function words)
    STOPWORDS: Set[str] = {
        "the","a","an","and","or","of","for","to","in","on","by","with","from","at","as","is","are","be","show","list",
        "give","me","what","which","than","more","over",">","between","per","each","all","into","that","those","these",
        "please","could","you","i","we","it","this","last","month","months","year","years"
    }

    # Small whitelist of HR domain words that users commonly include but may not be in columns
    HR_WHITELIST: Set[str] = {
        "employee","employees","workforce","assignment","assignments","employment","status","active","inactive",
        "terminated","headcount","tenure","manager","managers","report","reports","direct","primary","type",
        "full-time","part-time","contract","location","division","department","hours","standard","hired","hire","hiring",
        "engineering","finance","hr","marketing","operations","sales","technology","brand","legal","qa","devops","benefits",
        "recruiting","people","accounting","data","science","digital","supply","chain","facilities","tax","inside","enterprise",
        "software","fp&a","paygrade","flsastatus","jobcode","job","title"
    }

    def __init__(self, meta: Dict[str, Any]):
        self.meta = meta or {}
        self.hf_token = os.getenv("HF_TOKEN")
        self.model = os.getenv("HF_ZS_MODEL", "facebook/bart-large-mnli")
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else None
        try:
            self.min_score = float(os.getenv("HF_MIN_SCORE", "0.35"))
        except Exception:
            self.min_score = 0.35

        # Build allowed vocabulary once
        self.allowed_vocab = self._build_allowed_vocab(self.meta)

    # ---------------- public ----------------
    def parse(self, text: str) -> Dict[str, Any]:
        if not self.headers:
            return {"intent": "unknown", "error": "Hugging Face token missing. Set HF_TOKEN for the API container."}

        q_raw = (text or "").strip()
        q = q_raw.lower()
        if not q:
            return {"intent": "unknown", "error": "Empty query."}

        # 0) GLOSSARY/SCHEMA SCREEN — reject queries containing unknown domain terms
        unknown = self._unknown_domain_tokens(q)
        if unknown:
            # Fail gracefully and explain the unknown tokens (first 5 shown)
            return {
                "intent": "unknown",
                "error": "Query references terms not found in the HR glossary/schema.",
                "unknown_terms": sorted(list(unknown))[:5]
            }

        # 1) OVERRIDE: headcount + division/department + status → exact organizational template
        if ("headcount" in q) and ("division" in q or "department" in q) and ("status" in q or "employment status" in q):
            plan = self._plan_for_label("org_headcount_division_status", q_raw)
            plan["_nlp_label"] = "org_headcount_division_status (override)"
            plan["_nlp_score"] = None
            return plan

        # 2) HF zero-shot
        label_key, score, debug = self._hf_choose_label(q_raw)

        # 3) Confidence / non-HR guard (kept simple; most non-HR weeds out in step 0)
        if (not label_key) or (label_key == "out_of_scope") or (score is None or score < self.min_score):
            return {"intent": "unknown", "error": "Unable to classify query confidently.", "debug": {"score": score, **(debug or {})}}

        # 4) Map label -> plan
        plan = self._plan_for_label(label_key, q_raw)
        plan["_nlp_label"] = label_key
        plan["_nlp_score"] = score
        return plan

    # ---------------- glossary/schema vocabulary ----------------
    def _build_allowed_vocab(self, meta: Dict[str, Any]) -> Set[str]:
        vocab: Set[str] = set()

        # From glossary keys
        gl = (meta or {}).get("glossary") or {}
        for term in gl.keys():
            for token in self._simple_tokens(term):
                vocab.add(token)

        # From table column mappings (canonical column names)
        tables = (meta or {}).get("tables") or {}
        for tname, tinfo in tables.items():
            mapping = (tinfo or {}).get("mapping") or {}
            for canon_col in mapping.keys():
                for token in self._simple_tokens(canon_col):
                    vocab.add(token)

        # Add core HR whitelist words
        vocab |= {w.lower() for w in self.HR_WHITELIST}

        # Common org dimension names (often appear with punctuation)
        vocab |= {"division","department","location","status","employment","employmentstatus",
                  "manager","managerid","tenure","headcount","assignment","assignments","assignmentstatus","isprimary"}

        return vocab

    def _unknown_domain_tokens(self, q: str) -> Set[str]:
        # Tokenize, filter stopwords and purely numeric
        tokens = [t for t in self._simple_tokens(q) if t not in self.STOPWORDS and not t.isdigit()]
        # Keep only "contenty" tokens (length >= 3)
        tokens = [t for t in tokens if len(t) >= 3]
        # Allowed?
        unknown = {t for t in tokens if t not in self.allowed_vocab}
        return unknown

    def _simple_tokens(self, s: str) -> List[str]:
        s = s.lower()
        # normalize some HR punctuations (e.g., full-time)
        s = s.replace("&", " ").replace("-", " ").replace("/", " ")
        return [t for t in re.split(r"[^a-z0-9]+", s) if t]

    # ---------------- HF call ----------------
    def _hf_choose_label(self, text: str) -> Tuple[Optional[str], Optional[float], Dict[str, Any]]:
        debug: Dict[str, Any] = {"model": self.model}
        if not self.headers:
            debug["reason"] = "missing_token"
            return None, None, debug

        candidate_texts = [l["text"] for l in self.LABELS]
        debug["candidates"] = candidate_texts

        payload = {
            "inputs": text,
            "parameters": {
                "candidate_labels": candidate_texts,
                "multi_label": False,
                "hypothesis_template": "This HR question is about {}."
            }
        }

        for attempt in range(4):
            try:
                r = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                debug["status_code"] = r.status_code
                if r.status_code in (429, 503):
                    debug["transient"] = r.status_code
                    time.sleep(1.2 * (attempt + 1))
                    continue
                if r.status_code != 200:
                    try:
                        debug["body"] = r.json()
                    except Exception:
                        debug["body"] = r.text[:800]
                    return None, None, debug

                data = r.json()
                if isinstance(data, list) and data:
                    data = data[0]

                labels = data.get("labels")
                scores = data.get("scores")
                if not labels or not scores or len(labels) != len(scores):
                    debug["body"] = data
                    return None, None, debug

                top_text = labels[0]
                top_score = scores[0]
                key = None
                for l in self.LABELS:
                    if l["text"] == top_text:
                        key = l["key"]
                        break
                return key, float(top_score), debug
            except Exception as e:
                debug["exception"] = str(e)
                if attempt == 3:
                    return None, None, debug
                time.sleep(1.0 + attempt)

        return None, None, debug

    # -------------- label -> plan --------------
    def _plan_for_label(self, key: str, q_raw: str) -> Dict[str, Any]:
        ql = q_raw.lower()

        if key == "org_headcount_division_status":
            return {
                "intent": self.INTENT_ORG,
                "tables": ["workforce"],
                "select": ["division", "status"],
                "group_by": ["division", "status"],
                "aggregations": [{"function": "count_distinct", "column": "personIdExternal", "alias": "headcount"}],
                "order_by": [{"column": "headcount", "direction": "desc"}],
                "limit": 1000,
                "glossary_refs": ["headcount"],
                "business_context": "Headcount grouped by division and employment status",
            }

        if key == "org_top5_departments_headcount":
            return {
                "intent": self.INTENT_ORG,
                "tables": ["workforce"],
                "select": ["department"],
                "group_by": ["department"],
                "aggregations": [{"function": "count_distinct", "column": "personIdExternal", "alias": "headcount"}],
                "order_by": [{"column": "headcount", "direction": "desc"}],
                "limit": 5,
                "glossary_refs": ["headcount"],
                "business_context": "Top 5 departments by headcount",
            }

        if key == "wf_active_employee_count":
            return {
                "intent": self.INTENT_WORKFORCE,
                "tables": ["workforce"],
                "where": [{"table": "workforce", "column": "status", "operator": "=", "value": "A"}],
                "select": [],
                "group_by": [],
                "aggregations": [{"function": "count_distinct", "column": "personIdExternal", "alias": "active_headcount"}],
                "order_by": [],
                "limit": 1,
                "glossary_refs": ["active_employees", "headcount"],
                "business_context": "Active employee count (status = 'A')",
            }

        if key == "wf_avg_tenure_by_department_full_time":
            return {
                "intent": self.INTENT_WORKFORCE,
                "tables": ["workforce", "assignment"],
                "joins": [{"left_table": "workforce", "right_table": "assignment", "on": "personIdExternal"}],
                "where": [
                    {"table": "assignment", "column": "assignmentType", "operator": "in",
                     "value": ["FULL-TIME", "FULL TIME", "FT"]},
                    {"table": "workforce", "column": "status", "operator": "=", "value": "A"},
                ],
                "select": ["department"],
                "group_by": ["department"],
                "aggregations": [{"function": "avg_tenure_years", "column": "startDate", "alias": "avg_tenure_years"}],
                "order_by": [{"column": "avg_tenure_years", "direction": "desc"}],
                "limit": 1000,
                "glossary_refs": ["tenure_years", "full_time", "active_employees"],
                "business_context": "Average tenure by department for full-time active employees",
            }

        if key == "asg_active_by_type":
            return {
                "intent": self.INTENT_ASSIGNMENT,
                "tables": ["assignment"],
                "where": [{"table": "assignment", "column": "assignmentStatus", "operator": "=", "value": "A"}],
                "select": ["assignmentType"],
                "group_by": ["assignmentType"],
                "aggregations": [{"function": "count", "column": "assignmentType", "alias": "active_assignments"}],
                "order_by": [{"column": "active_assignments", "direction": "desc"}],
                "limit": 1000,
                "glossary_refs": ["active_assignment"],
                "business_context": "Active assignments by employment type",
            }

        if key == "asg_multiple_active_assignments":
            return {
                "intent": self.INTENT_ASSIGNMENT,
                "tables": ["assignment"],
                "where": [{"table": "assignment", "column": "assignmentStatus", "operator": "=", "value": "A"}],
                "select": ["personIdExternal"],
                "group_by": ["personIdExternal"],
                "aggregations": [{"function": "count", "column": "assignmentIdExternal", "alias": "active_assignments"}],
                "having": [{"column": "active_assignments", "operator": ">", "value": 1}],
                "order_by": [{"column": "active_assignments", "direction": "desc"}],
                "limit": 1000,
                "glossary_refs": ["active_assignment"],
                "business_context": "Employees with multiple active assignments",
            }

        if key == "cross_hired_last_n_months_active_asg":
            months = self._extract_first_int(ql, default=6)
            since = (date.today() - timedelta(days=30 * months)).isoformat()
            return {
                "intent": self.INTENT_CROSS,
                "tables": ["workforce", "assignment"],
                "joins": [{"left_table": "workforce", "right_table": "assignment", "on": "personIdExternal"}],
                "where": [
                    {"table": "workforce", "column": "startDate", "operator": ">=", "value": since},
                    {"table": "assignment", "column": "assignmentStatus", "operator": "=", "value": "A"},
                ],
                "select": ["personIdExternal", "startDate", "division", "department", "assignmentType", "assignmentStatus"],
                "order_by": [{"column": "startDate", "direction": "desc"}],
                "limit": 1000,
                "glossary_refs": ["active_assignment"],
                "business_context": f"Employees hired since {since} with active assignments",
            }

        if key == "mgmt_managers_gt_n_reports":
            threshold = self._extract_threshold(ql, default=5)
            return {
                "intent": self.INTENT_MGMT,
                "tables": ["workforce"],
                "select": ["managerId"],
                "group_by": ["managerId"],
                "aggregations": [{"function": "count", "column": "personIdExternal", "alias": "direct_reports"}],
                "having": [{"column": "direct_reports", "operator": ">", "value": threshold}],
                "order_by": [{"column": "direct_reports", "direction": "desc"}],
                "limit": 1000,
                "glossary_refs": [],
                "business_context": f"Managers with more than {threshold} direct reports",
            }

        return {"intent": "unknown", "error": f"No plan mapping for label '{key}'."}

    # -------------- helpers --------------
    def _extract_first_int(self, q: str, default: int) -> int:
        m = re.search(r"\b(\d+)\b", q)
        if not m:
            return default
        try:
            return int(m.group(1))
        except Exception:
            return default

    def _extract_threshold(self, q: str, default: int = 5) -> int:
        m = re.search(r"(?:more than|over|>\s*)(\d+)", q)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return default
        return self._extract_first_int(q, default)
