from __future__ import annotations
from typing import Dict, Any, List, Tuple
from pyspark.sql import SparkSession, DataFrame, functions as F, types as T

# ---------- small utilities ----------
def _to_date(col):
    return F.to_date(F.col(col).cast("string"))

def _ensure_cols(df: DataFrame, required: List[str]) -> DataFrame:
    for c in required:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("string"))
    return df

def _agg_expr(func: str, col: str) -> F.Column:
    func = func.lower()
    if func == "count":
        return F.count(F.col(col))
    if func == "count_distinct":
        return F.countDistinct(F.col(col))
    if func == "avg":
        return F.avg(F.col(col))
    raise ValueError(f"Unsupported aggregation function: {func}")

# ---------- canonicalization ----------
def _canon_workforce(df: DataFrame, m: Dict[str,str]) -> DataFrame:
    out = df
    # required core columns (rename if mapping differs)
    if m.get("personIdExternal") != "personIdExternal":
        out = out.withColumnRenamed(m["personIdExternal"], "personIdExternal")
    if m.get("startDate"):
        out = out.withColumn("startDate", _to_date(m["startDate"]))
    else:
        out = out.withColumn("startDate", F.lit(None).cast("date"))
    # optional
    for k in ["status","division","department","location","managerId","endDate"]:
        phys = m.get(k)
        if phys and phys in out.columns and phys != k:
            out = out.withColumnRenamed(phys, k)
        elif k not in out.columns:
            out = out.withColumn(k, F.lit(None))
    # cast endDate to date if present
    out = out.withColumn("endDate", _to_date("endDate"))
    return out

def _canon_assignment(df: DataFrame, m: Dict[str,str]) -> DataFrame:
    out = df

    # Non-overlapping, keep canonical names
    for k in ["personIdExternal","assignmentStatus","isPrimary","assignmentType","standardHours"]:
        phys = m.get(k)
        if phys and phys in out.columns and phys != k:
            out = out.withColumnRenamed(phys, k)
        elif k not in out.columns:
            if k == "standardHours":
                out = out.withColumn(k, F.lit(None).cast("double"))
            elif k == "isPrimary":
                out = out.withColumn(k, F.lit(None).cast("boolean"))
            else:
                out = out.withColumn(k, F.lit(None).cast("string"))

    # Overlapping organizational/date columns → prefix with asg*
    overlap_map = {
        "division":     "asgDivision",
        "department":   "asgDepartment",
        "location":     "asgLocation",
        "startDate":    "asgStartDate",
        "endDate":      "asgEndDate",
    }
    for logical, prefixed in overlap_map.items():
        phys = m.get(logical)
        if phys and phys in out.columns:
            if phys != prefixed:
                out = out.withColumnRenamed(phys, prefixed)
        elif prefixed not in out.columns:
            # create missing columns with appropriate types
            if logical in ("startDate","endDate"):
                out = out.withColumn(prefixed, F.lit(None).cast("date"))
            else:
                out = out.withColumn(prefixed, F.lit(None).cast("string"))

    # Ensure proper types for dates
    out = out.withColumn("asgStartDate", _to_date("asgStartDate"))
    out = out.withColumn("asgEndDate", _to_date("asgEndDate"))

    return out

# ---------- glossary application ----------
def _apply_glossary_filters(df_map: Dict[str, DataFrame], where_terms: List[Dict[str, Any]]) -> Dict[str, DataFrame]:
    """
    where_terms is already expanded to explicit conditions; this function just applies them.
    """
    for cond in where_terms or []:
        table = cond["table"]
        col   = cond["column"]
        op    = cond["operator"]
        val   = cond.get("value")
        c = F.col(col)
        if op == "=":
            df_map[table] = df_map[table].filter(c == val)
        elif op.lower() == "in":
            df_map[table] = df_map[table].filter(c.isin(val))
        elif op == ">=":
            df_map[table] = df_map[table].filter(c >= F.lit(val))
        elif op == "<=":
            df_map[table] = df_map[table].filter(c <= F.lit(val))
        elif op == ">":
            df_map[table] = df_map[table].filter(c > F.lit(val))
        elif op == "<":
            df_map[table] = df_map[table].filter(c < F.lit(val))
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return df_map

def _expand_glossary(meta: Dict[str, Any], plan: Dict[str, Any]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]], List[str]]:
    """
    Use glossary entries to expand abstract terms into concrete plan pieces.
    Returns:
      - extra_where: flattened where conditions from glossary_refs
      - extra_aggs: aggregations substituted via glossary (e.g., headcount)
      - used_terms: resolved glossary keys
    """
    glossary_refs = plan.get("glossary_refs") or []
    glossary = meta.get("business_terms") or {}
    used: List[str] = []
    extra_where: List[Dict[str,Any]] = []
    extra_aggs: List[Dict[str,Any]] = []

    for ref in glossary_refs:
        g = glossary.get(ref)
        if not g: 
            continue
        used.append(ref)
        # where expansion
        if "where" in g:
            extra_where.extend(g["where"])
        # aggregation expansion
        if "aggregation" in g:
            extra_aggs.append(g["aggregation"])
        # (tenure_years handled in agg function below)
    return extra_where, extra_aggs, used

def _tenure_years_expr(start_col: str, end_col: str | None = None) -> F.Column:
    # tenure in years using current_date if endDate is null
    start = F.col(start_col)
    end = F.coalesce(F.col(end_col) if end_col else F.lit(None).cast("date"), F.current_date())
    return (F.datediff(end, start) / F.lit(365.25)).cast("double")

# ---------- main execute ----------
def execute_plan(spark: SparkSession, meta: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    # Load
    paths = meta["data_paths"]
    df_wf_raw = spark.read.parquet(paths["workforce"])
    df_asg_raw = spark.read.parquet(paths["assignment"])

    wf_map = meta["tables"]["workforce"]["mapping"]
    asg_map = meta["tables"]["assignment"]["mapping"]
    wf = _canon_workforce(df_wf_raw, wf_map)
    asg = _canon_assignment(df_asg_raw, asg_map)

    tables = plan.get("tables") or []
    df_map: Dict[str, DataFrame] = {}
    if "workforce" in tables:
        df_map["workforce"] = wf
    if "assignment" in tables:
        df_map["assignment"] = asg

    # Expand glossary references to concrete 'where' and 'aggregations'
    extra_where, extra_aggs, used_terms = _expand_glossary(meta, plan)

    # Apply per-table WHEREs (plan + glossary)
    where_all = (plan.get("where") or []) + extra_where
    if where_all:
        df_map = _apply_glossary_filters(df_map, where_all)

    # JOIN if needed
    if len(tables) > 1:
        join_key = meta.get("join_key") or "personIdExternal"
        result = df_map["workforce"].join(df_map["assignment"], on=join_key, how="inner")
    else:
        # choose the only table result
        key = tables[0]
        result = df_map[key]

    # SELECT / GROUP BY / AGG
    select_cols = plan.get("select") or []
    group_by = plan.get("group_by") or []
    agg_specs = (plan.get("aggregations") or []) + extra_aggs

    # Special handler for avg_tenure_years aggregation (via glossary 'tenure_years')
    # Convert any {"function":"avg_tenure_years", "column":"startDate", "alias":"..."} to concrete expr.
    agg_exprs: List[F.Column] = []
    out_cols: List[F.Column] = [F.col(c) for c in select_cols] if select_cols else []

    for spec in agg_specs:
        fn = spec["function"].lower()
        alias = spec.get("alias") or spec["function"]
        if fn == "avg_tenure_years":
            # use workforce.startDate and (optional) workforce.endDate
            col_start = "startDate"  # already canonical
            col_end = "endDate"      # may be null
            expr = F.avg(_tenure_years_expr(col_start, col_end)).alias(alias)
            agg_exprs.append(expr)
        else:
            expr = _agg_expr(spec["function"], spec["column"]).alias(alias)
            agg_exprs.append(expr)

    if group_by:
        result = result.groupBy(*group_by).agg(*agg_exprs)
    elif agg_exprs:
        # aggregation without groupBy -> aggregate over entire set
        result = result.agg(*agg_exprs)
    elif select_cols:
        result = result.select(*out_cols)

    # HAVING
    for hv in plan.get("having") or []:
        col = hv["column"]
        op = hv["operator"]
        val = hv["value"]
        c = F.col(col)
        if op == ">":
            result = result.filter(c > F.lit(val))
        elif op == ">=":
            result = result.filter(c >= F.lit(val))
        elif op == "<":
            result = result.filter(c < F.lit(val))
        elif op == "<=":
            result = result.filter(c <= F.lit(val))
        elif op == "=":
            result = result.filter(c == F.lit(val))

    # ORDER BY
    if plan.get("order_by"):
        order_cols = []
        for ob in plan["order_by"]:
            c = F.col(ob["column"])
            order_cols.append(c.desc() if ob.get("direction","desc").lower()=="desc" else c.asc())
        result = result.orderBy(*order_cols)

    # LIMIT
    if plan.get("limit"):
        result = result.limit(int(plan["limit"]))

    # Collect
    rows = [ {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k,v in r.asDict().items()} for r in result.collect() ]

    return {
        "intent": plan.get("intent"),
        "business_context": plan.get("business_context"),
        "glossary_refs": list(dict.fromkeys(used_terms)),  # unique preserve order
        "count": len(rows),
        "rows": rows
    }
