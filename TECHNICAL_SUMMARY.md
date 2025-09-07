---

#  `TECHNICAL_SUMMARY.md`

```markdown
# Technical Summary – Workforce IDQS (HR Analytics Prototype)

## HR Domain-Specific Architecture Decisions
- **Metadata-driven**: Uses CSN JSON catalogs to define canonical column names, join keys, and HR glossary terms (e.g., headcount, tenure).
- **Glossary Guardrails**: Every NL query is screened against glossary and schema to ensure only HR-related terms are executed.
- **Explainable Plans**: Queries are always translated into structured plan JSONs with clear `intent`, `tables`, `where`, `group_by`, and `aggregations`.
- **Deterministic Overrides**: Common HR queries (e.g., “headcount by division & status”) are recognized directly, ensuring predictable outputs.

---

## SuccessFactors Integration Challenges
- **Multiple Assignments**: Employees often have multiple active assignments in SuccessFactors; enforcing “primary assignment” rules requires additional logic.
- **Data Latency**: SuccessFactors extracts may lag; the system must handle late-arriving data gracefully.

---

## Business Rule Implementation Approach
- **Active Employees**: Enforced by `workforce.status = 'A'`.
- **Active Assignments**: Enforced by `assignment.assignmentStatus = 'A'`.
- **Primary Assignment**: Enforced by `assignment.isPrimary = true`.
- **Full-Time Employees**: Filtered by `assignmentType ∈ {FULL-TIME, FT}`.
- **Tenure Calculation**: Implemented as `DATEDIFF(COALESCE(endDate, today), startDate) / 365.25`.

Rules are stored in the **CSN glossary** for reusability, not hard-coded in queries.

---

## Workforce Analytics Scaling Considerations
- **Data Volume**: Workforce + assignment datasets may reach millions of rows. Spark provides distributed processing.
- **Query Performance**:  
  - Partitioning by `startDate` improves date-based queries.  
  - Caching workforce table speeds up repeated joins.  
- **Deployment**: Prototype runs in Docker with Spark local mode.  
  In production, can scale to Spark-on-K8s or Dataproc with S3/GCS data.  
- **Extensibility**: New glossary terms or CSN updates can extend analytics without code changes.

---

##  Key Value
- **Metadata + glossary** make the system HR-aware.  
- **NLP with Hugging Face** ensures HR-specific natural language understanding.  
- **Spark backend** guarantees scalability.  
- **Graceful error handling** keeps queries safe and explainable.