# ==============================================================================
# Unified p53 Audit Engine
# V1.4 (Combines Verification V20.4 and Narrative V2.4 with Patches)
# ==============================================================================
#
# This single script performs the complete end-to-end audit process for the
# p53 simulation study.
#
# WHAT'S NEW (v1.4 Full Reporting):
# - As requested, the engine now saves the detailed intermediate report,
#   `Comprehensive_Verification_Report.json`, to disk.
# - This ensures all three key artifacts are generated for maximum clarity and
#   auditability:
#   1. Comprehensive_Verification_Report.json (detailed, machine-readable)
#   2. Biophysics_Final_Report.md (final, human-readable)
#   3. Biophysics_Final_Report_Summary.json (final, summary metrics)
#
# WORKFLOW:
# 1. **Initialize AI Connection (Mandatory):** Stop if the Gemini model cannot be initialized.
# 2. Loads the `findings.json` manifest and locates all data artifacts.
# 3. Analyzes each artifact, performing local calculations and using the Gemini API.
# 4. Deterministically evaluates the claims and constraints for each run.
# 5. **Saves the `Comprehensive_Verification_Report.json`.**
# 6. Processes the comprehensive report to generate the final `Biophysics_Final_Report.md`
#    and `Biophysics_Final_Report_Summary.json`.
#
# ==============================================================================

import os, sys, re, json, time, hashlib, logging, random, datetime, statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from PIL import Image
import requests
import io

# (Colab helpers are optional; guarded)
try:
    from google.colab import drive, files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Gemini
try:
    import google.generativeai as genai
    import google.api_core.exceptions
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --------------------------- Logging -----------------------------------------
logger = logging.getLogger("unified_p53_engine")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)
fh = logging.FileHandler("unified_audit_engine.log")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)

# --------------------------- Constants & Prompts ------------------------------
MAX_PAYLOAD_SIZE_BYTES = 1_500_000
PNG_MAX_SIDE = 1600
MODEL_NAME = "gemini-1.5-pro-latest"

PROMPT_LIBRARY = {
    "md": """ROLE: Senior biophysicist.
TASK: Extract ONLY the reported numbers from the markdown report below.

REPORT_MD:
<<<
{FILE_CONTENT}
>>>

RULES:
- Use ONLY explicit numbers, do not infer.
- If multiple values, use the 'Top 10 runs' table (lowest RMSD row).
- RMSD thresholds: <=5 Å near_native; 5–12 Å intermediate; >=20 Å misfolded.

SCHEMA:
{{
  "runs_count": int | null,
  "failures": int | null,
  "best_final_RMSD_A": float | null,
  "best_final_Rg_A": float | null,
  "classification": "near_native" | "intermediate" | "misfolded" | "unknown",
  "table_row_source": "string"
}}
OUTPUT: JSON only.
""",
    "raw_csv": """ROLE: Data auditor.
TASK: Check CSV snippet consistency vs LOCAL_SUMMARY.

CSV_CONTENT:
<<<
{FILE_CONTENT}
>>>

LOCAL_SUMMARY:
{LOCAL_SUMMARY}

SCHEMA:
{{
  "consistency_check": "consistent"|"inconsistent"|"cannot_determine",
  "notes": "short explanation"
}}
OUTPUT: JSON only.
""",
    "by_param_csv": """ROLE: Computational chemist.
TASK: Review the parameter-sweep CSV snippet and the local summary to assess parameter sensitivity.

CSV_CONTENT:
<<<
{FILE_CONTENT}
>>>

LOCAL_SUMMARY:
{LOCAL_SUMMARY}

GUIDE:
- If best_final_RMSD varies < 0.5 Å across parameter sets → "low"
- 0.5–2.0 Å → "moderate"
- > 2.0 Å → "high"
- If you cannot tell from snippet, return "unknown".

SCHEMA:
{{
  "n_rows": int | null,
  "distinct_param_sets": int | null,
  "sensitivity": "low" | "moderate" | "high" | "unknown",
  "notes": "brief rationale"
}}
OUTPUT: JSON only.
""",
    "diagnostics_csv": """ROLE: Quantitative analyst.
TASK: Compare timeseries CSV snippet vs LOCAL_SUMMARY.

CSV_CONTENT:
<<<
{FILE_CONTENT}
>>>

LOCAL_SUMMARY:
{LOCAL_SUMMARY}

SCHEMA:
{{
  "consistency_check": "consistent"|"inconsistent"|"cannot_determine",
  "notes": "short explanation"
}}
OUTPUT: JSON only.
""",
    "pdb": """ROLE: Structural biologist.
TASK: Assess PDB snippet (focus on CA atoms if present).

PDB_SNIPPET:
<<<
{FILE_CONTENT}
>>>

SCHEMA:
{{
  "frames_sampled":[int,...]|null,
  "qualitative_compaction":"yes"|"no"|"uncertain",
  "notes":"string"
}}
OUTPUT: JSON only.
""",
    "comprehensive_png": """ROLE: Structural biologist.
TASK: Assess comprehensive PNG report (RMSD, Rg, snapshot).

SCHEMA:
{{
  "time_series_description":"string",
  "final_structure_description":"string",
  "overall_conclusion":"string"
}}
OUTPUT: JSON only.
""",
    "diagnostics_png": """ROLE: Analyst.
TASK: Assess diagnostics PNG with RMSD/Rg/H-bonds/salt bridges).

SCHEMA:
{{
  "rmsd_trend":"string",
  "rg_trend":"string",
  "interaction_trends":"string",
  "mechanical_interpretation":"string"
}}
OUTPUT: JSON only.
"""
}

# ==============================================================================
# SECTION 1: VERIFICATION ENGINE LOGIC (STAGE 2/3)
# ==============================================================================

# --------------------------- Helpers: safe text / CSV ------------------------
def safe_text_snippet(s: str, max_bytes: int, head: int = 200, tail: int = 200) -> str:
    b = s.encode("utf-8", errors="ignore")
    if len(b) <= max_bytes:
        return s
    lines = s.splitlines(True)
    return "".join(lines[:head]) + "\n...TRUNCATED...\n" + "".join(lines[-tail:])

def csv_focus_minimal(df: pd.DataFrame, byte_limit: int = MAX_PAYLOAD_SIZE_BYTES) -> str:
    parts = [",".join(map(str, df.columns)) + "\n"]
    if len(df) > 0:
        parts.append(df.head(30).to_csv(index=False, header=False))
        parts.append("\n...rows truncated...\n")
        parts.append(df.tail(30).to_csv(index=False, header=False))
    return safe_text_snippet("".join(parts), byte_limit)

def _read_csv_guard(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.warning(f"    - Could not read CSV at {path}: {e}")
        return None

def summarize_raw_csv_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    out = {"source": "raw_csv"}
    cols_norm = {c.lower().strip().replace('_a','').replace(' ',''): c for c in df.columns}
    rmsd_key = cols_norm.get("finalrmsd") or cols_norm.get("bestrmsd") or cols_norm.get("finalrmsdå")
    rg_key   = cols_norm.get("finalrg") or cols_norm.get("bestrg") or cols_norm.get("finalrgå")
    if rmsd_key:
        s = pd.to_numeric(df[rmsd_key], errors="coerce").dropna()
        if not s.empty:
            out["best_final_RMSD_A"] = round(float(s.min()), 5)
    if rg_key:
        s = pd.to_numeric(df[rg_key], errors="coerce").dropna()
        if not s.empty:
            out["best_final_Rg_A"] = round(float(s.min()), 5)
    out["runs_count"] = int(len(df))
    return out

def summarize_timeseries_csv_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    out = {"source": "diagnostics_csv"}
    cols = {c.lower().strip(): c for c in df.columns}
    def find_col(cands):
        for k in cands:
            if k in cols:
                return cols[k]
        return None
    rmsd_col = find_col(["rmsd", "rmsd_a", "rmsd(å)", "rmsd(aa)", "final_rmsd"])
    rg_col   = find_col(["rg", "rg_a", "rg(å)", "final_rg"])
    sb_col   = find_col(["salt_bridges", "salt-bridges", "n_salt_bridges", "saltbridges", "n_salt-bridges"])
    if rmsd_col:
        s = pd.to_numeric(df[rmsd_col], errors="coerce").ffill().bfill().dropna()
        if len(s) > 0:
            out["final_RMSD_A"] = round(float(s.iloc[-1]), 5)
            tail = s.iloc[max(0, int(len(s) * 0.9)):]
            out["stabilized"] = bool(tail.std() <= 1.0)
    if rg_col:
        g = pd.to_numeric(df[rg_col], errors="coerce").ffill().bfill().dropna()
        if len(g) > 0:
            out["final_Rg_A"] = round(float(g.iloc[-1]), 5)
    if sb_col:
        b = pd.to_numeric(df[sb_col], errors="coerce").ffill().bfill().dropna()
        if len(b) > 0:
            out["median_salt_bridges"] = float(b.median())
    return out

def summarize_by_param_csv_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    out = {"source": "by_param_csv"}
    out["n_rows"] = int(len(df))
    outputs = set([c for c in df.columns for k in ["rmsd", "rg"] if k in c.lower()])
    param_cols = [c for c in df.columns if c not in outputs]
    if param_cols:
        out["distinct_param_sets"] = int(df[param_cols].drop_duplicates().shape[0])
    rmsd_cols = [c for c in df.columns if "rmsd" in c.lower()]
    if rmsd_cols:
        s = pd.to_numeric(df[rmsd_cols[0]], errors="coerce").dropna()
        if not s.empty:
            out["best_final_RMSD_A"] = round(float(s.min()), 2)
            out["spread_RMSD_A"] = round(float(s.max() - s.min()), 2)
    return out

def pdb_stratified_snippet(text: str, max_frames: int = 5) -> Tuple[str, List[int]]:
    lines = text.splitlines(True)
    header = [ln for ln in lines if not ln.startswith(("MODEL", "ATOM", "HETATM"))]
    models, cur, in_model = [], [], False
    frames = []
    for ln in lines:
        if ln.startswith("MODEL"):
            in_model = True; cur = [ln]
        elif ln.startswith("ENDMDL"):
            cur.append(ln); models.append(cur); in_model = False
        elif in_model:
            cur.append(ln)
    if not models:
        return "".join(lines[:8000]), []
    num = len(models)
    if max_frames < 2:
        idx = [0]
    else:
        idx = sorted(set([0, num - 1] + [int(num * (i / (max_frames - 1))) for i in range(1, max_frames - 1)]))
    frames = [i + 1 for i in idx]
    out = header + [f"REMARK SAMPLED FRAMES: {frames}\n"]
    for i in idx:
        m = models[i]
        ca = [ln for ln in m if " CA " in ln]
        out.extend([m[0]] + ca[:2000] + [m[-1]])
    snippet = "".join(out)
    if len(snippet.encode("utf-8", errors="ignore")) > MAX_PAYLOAD_SIZE_BYTES:
        snippet = safe_text_snippet(snippet, MAX_PAYLOAD_SIZE_BYTES)
    return snippet, frames

def short_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()[:12]

# --------------------------- Gemini call with repair --------------------------
def call_gemini_with_retry(model, payload, max_retries=3, timeout=300):
    # This function now assumes `model` is a valid, initialized object.
    for attempt in range(max_retries):
        try:
            cfg = {"temperature": 0.0, "response_mime_type": "application/json"}
            logger.info(f"    --> Attempting Gemini API call (Attempt {attempt + 1}/{max_retries})...")
            resp = model.generate_content(payload, generation_config=cfg, request_options={"timeout": timeout})
            logger.info(f"    --> API call returned.")
            txt = resp.text.strip().replace("```json","").replace("```","")
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                logger.warning("    --> Invalid JSON detected. Attempting one repair pass.")
                repair = f"Fix JSON only:\n{txt}"
                r2 = model.generate_content(repair, generation_config=cfg)
                return json.loads(r2.text.strip().replace("```json","").replace("```",""))
        except (google.api_core.exceptions.ResourceExhausted,
                google.api_core.exceptions.ServiceUnavailable,
                google.api_core.exceptions.DeadlineExceeded,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            wait_time = (2**attempt) + random.uniform(0, 1)
            logger.warning(f"    --> Retriable API error {attempt+1}/{max_retries}: {e}. Retry in {wait_time:.2f}s.")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"    --> Non-retriable API error: {e}", exc_info=True)
            return {"error": str(e)}
    return {"error": "API failed after all retries"}

# --------------------------- Role & Metric Plumbing ---------------------------
ROLE_TO_PROMPT = {
    "md_report": "md", "raw_csv": "raw_csv", "by_param_csv": "by_param_csv",
    "diagnostics_csv": "diagnostics_csv", "comprehensive_png": "comprehensive_png",
    "diagnostics_png": "diagnostics_png", "trajectory_pdb": "pdb",
}

def get_metric_from_sources(metric: str, nums: Dict, vocab: Dict) -> Tuple[Optional[Any], Optional[str]]:
    preferred = vocab.get(metric, {}).get("preferred_sources", [])
    key_map = {
        "md": {"best_final_RMSD_A": "md_best_rmsd", "best_final_Rg_A": "md_best_rg", "runs_count": "md_runs", "failures": "md_failures"},
        "raw_csv": {"best_final_RMSD_A": "raw_best_rmsd", "best_final_Rg_A": "raw_best_rg", "runs_count": "raw_runs"},
        "diagnostics_csv": {"best_final_RMSD_A": "ts_final_rmsd", "best_final_Rg_A": "ts_final_rg", "median_salt_bridges": "ts_median_salt_bridges"},
    }
    for source in preferred:
        key = key_map.get(source, {}).get(metric)
        if key and nums.get(key) is not None:
            return nums[key], source
    return None, None

def apply_op(lhs, op, rhs, tol=0.0):
    if lhs is None or rhs is None: return None
    try:
        lhs, rhs, tol = float(lhs), float(rhs), float(tol or 0.0)
    except (ValueError, TypeError):
        return None
    ops = {"<=": lambda a, b: a <= b + tol, ">=": lambda a, b: a >= b - tol, "==": lambda a, b: abs(a - b) <= tol,
           "<": lambda a, b: a < b + tol, ">": lambda a, b: a > b - tol, "!=": lambda a, b: abs(a - b) > tol}
    return ops.get(op)(lhs, rhs) if op in ops else None

def evaluate_constraints(constraints: List[Dict], nums: Dict, vocab: Dict) -> Tuple[str, List[Dict]]:
    if not constraints: return "NO_EVALUABLE_CONSTRAINTS", []
    results, any_dev, any_eval = [], False, False
    for const in constraints:
        metric, op, value, tol = const.get("metric"), const.get("op"), const.get("value"), const.get("tolerance", 0.0)
        actual, source = get_metric_from_sources(metric, nums, vocab)
        if actual is None:
            results.append({"constraint": const, "status": "NOT_EVALUATED", "reason": f"Metric '{metric}' not found."})
            continue
        any_eval = True
        ok = apply_op(actual, op, value, tol)
        delta = float(actual) - float(value) if actual is not None and value is not None else None
        results.append({"constraint": const, "status": "CONFIRMED" if ok else "DEVIATION", "actual_value": actual, "source": source, "delta": delta})
        if ok is False: any_dev = True
    if not any_eval: return "NO_EVALUABLE_CONSTRAINTS", results
    return "CONFIRMED_WITH_DEVIATIONS" if any_dev else "ALL_CONFIRMED", results

def precheck_and_cross_validate(analyses):
    nums = {"md_best_rmsd": None, "md_best_rg": None, "raw_best_rmsd": None, "raw_best_rg": None, "md_runs": None,
            "raw_runs": None, "md_failures": None, "diagnostics_stable": None, "ts_final_rmsd": None, "ts_final_rg": None,
            "ts_median_salt_bridges": None}
    for a in analyses:
        role, an, loc = a.get("role_key"), a.get("analysis", {}), a.get("local_summary", {}) or {}
        if not isinstance(an, dict): continue
        if role == "md": nums.update({"md_best_rmsd": an.get("best_final_RMSD_A"), "md_best_rg": an.get("best_final_Rg_A"), "md_runs": an.get("runs_count"), "md_failures": an.get("failures")})
        elif role == "raw_csv": nums.update({"raw_best_rmsd": loc.get("best_final_RMSD_A"), "raw_best_rg": loc.get("best_final_Rg_A"), "raw_runs": loc.get("runs_count")})
        elif role == "diagnostics_csv": nums.update({"ts_final_rmsd": loc.get("final_RMSD_A"), "ts_final_rg": loc.get("final_Rg_A"), "diagnostics_stable": bool(loc.get("stabilized")), "ts_median_salt_bridges": loc.get("median_salt_bridges")})
    def status(md, raw, tol=0.5):
        if md is None or raw is None: return {"md": md, "raw": raw, "status": "MISSING"}
        return {"md": md, "raw": raw, "status": "MATCH" if abs(md - raw) <= tol else "MISMATCH"}
    checks = {"best_RMSD_md_vs_raw": status(nums["md_best_rmsd"], nums["raw_best_rmsd"], 0.5),
              "best_Rg_md_vs_raw": status(nums["md_best_rg"], nums["raw_best_rg"], 0.25),
              "runs_count_md_vs_raw": status(nums["md_runs"], nums["raw_runs"], 0.0)}
    discrepancies = [{"metric": k, "details": f"MD={v['md']}, RAW={v['raw']}"} for k, v in checks.items() if v["status"] == "MISMATCH"]
    return nums, checks, discrepancies

# --------------------------- IO Helpers --------------------------------------
def _resolve_path(root_dir: Path, dataset_root: Optional[str], source_folder: str, artifact_path: str) -> Path:
    base_path = Path(root_dir)
    run_path = base_path / source_folder
    artifact_full_path = run_path / artifact_path
    return artifact_full_path.resolve()

def _open_artifact(path: Path) -> Tuple[Optional[bytes], Optional[str]]:
    if not path.exists(): return None, None
    if str(path).lower().endswith((".png", ".jpg", ".jpeg")):
        with open(path, "rb") as f: return f.read(), None
    try:
        return None, path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        with open(path, "rb") as f: return f.read(), None

def _prepare_png_payload(raw_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.thumbnail((PNG_MAX_SIDE, PNG_MAX_SIDE))
        return img
    except Exception as e:
        logger.warning(f"    - PNG prepare failed: {e}")
        return raw_bytes

# --------------------------- Hybrid Confidence Logic --------------------------
def _llm_confidence_assess(model, claim_text, constraint_summary, nums, checks, analyses):
    payload = f"""
ROLE: Senior structural biophysicist. TASK: Assign a confidence level (HIGH | MEDIUM | LOW).
CLAIM: {claim_text}
CONSTRAINT_SUMMARY: {constraint_summary}
KEY_NUMBERS: {json.dumps(nums, indent=2)}
CROSS_CHECKS: {json.dumps(checks, indent=2)}
EVIDENCE_SUMMARY: {json.dumps([{"file": a.get("file"), "role": a.get("role_key")} for a in analyses], indent=2)}
POLICY: HIGH for consistent evidence & ALL_CONFIRMED. MEDIUM for minor gaps or DEVIATIONS. LOW for NO_EVALUABLE_CONSTRAINTS or contradictions.
SCHEMA: {{"level":"HIGH|MEDIUM|LOW","reasons":"short string"}}
"""
    try:
        cfg = {"temperature": 0.1, "response_mime_type": "application/json"}
        resp = model.generate_content(payload, generation_config=cfg)
        out = json.loads((resp.text or "").strip().replace("```json","").replace("```",""))
        lvl = str(out.get("level","")).upper()
        return {"level": lvl if lvl in ("HIGH","MEDIUM","LOW") else "MEDIUM", "reasons": out.get("reasons","")}
    except Exception as e:
        return {"level": "MEDIUM", "reasons": f"LLM confidence fallback: {e}"}

def _hybrid_confidence_guardrails(llm_level, constraint_summary):
    lvl = llm_level.upper()
    if constraint_summary == "ALL_CONFIRMED": return "HIGH" if lvl == "HIGH" else "MEDIUM"
    if constraint_summary == "CONFIRMED_WITH_DEVIATIONS": return "MEDIUM"
    if constraint_summary == "NO_EVALUABLE_CONSTRAINTS": return "LOW"
    return lvl if lvl in ("HIGH","MEDIUM","LOW") else "MEDIUM"

# --------------------------- Verification Engine Core --------------------------------------
def run_verification_engine(findings_path: str, root_dir: str, model: Any) -> Dict[str, Any]:
    try:
        with open(findings_path, "r", encoding="utf-8") as f: doc = json.load(f)
        findings, vocab = doc.get("findings", []), doc.get("metrics_vocabulary", {})
        dataset_root = doc.get("dataset_root")
        root_path = Path(root_dir).resolve()
        logger.info(f"ENGINE: Loaded {len(findings)} runs from {Path(findings_path).name} (schema v{doc.get('schema_version')})")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to load findings file: {e}", exc_info=True)
        return {}
    report = []
    for finding in findings:
        run_id, source_folder, claim_text = str(finding.get("run_id", "UNK")), finding.get("source_folder", ""), (finding.get("canonical_claim") or {}).get("statement", "")
        logger.info(f"\n----- RUN {run_id}: {source_folder} -----")
        artifacts, analyses, missing_artifacts = finding.get("artifacts", []), [], []
        for art in artifacts:
            role_name, rel_path = art.get("role"), art.get("path")
            role_key = ROLE_TO_PROMPT.get(role_name)
            if not role_key or not rel_path: continue
            resolved_path = _resolve_path(root_path, dataset_root, source_folder, rel_path)
            raw_bytes, text = _open_artifact(resolved_path)
            if raw_bytes is None and text is None:
                logger.warning(f"    - MISSING: {rel_path}"); missing_artifacts.append(rel_path); continue
            logger.info(f"  - Analyzing artifact: {rel_path}  (Role: {role_key})")
            file_info = {"file": rel_path, "role_key": role_key, "sha": None, "size": None}
            payload, local_summary = None, {}
            try:
                if role_key in ("comprehensive_png", "diagnostics_png"): payload = [PROMPT_LIBRARY[role_key], _prepare_png_payload(raw_bytes)]
                elif role_key in ("raw_csv", "diagnostics_csv", "by_param_csv"):
                    df = pd.read_csv(io.BytesIO(raw_bytes) if raw_bytes else resolved_path)
                    if role_key == "raw_csv": local_summary = summarize_raw_csv_from_df(df)
                    elif role_key == "diagnostics_csv": local_summary = summarize_timeseries_csv_from_df(df)
                    else: local_summary = summarize_by_param_csv_from_df(df)
                    payload = PROMPT_LIBRARY[role_key].format(FILE_CONTENT=csv_focus_minimal(df), LOCAL_SUMMARY=json.dumps(local_summary))
                elif role_key == "pdb":
                    snippet, frames = pdb_stratified_snippet(text or ""); payload = PROMPT_LIBRARY[role_key].format(FILE_CONTENT=snippet); local_summary = {"pdb_sampled_frames": frames}
                elif role_key == "md": payload = PROMPT_LIBRARY[role_key].format(FILE_CONTENT=safe_text_snippet(text or "", MAX_PAYLOAD_SIZE_BYTES), LOCAL_SUMMARY="{}")
                if resolved_path.exists(): file_info.update({"sha": short_sha256(resolved_path), "size": resolved_path.stat().st_size})
                an = call_gemini_with_retry(model, payload)
                file_info.update({"analysis": an, "local_summary": local_summary or None}); analyses.append(file_info)
                logger.info(f"    - Analysis complete for: {rel_path}")
            except Exception as e:
                logger.error(f"    - ERROR processing artifact {rel_path}: {e}", exc_info=True)
                analyses.append({"file": rel_path, "role_key": role_key, "analysis": {"error": f"processing_failed: {e}"}})
        nums, checks, discrepancies = precheck_and_cross_validate(analyses)
        constraints = (finding.get("canonical_claim") or {}).get("constraints", [])
        run_constraint_summary, constraint_results = evaluate_constraints(constraints, nums, vocab)
        verdict = "CONFIRMED" if run_constraint_summary == "ALL_CONFIRMED" else "DEVIATION" if run_constraint_summary == "CONFIRMED_WITH_DEVIATIONS" else "INDETERMINATE"
        logger.info(f"  - Deterministic Verdict: {verdict}")
        llm_conf = _llm_confidence_assess(model, claim_text, run_constraint_summary, nums, checks, analyses)
        conf = _hybrid_confidence_guardrails(llm_conf["level"], run_constraint_summary)
        rationale = call_gemini_with_retry(model, f"CLAIM:\n{claim_text}\n\nEVIDENCE:\n{json.dumps([{'file': a.get('file'), 'analysis': a.get('analysis')} for a in analyses])[:45000]}\n\nSCHEMA: {{\"rationale\":\"string\",\"evidence_citations\":[\"string\",...]}}")
        key_numbers = {metric: get_metric_from_sources(metric, nums, vocab)[0] for metric in vocab.keys()}
        report.append({"run_id": run_id, "claim_verified": claim_text, "independent_analyses": analyses,
                       "biophysics_verification": {"verdict": verdict, "constraint_summary": run_constraint_summary, "rationale": rationale.get("rationale", ""),
                                                   "key_numbers": key_numbers, "constraint_evaluations": constraint_results, "evidence_citations": rationale.get("evidence_citations", [])},
                       "data_QA": {"cross_checks": checks, "confidence": {"level": conf, "drivers": [llm_conf.get("reasons")]},
                                   "qa_issues": [{"missing_artifacts": missing_artifacts}, {"md_raw_discrepancies": discrepancies}] if missing_artifacts or discrepancies else []}})
    return {"comprehensive_verification_report": report}

# ==============================================================================
# SECTION 2: NARRATIVE ENGINE LOGIC (STAGE 4)
# ==============================================================================

def evaluate_thesis(thesis_cfg, per_run_numbers):
    if not thesis_cfg or not (cons := thesis_cfg.get("constraints")): return {"present": False}
    c0, agg, where, op, val = cons[0], cons[0].get("aggregation"), cons[0].get("where", []), cons[0].get("op"), cons[0].get("value")
    if agg != "count_runs_meeting": return {"present": True, "satisfied": None, "explanation": f"Unsupported aggregator '{agg}'."}
    def run_meets(kn): return all(apply_op(kn.get(w.get("metric")), w.get("op"), w.get("value"), w.get("tolerance", 0.0)) for w in where)
    count_meeting = sum(1 for kn in per_run_numbers if run_meets(kn))
    sat = apply_op(count_meeting, op, val, 0.0)
    return {"present": True, "satisfied": bool(sat), "count_meeting": count_meeting, "required": {"op": op, "value": val}, "explanation": f"{count_meeting} runs meet filter; require {op} {val}."}

def generate_narrative(api_key, model_name, project, thesis, runs_rows_md, exec_summary_md):
    if not api_key or not GEMINI_AVAILABLE:
        logger.warning("Gemini SDK/API Key not available. Skipping narrative generation.")
        return "_API key not found; skipping narrative generation._"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"""
ROLE: Senior structural biophysicist. Write a concise, publication-grade final assessment.
PROJECT: {project}
THESIS: {json.dumps(thesis or {}, indent=2)}
DETERMINISTIC SUMMARY: {exec_summary_md}
RUN-BY-RUN TABLE: {runs_rows_md}
STRICT INSTRUCTIONS: Be precise and non-speculative. Ground statements in the verified numbers. Keep to 250–350 words. No fluff.
STRUCTURE: 1) Overview. 2) Bullet findings. 3) Bullet limitations. 4) Bullet next steps.
"""
    try:
        return (model.generate_content(prompt, generation_config={"temperature":0.2}).text or "").strip()
    except Exception as e:
        logger.warning(f"Gemini narrative failed: {e}")
        return f"_Generated narrative unavailable (API call failed: {e})._"

def write_final_report(findings_doc, verification_report, api_key, out_md="Biophysics_Final_Report.md", out_json="Biophysics_Final_Report_Summary.json"):
    project, thesis_cfg, findings = findings_doc.get("project",""), (findings_doc.get("thesis") or {}).get("canonical_field_claim"), findings_doc.get("findings", [])
    cfg_by_id = {int(r.get("run_id")): r for r in findings if "run_id" in r}
    blocks = verification_report.get("comprehensive_verification_report", [])
    headers = ["run_id", "source_folder", "verdict(confidence)", "best_final_RMSD_A", "best_final_Rg_A", "runs_count", "failures"]
    rows, verdict_counts, conf_counts, rmsds, per_run_keynums = [], {"CONFIRMED": 0, "DEVIATION": 0, "INDETERMINATE": 0}, {"HIGH": 0, "MEDIUM": 0, "LOW": 0}, [], []
    for b in blocks:
        rid = int(b.get("run_id"))
        biophys, dataqa = b.get("biophysics_verification", {}), b.get("data_QA", {})
        key_numbers = biophys.get("key_numbers", {}) or {}
        per_run_keynums.append(key_numbers)
        rmsds.append(key_numbers.get("best_final_RMSD_A"))
        verdict, conf = (biophys.get("verdict") or "INDETERMINATE").upper(), ((dataqa.get("confidence") or {}).get("level") or "MEDIUM").upper()
        verdict_counts[verdict] += 1; conf_counts[conf] += 1
        rows.append([rid, cfg_by_id.get(rid, {}).get("source_folder", ""), f"{verdict} — {conf}", f"{key_numbers.get('best_final_RMSD_A'):.2f}" if key_numbers.get('best_final_RMSD_A') is not None else "—", f"{key_numbers.get('best_final_Rg_A'):.2f}" if key_numbers.get('best_final_Rg_A') is not None else "—", key_numbers.get('runs_count', '—'), key_numbers.get('failures', '—')])
    rows.sort(key=lambda r: int(r[0]))
    valid_rmsds = [v for v in rmsds if v is not None]
    median_rmsd, min_rmsd, max_rmsd = (statistics.median(valid_rmsds) if valid_rmsds else None, min(valid_rmsds) if valid_rmsds else None, max(valid_rmsds) if valid_rmsds else None)
    thesis_eval = evaluate_thesis(thesis_cfg, per_run_keynums)
    exec_summary_md = (f"- Runs analyzed: **{len(blocks)}**\n"
                       f"- Verdicts — CONFIRMED: **{verdict_counts['CONFIRMED']}**, DEVIATION: **{verdict_counts['DEVIATION']}**, INDETERMINATE: **{verdict_counts['INDETERMINATE']}**\n"
                       f"- RMSD summary: median **{median_rmsd:.2f} Å**, min **{min_rmsd:.2f} Å**, max **{max_rmsd:.2f} Å**\n"
                       f"- Thesis check: **{'✅ satisfied' if thesis_eval.get('satisfied') else '❌ not satisfied' if thesis_eval.get('satisfied') is False else '—'}** — {thesis_eval.get('explanation','')}")
    table_md = "| " + " | ".join(headers) + " |\n| " + " | ".join([":--" for _ in headers]) + " |\n" + "\n".join("| " + " | ".join(map(str,r)) + " |" for r in rows)
    narrative = generate_narrative(api_key, MODEL_NAME, project, thesis_cfg, table_md, exec_summary_md)
    md_content = (f"# Biophysics Final Report\n\n**Project:** {project}\n\n## Executive Summary (deterministic)\n{exec_summary_md}\n\n"
                  f"## Run-by-Run Summary\n{table_md}\n\n## Expert Narrative (Gemini)\n{narrative}\n")
    with open(out_md, "w", encoding="utf-8") as f: f.write(md_content)
    with open(out_json, "w", encoding="utf-8") as f: json.dump({"verdict_counts": verdict_counts, "confidence_counts": conf_counts, "median_best_RMSD_A": median_rmsd, "thesis_evaluation": thesis_eval}, f, indent=2)
    logger.info(f"✅ Wrote final reports: {out_md} and {out_json}")

# ==============================================================================
# SECTION 3: MAIN ORCHESTRATOR
# ==============================================================================

def main():
    if IN_COLAB:
        try:
            logger.info("Mounting Google Drive (if available)...")
            drive.mount("/content/drive", force_remount=True)
        except Exception:
            logger.warning("Drive mount skipped or failed.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and os.path.exists("API_KEY.txt"):
        api_key = Path("API_KEY.txt").read_text().strip()
    elif IN_COLAB and not api_key:
        try:
            logger.info("Please upload API_KEY.txt")
            up = files.upload()
            if "API_KEY.txt" in up: api_key = up["API_KEY.txt"].decode().strip()
        except Exception: pass
    
    if not api_key:
        logger.critical("FATAL: Gemini API key is missing. The audit engine requires the API key to perform AI peer review and cannot proceed.")
        return

    findings_path = os.getenv("FINDINGS_JSON", "findings.json")
    default_root = "/content/drive/MyDrive/Folding/PharmaApp/ProductDetails/P53 Study/Plots/_analysis" if IN_COLAB else "."
    root_dir = os.getenv("DATA_ROOT_DIR", default_root)

    logger.info(f"Gemini model: {MODEL_NAME}")
    logger.info(f"Using findings: {findings_path}")
    logger.info(f"Base data root_dir: {root_dir}")

    model = None
    if GEMINI_AVAILABLE:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_NAME)
            logger.info("Successfully initialized Gemini model.")
        except Exception as e:
            logger.critical(f"FATAL: Failed to initialize Gemini model, even with an API key. Error: {e}")
            return
    else:
        logger.critical("FATAL: The 'google-generativeai' library is not installed. Cannot proceed.")
        return

    if not model:
        logger.critical("FATAL: Gemini model could not be initialized. Aborting.")
        return

    # --- Execute Verification and Narrative ---
    verification_report = run_verification_engine(findings_path=findings_path, root_dir=root_dir, model=model)
    if not verification_report or not verification_report.get("comprehensive_verification_report"):
        logger.critical("Verification engine failed to produce a report. Aborting.")
        return
    
    # PATCH v1.4: Save the comprehensive report to disk
    comprehensive_report_path = "Comprehensive_Verification_Report.json"
    with open(comprehensive_report_path, "w", encoding="utf-8") as f:
        json.dump(verification_report, f, indent=2)
    logger.info(f"✅ Wrote comprehensive verification report: {comprehensive_report_path}")

    with open(findings_path, "r", encoding="utf-8") as f:
        findings_doc = json.load(f)

    write_final_report(findings_doc, verification_report, api_key)

if __name__ == "__main__":
    main()
