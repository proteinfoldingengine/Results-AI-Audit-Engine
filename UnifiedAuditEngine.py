# ==============================================================================
# Unified Audit Engine (p53/MECP2-ready)
# v2.2  — with 8 integrated patches
# ==============================================================================
#
# WHAT'S NEW (v2.2):
# PATCH #7: Fixed metric lookup logic to correctly use original vocab keys.
# PATCH #8: Correctly passed API key to the final narrative generation stage.
#
# PREVIOUS PATCHES (v2.1):
# PATCH #6: Safe prompt formatting to prevent KeyErrors from literal braces in schemas.
# PATCH #5: Schema auto-repair for findings.json to normalize aliases and enums.
# PATCH #4: Canonical metric aliasing (e.g., "final_RMSD_A" <-> "best_final_RMSD_A").
# PATCH #3: Canonical aliasing layer for sources and artifact roles.
# PATCH #2: Final report guard when no valid RMSDs are present (no TypeError).
# PATCH #1: Robust PDB payload guard for large/single-frame PDBs (prevents timeouts).
#
# WORKFLOW:
# 1) Clone repo
# 2) Load & auto-repair findings.json (canonicalize schema enums & aliases)
# 3) Analyze artifacts deterministically + AI assists
# 4) Evaluate constraints and thesis
# 5) Emit comprehensive JSON + concise MD report
# ==============================================================================

import os, sys, re, json, time, hashlib, logging, random, datetime, statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from PIL import Image
import requests
import io

# (Colab helpers)
try:
    from google.colab import files
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
logger = logging.getLogger("unified_engine")
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

# --- CONFIGURATION ---
GITHUB_DATA_REPO_URL = "https://github.com/ProteinFoldingEngine/Results-AI-Audit-Engine.git"
PROJECT_NAME = "MECP2"   # "p53" or "MECP2" etc.

# ---------------- Canonical vocab, aliases, and role handling (PATCH #3 & #4 & #5) -------------
CANONICAL_PREFERRED_SOURCES = {"md", "raw_csv", "diagnostics_csv"}  # schema enum
SOURCE_ALIASES = {  # input -> canonical
    "md_report": "md",
    "markdown": "md",
    "report": "md",
    "md": "md",
    "raw": "raw_csv",
    "raw_csv": "raw_csv",
    "diagnostics": "diagnostics_csv",
    "diagnostics_csv": "diagnostics_csv",
}

# primary roles we actively analyze (schema enum)
PRIMARY_ROLES = {
    "comprehensive_png", "diagnostics_png", "diagnostics_csv",
    "raw_csv", "by_param_csv", "trajectory_pdb", "md_report"
}

# Auxiliary roles we *allow* but skip analysis (don’t break schema; just logged)
AUX_ROLES = {
    "contact_maps_npz": "aux_contact_maps",
    "audit_log": "aux_audit_log",
    "npz_contact_maps": "aux_contact_maps",  # extra guard
}

# Map roles to prompt-keys (only primary analyzed roles)
ROLE_TO_PROMPT = {
    "md_report": "md",
    "raw_csv": "raw_csv",
    "by_param_csv": "by_param_csv",
    "diagnostics_csv": "diagnostics_csv",
    "comprehensive_png": "comprehensive_png",
    "diagnostics_png": "diagnostics_png",
    "trajectory_pdb": "pdb",
}

# Canonical metric aliases (PATCH #4)
# We normalize *incoming* metric names and lookups to the canonical keys used in numerics
METRIC_ALIASES = {
    # RMSD
    "final_RMSD_A": "best_final_RMSD_A",
    "best_final_RMSD_A": "best_final_RMSD_A",
    "final_rmsd_a": "best_final_RMSD_A",
    "best_final_rmsd_a": "best_final_RMSD_A",
    # Rg
    "final_Rg_A": "best_final_Rg_A",
    "best_final_Rg_A": "best_final_Rg_A",
    "final_rg_a": "best_final_Rg_A",
    "best_final_rg_a": "best_final_Rg_A",
    # Other metrics pass-through
    "runs_count": "runs_count",
    "failures": "failures",
    "median_salt_bridges": "median_salt_bridges",
}

def normalize_metric_name(metric: Optional[str]) -> Optional[str]:
    if not metric: return metric
    return METRIC_ALIASES.get(metric, metric)

def normalize_source_name(src: str) -> Optional[str]:
    s = SOURCE_ALIASES.get(src, src)
    return s if s in CANONICAL_PREFERRED_SOURCES else None

def normalize_role_name(role: str) -> Tuple[str, bool]:
    """Return (role_key, is_aux). Unknown roles become aux_* so they don't break the flow."""
    if role in PRIMARY_ROLES:
        return role, False
    if role in AUX_ROLES:
        return AUX_ROLES[role], True
    # anything else -> generic aux
    return f"aux_{role}", True

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
{
  "runs_count": int | null,
  "failures": int | null,
  "best_final_RMSD_A": float | null,
  "best_final_Rg_A": float | null,
  "classification": "near_native" | "intermediate" | "misfolded" | "unknown",
  "table_row_source": "string"
}
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
{
  "consistency_check": "consistent"|"inconsistent"|"cannot_determine",
  "notes": "short explanation"
}
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
{
  "n_rows": int | null,
  "distinct_param_sets": int | null,
  "sensitivity": "low" | "moderate" | "high" | "unknown",
  "notes": "brief rationale"
}
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
{
  "consistency_check": "consistent"|"inconsistent"|"cannot_determine",
  "notes": "short explanation"
}
OUTPUT: JSON only.
""",
    "pdb": """ROLE: Structural biologist.
TASK: Assess PDB snippet (focus on CA atoms if present).

PDB_SNIPPET:
<<<
{FILE_CONTENT}
>>>

SCHEMA:
{
  "frames_sampled":[int,...]|null,
  "qualitative_compaction":"yes"|"no"|"uncertain",
  "notes":"string"
}
OUTPUT: JSON only.
""",
    "comprehensive_png": """ROLE: Structural biologist.
TASK: Assess comprehensive PNG report (RMSD, Rg, snapshot).

SCHEMA:
{
  "time_series_description":"string",
  "final_structure_description":"string",
  "overall_conclusion":"string"
}
OUTPUT: JSON only.
""",
    "diagnostics_png": """ROLE: Analyst.
TASK: Assess diagnostics PNG with RMSD/Rg/H-bonds/salt bridges).

SCHEMA:
{
  "rmsd_trend":"string",
  "rg_trend":"string",
  "interaction_trends":"string",
  "mechanical_interpretation":"string"
}
OUTPUT: JSON only.
"""
}

# ==============================================================================
# SECTION 1: VERIFICATION ENGINE LOGIC
# ==============================================================================

# --- PATCH #6: FIX FOR PROMPT FORMATTING KEYERROR ---
def _safe_prompt_format(template: str, **kwargs) -> str:
    """
    Safely format a template that includes literal { } (e.g., JSON SCHEMA blocks).
    We escape all braces, then unescape our two placeholders.
    """
    # Step 1: escape everything
    esc = template.replace("{", "{{").replace("}", "}}")
    # Step 2: re-enable the two placeholders we actually support
    esc = esc.replace("{{FILE_CONTENT}}", "{FILE_CONTENT}")
    esc = esc.replace("{{LOCAL_SUMMARY}}", "{LOCAL_SUMMARY}")
    # Step 3: standard format
    return esc.format(**kwargs)

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
    if rmsd_col is not None:
        s = pd.to_numeric(df[rmsd_col], errors="coerce").ffill().bfill().dropna()
        if len(s) > 0:
            out["final_RMSD_A"] = round(float(s.iloc[-1]), 5)
            tail = s.iloc[max(0, int(len(s) * 0.9)):]
            out["stabilized"] = bool(tail.std() <= 1.0)
    if rg_col is not None:
        g = pd.to_numeric(df[rg_col], errors="coerce").ffill().bfill().dropna()
        if len(g) > 0:
            out["final_Rg_A"] = round(float(g.iloc[-1]), 5)
    if sb_col is not None:
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

# --- PATCH #1: FIX FOR PDB TIMEOUT / PAYLOAD SIZE ---
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
    snippet_content = ""
    if not models:
        snippet_content = "".join(lines[:4000])
    else:
        num = len(models)
        if max_frames < 2:
            idx = [0]
        else:
            idx = sorted(set([0, num - 1] + [int(num * (i / (max_frames - 1))) for i in range(1, max_frames - 1)]))
        frames = [i + 1 for i in idx]
        out_lines = header + [f"REMARK SAMPLED FRAMES: {frames}\n"]
        for i in idx:
            m = models[i]
            ca = [ln for ln in m if " CA " in ln]
            out_lines.extend([m[0]] + ca[:1000] + [m[-1]])
        snippet_content = "".join(out_lines)
    if len(snippet_content.encode("utf-8", errors="ignore")) > MAX_PAYLOAD_SIZE_BYTES:
        snippet_content = safe_text_snippet(snippet_content, MAX_PAYLOAD_SIZE_BYTES)
    return snippet_content, frames

def short_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()[:12]

# --------------------------- Gemini call with repair --------------------------
def call_gemini_with_retry(model, payload, max_retries=3, timeout=300):
    for attempt in range(max_retries):
        try:
            cfg = {"temperature": 0.0, "response_mime_type": "application/json"}
            logger.info(f"    --> Attempting Gemini API call (Attempt {attempt + 1}/{max_retries})...")
            resp = model.generate_content(payload, generation_config=cfg, request_options={"timeout": timeout})
            logger.info(f"    --> API call returned.")
            txt = (resp.text or "").strip().replace("```json","").replace("```","")
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                logger.warning("    --> Invalid JSON detected. Attempting one repair pass.")
                repair = f"Fix JSON only:\n{txt}"
                r2 = model.generate_content(repair, generation_config=cfg)
                return json.loads((r2.text or "").strip().replace("```json","").replace("```",""))
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

# --------------------------- Metric & Source Lookup --------------------------
# --- PATCH #7: FIX FOR METRIC LOOKUP LOGIC ---
def get_metric_from_sources(metric: str, nums: Dict, vocab: Dict) -> Tuple[Optional[Any], Optional[str]]:
    # The `metric` variable passed in is the ORIGINAL name from the vocab declaration
    # We normalize it here to get the CANONICAL name for the key_map lookup
    metric_norm = normalize_metric_name(metric)

    # Use the ORIGINAL metric name to look up its own definition in the vocab
    preferred = [normalize_source_name(s) for s in (vocab.get(metric, {}) or {}).get("preferred_sources", [])]
    preferred = [s for s in preferred if s]  # drop non-canonical

    # Key map into our numeric store (uses CANONICAL names)
    key_map = {
        "md": {
            "best_final_RMSD_A": "md_best_rmsd",
            "best_final_Rg_A": "md_best_rg",
            "runs_count": "md_runs",
            "failures": "md_failures",
        },
        "raw_csv": {
            "best_final_RMSD_A": "raw_best_rmsd",
            "best_final_Rg_A": "raw_best_rg",
            "runs_count": "raw_runs",
        },
        "diagnostics_csv": {
            "best_final_RMSD_A": "ts_final_rmsd",  # final of timeseries
            "best_final_Rg_A": "ts_final_rg",
            "median_salt_bridges": "ts_median_salt_bridges",
        },
    }
    for source in preferred:
        # Use the CANONICAL name to look up the key for the `nums` dictionary
        key = key_map.get(source, {}).get(metric_norm)
        if key and nums.get(key) is not None:
            return nums[key], source
    return None, None

def apply_op(lhs, op, rhs, tol=0.0):
    if lhs is None or rhs is None: return None
    try:
        lhs, rhs, tol = float(lhs), float(rhs), float(tol or 0.0)
    except (ValueError, TypeError):
        return None
    ops = {
        "<=": lambda a, b: a <= b + tol,
        ">=": lambda a, b: a >= b - tol,
        "==": lambda a, b: abs(a - b) <= tol,
        "<":  lambda a, b: a <  b + tol,
        ">":  lambda a, b: a >  b - tol,
        "!=": lambda a, b: abs(a - b) > tol
    }
    return ops.get(op)(lhs, rhs) if op in ops else None

def evaluate_constraints(constraints: List[Dict], nums: Dict, vocab: Dict) -> Tuple[str, List[Dict]]:
    if not constraints: return "NO_EVALUABLE_CONSTRAINTS", []
    results, any_dev, any_eval = [], False, False
    for const in constraints:
        # Use the REPAIRED metric name from the constraint for evaluation
        metric_in_constraint = normalize_metric_name(const.get("metric"))
        op, value, tol = const.get("op"), const.get("value"), const.get("tolerance", 0.0)

        # Find the original metric key in the vocab that aliases to our canonical name
        original_metric_key = const.get("metric") # Fallback
        for k in vocab.keys():
            if normalize_metric_name(k) == metric_in_constraint:
                original_metric_key = k
                break

        actual, source = get_metric_from_sources(original_metric_key, nums, vocab)

        if actual is None:
            results.append({"constraint": const, "status": "NOT_EVALUATED", "reason": f"Metric '{metric_in_constraint}' not found."})
            continue
        any_eval = True
        ok = apply_op(actual, op, value, tol)
        delta = float(actual) - float(value) if actual is not None and value is not None else None
        results.append({"constraint": const, "status": "CONFIRMED" if ok else "DEVIATION",
                          "actual_value": actual, "source": source, "delta": delta})
        if ok is False: any_dev = True
    if not any_eval: return "NO_EVALUABLE_CONSTRAINTS", results
    return "CONFIRMED_WITH_DEVIATIONS" if any_dev else "ALL_CONFIRMED", results

def precheck_and_cross_validate(analyses):
    nums = {
        "md_best_rmsd": None, "md_best_rg": None, "raw_best_rmsd": None, "raw_best_rg": None,
        "md_runs": None, "raw_runs": None, "md_failures": None, "diagnostics_stable": None,
        "ts_final_rmsd": None, "ts_final_rg": None, "ts_median_salt_bridges": None
    }
    for a in analyses:
        role, an, loc = a.get("role_key"), a.get("analysis", {}), a.get("local_summary", {}) or {}
        if not isinstance(an, dict): continue
        if role == "md":
            nums.update({
                "md_best_rmsd": an.get("best_final_RMSD_A"),
                "md_best_rg": an.get("best_final_Rg_A"),
                "md_runs": an.get("runs_count"),
                "md_failures": an.get("failures")
            })
        elif role == "raw_csv":
            nums.update({
                "raw_best_rmsd": loc.get("best_final_RMSD_A"),
                "raw_best_rg": loc.get("best_final_Rg_A"),
                "raw_runs": loc.get("runs_count")
            })
        elif role == "diagnostics_csv":
            nums.update({
                "ts_final_rmsd": loc.get("final_RMSD_A"),
                "ts_final_rg": loc.get("final_Rg_A"),
                "diagnostics_stable": bool(loc.get("stabilized")),
                "ts_median_salt_bridges": loc.get("median_salt_bridges")
            })
    def status(md, raw, tol=0.5):
        if md is None or raw is None: return {"md": md, "raw": raw, "status": "MISSING"}
        return {"md": md, "raw": raw, "status": "MATCH" if abs(md - raw) <= tol else "MISMATCH"}
    checks = {
        "best_RMSD_md_vs_raw": status(nums["md_best_rmsd"], nums["raw_best_rmsd"], 0.5),
        "best_Rg_md_vs_raw":   status(nums["md_best_rg"], nums["raw_best_rg"], 0.25),
        "runs_count_md_vs_raw": status(nums["md_runs"], nums["raw_runs"], 0.0)
    }
    discrepancies = [{"metric": k, "details": f"MD={v['md']}, RAW={v['raw']}"} for k, v in checks.items() if v["status"] == "MISMATCH"]
    return nums, checks, discrepancies

# --------------------------- IO Helpers --------------------------------------
def _resolve_path(root_dir: Path, source_folder: str, artifact_path: str) -> Path:
    return (Path(root_dir) / source_folder / artifact_path).resolve()

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

# --------------------------- Confidence Logic --------------------------------
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
    lvl = (llm_level or "").upper()
    if constraint_summary == "ALL_CONFIRMED": return "HIGH" if lvl == "HIGH" else "MEDIUM"
    if constraint_summary == "CONFIRMED_WITH_DEVIATIONS": return "MEDIUM"
    if constraint_summary == "NO_EVALUABLE_CONSTRAINTS": return "LOW"
    return lvl if lvl in ("HIGH","MEDIUM","LOW") else "MEDIUM"

# --------------------------- Schema Auto-Repair (PATCH #5) -------------------
def _auto_repair_findings(doc: Dict[str, Any]) -> Dict[str, Any]:
    repaired = json.loads(json.dumps(doc))  # deep copy
    changes = []

    # Fix metrics_vocabulary preferred_sources aliases
    mv = repaired.get("metrics_vocabulary") or {}
    for metric_key, mcfg in mv.items():
        ps = (mcfg or {}).get("preferred_sources", [])
        new_ps = []
        for s in ps:
            ns = normalize_source_name(s)
            if ns:
                new_ps.append(ns)
            else:
                changes.append(f"metrics_vocabulary.{metric_key}.preferred_sources: dropped non-canonical '{s}'")
        # default fallbacks if empty
        if not new_ps:
            # if metric looks like failures, prefer md; else md->raw->diag
            if normalize_metric_name(metric_key) == "failures":
                new_ps = ["md"]
            else:
                new_ps = ["md", "raw_csv", "diagnostics_csv"]
            changes.append(f"metrics_vocabulary.{metric_key}.preferred_sources: filled default {new_ps}")
        if new_ps != ps:
            mcfg["preferred_sources"] = new_ps
            changes.append(f"metrics_vocabulary.{metric_key}.preferred_sources: {ps} -> {new_ps}")

    # Normalize constraints metric names (runs & thesis)
    def fix_constraints(constraints, path_hint):
        if not constraints: return
        for c in constraints:
            m = c.get("metric")
            if m:
                nm = normalize_metric_name(m)
                if nm != m:
                    c["metric"] = nm
                    changes.append(f"{path_hint}.constraints.metric: '{m}' -> '{nm}'")

    # Thesis constraints
    thesis_claim = ((repaired.get("thesis") or {}).get("canonical_field_claim") or {})
    fix_constraints(thesis_claim.get("constraints"), "thesis.canonical_field_claim")
    for constraint in thesis_claim.get("constraints", []):
        fix_constraints(constraint.get("where"), "thesis.canonical_field_claim.where")


    # Findings constraints and artifacts
    for i, run in enumerate(repaired.get("findings", []), start=1):
        cpath = f"findings[{i}]"
        # Fix claim constraints metric names
        fix_constraints(((run.get("canonical_claim") or {}).get("constraints")), f"{cpath}.canonical_claim")

        # Normalize artifact roles (allow AUX; skip unknown)
        arts = run.get("artifacts") or []
        for a in arts:
            role = a.get("role")
            if not role:
                continue
            new_role, is_aux = normalize_role_name(role)
            if new_role != role:
                a["role"] = new_role
                changes.append(f"{cpath}.artifacts.role: '{role}' -> '{new_role}'")
            # AUX roles are kept but not analyzed

    if changes:
        logger.info("=== SCHEMA AUTO-REPAIR SUMMARY ===")
        for ch in changes:
            logger.info(f"  - {ch}")
        logger.info("=== END AUTO-REPAIR ===")
    return repaired

# --------------------------- Engine Core -------------------------------------
def run_verification_engine(findings_path: str, root_dir: str, model: Any) -> Dict[str, Any]:
    try:
        with open(findings_path, "r", encoding="utf-8") as f:
            raw_doc = json.load(f)
        # PATCH #5: auto-repair
        doc = _auto_repair_findings(raw_doc)
        findings, vocab = doc.get("findings", []), doc.get("metrics_vocabulary", {})
        root_path = Path(root_dir).resolve()
        logger.info(f"ENGINE: Loaded {len(findings)} runs from {Path(findings_path).name} (schema v{doc.get('schema_version')})")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to load findings file: {e}", exc_info=True)
        return {}

    report = []
    for finding in findings:
        run_id = str(finding.get("run_id", "UNK"))
        source_folder = finding.get("source_folder", "")
        claim_text = (finding.get("canonical_claim") or {}).get("statement", "")
        logger.info(f"\n----- RUN {run_id}: {source_folder} -----")

        artifacts = finding.get("artifacts", []) or []
        analyses, missing_artifacts = [], []

        for art in artifacts:
            role_name_raw, rel_path = art.get("role"), art.get("path")
            if not rel_path or not role_name_raw:
                continue
            # Normalize role (PATCH #3/#5)
            role_name, is_aux = normalize_role_name(role_name_raw)
            role_key = ROLE_TO_PROMPT.get(role_name)  # only primary roles are mapped
            if is_aux or not role_key:
                # keep note but skip model analysis
                logger.info(f"  - Skipping AUX/unknown artifact: {rel_path}  (Role: {role_name_raw} -> {role_name})")
                analyses.append({"file": rel_path, "role_key": "aux", "analysis": {"note": "auxiliary_artifact_skipped"}})
                continue

            resolved_path = _resolve_path(root_path, source_folder, rel_path)
            raw_bytes, text = _open_artifact(resolved_path)
            if raw_bytes is None and text is None:
                logger.warning(f"    - MISSING: {rel_path}")
                missing_artifacts.append(rel_path)
                continue

            logger.info(f"  - Analyzing artifact: {rel_path}  (Role: {role_key})")
            file_info = {"file": rel_path, "role_key": role_key, "sha": None, "size": None}
            payload, local_summary = None, {}
            try:
                if role_key in ("comprehensive_png", "diagnostics_png"):
                    payload = [PROMPT_LIBRARY[role_key], _prepare_png_payload(raw_bytes)]
                elif role_key in ("raw_csv", "diagnostics_csv", "by_param_csv"):
                    df = pd.read_csv(io.BytesIO(raw_bytes) if raw_bytes else resolved_path)
                    if role_key == "raw_csv":
                        local_summary = summarize_raw_csv_from_df(df)
                    elif role_key == "diagnostics_csv":
                        local_summary = summarize_timeseries_csv_from_df(df)
                    else:
                        local_summary = summarize_by_param_csv_from_df(df)
                    # --- PATCH #6 APPLIED HERE ---
                    payload = _safe_prompt_format(
                        PROMPT_LIBRARY[role_key],
                        FILE_CONTENT=csv_focus_minimal(df),
                        LOCAL_SUMMARY=json.dumps(local_summary)
                    )
                elif role_key == "pdb":
                    snippet, frames = pdb_stratified_snippet(text or "")
                    # --- PATCH #6 APPLIED HERE ---
                    payload = _safe_prompt_format(PROMPT_LIBRARY[role_key], FILE_CONTENT=snippet)
                    local_summary = {"pdb_sampled_frames": frames}
                elif role_key == "md":
                    # --- PATCH #6 APPLIED HERE ---
                    payload = _safe_prompt_format(
                        PROMPT_LIBRARY[role_key],
                        FILE_CONTENT=safe_text_snippet(text or "", MAX_PAYLOAD_SIZE_BYTES),
                        LOCAL_SUMMARY="{}"
                    )

                if resolved_path.exists():
                    file_info.update({"sha": short_sha256(resolved_path), "size": resolved_path.stat().st_size})

                if isinstance(payload, str):
                    logger.info(f"    - Preparing to send text payload of size: {len(payload.encode('utf-8','ignore'))} bytes.")
                elif isinstance(payload, list) and len(payload) > 1 and isinstance(payload[1], Image.Image):
                    logger.info(f"    - Preparing to send image payload.")

                an = call_gemini_with_retry(model, payload)
                file_info.update({"analysis": an, "local_summary": local_summary or None})
                analyses.append(file_info)
                logger.info(f"    - Analysis complete for: {rel_path}")
            except Exception as e:
                logger.error(f"    - ERROR processing artifact {rel_path}: {e}", exc_info=True)
                analyses.append({"file": rel_path, "role_key": role_key, "analysis": {"error": f"processing_failed: {e}"}})

        nums, checks, discrepancies = precheck_and_cross_validate(analyses)
        constraints = (finding.get("canonical_claim") or {}).get("constraints", [])
        run_constraint_summary, constraint_results = evaluate_constraints(constraints, nums, vocab)
        verdict = "CONFIRMED" if run_constraint_summary == "ALL_CONFIRMED" else \
                  "DEVIATION" if run_constraint_summary == "CONFIRMED_WITH_DEVIATIONS" else \
                  "INDETERMINATE"
        logger.info(f"  - Deterministic Verdict: {verdict}")

        # Confidence (LLM + guardrails)
        llm_conf = _llm_confidence_assess(model, claim_text, run_constraint_summary, nums, checks, analyses)
        conf = _hybrid_confidence_guardrails(llm_conf.get("level","MEDIUM"), run_constraint_summary)

        # Rationale (compact)
        rationale = call_gemini_with_retry(
            model,
            f"CLAIM:\n{claim_text}\n\nEVIDENCE:\n{json.dumps([{'file': a.get('file'), 'analysis': a.get('analysis')} for a in analyses])[:45000]}\n\nSCHEMA: {{\"rationale\":\"string\",\"evidence_citations\":[\"string\",...]}}"
        )

        # Key numbers by original metric names from metrics_vocabulary
        key_numbers = {}
        for metric in (vocab.keys() if isinstance(vocab, dict) else []):
            key_numbers[metric] = get_metric_from_sources(metric, nums, vocab)[0]

        report.append({
            "run_id": run_id,
            "claim_verified": claim_text,
            "independent_analyses": analyses,
            "biophysics_verification": {
                "verdict": verdict,
                "constraint_summary": run_constraint_summary,
                "rationale": rationale.get("rationale", ""),
                "key_numbers": key_numbers,
                "constraint_evaluations": constraint_results,
                "evidence_citations": rationale.get("evidence_citations", [])
            },
            "data_QA": {
                "cross_checks": checks,
                "confidence": {"level": conf, "drivers": [llm_conf.get("reasons")]},
                "qa_issues": [{"missing_artifacts": missing_artifacts}, {"md_raw_discrepancies": discrepancies}] if missing_artifacts or discrepancies else []
            }
        })
    return {"comprehensive_verification_report": report}

# ==============================================================================
# SECTION 2: THESIS EVAL + REPORTING
# ==============================================================================

def evaluate_thesis(thesis_cfg, per_run_numbers):
    if not thesis_cfg or not (cons := thesis_cfg.get("constraints")):
        return {"present": False}

    # Handle multiple constraints in the thesis
    all_results = []
    for constraint in cons:
        agg, where, op, val = constraint.get("aggregation"), constraint.get("where", []), constraint.get("op"), constraint.get("value")
        if agg != "count_runs_meeting":
            all_results.append({"present": True, "satisfied": None, "explanation": f"Unsupported aggregator '{agg}'."})
            continue

        def run_meets(kn):
            for w in where:
                m = normalize_metric_name(w.get("metric"))
                # Use the canonical metric name for lookup in per_run_numbers
                if apply_op(kn.get(m), w.get("op"), w.get("value"), w.get("tolerance", 0.0)) is not True:
                    return False
            return True

        count_meeting = sum(1 for kn in per_run_numbers if run_meets(kn))
        sat = apply_op(count_meeting, op, val, 0.0)
        all_results.append({
            "satisfied": bool(sat),
            "count_meeting": count_meeting,
            "required": {"op": op, "value": val},
            "explanation": f"{count_meeting} runs meet filter; require {op} {val}."
        })

    # Combine results: satisfied if ALL constraints are satisfied
    final_satisfied = all(r.get("satisfied", False) for r in all_results)
    explanations = " | ".join(r.get("explanation", "") for r in all_results)
    return {
        "present": True,
        "satisfied": final_satisfied,
        "explanation": explanations,
        "details": all_results
    }


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
    project = findings_doc.get("project","")
    thesis_cfg = (findings_doc.get("thesis") or {}).get("canonical_field_claim")
    cfg_by_id = {int(r.get("run_id")): r for r in (findings_doc.get("findings", []) or []) if "run_id" in r}

    blocks = verification_report.get("comprehensive_verification_report", [])
    headers = ["run_id", "source_folder", "verdict(confidence)", "best_final_RMSD_A", "best_final_Rg_A", "runs_count", "failures"]
    rows, verdict_counts, conf_counts, rmsds, per_run_keynums = [], {"CONFIRMED": 0, "DEVIATION": 0, "INDETERMINATE": 0}, {"HIGH": 0, "MEDIUM": 0, "LOW": 0}, [], []

    for b in blocks:
        rid = int(b.get("run_id"))
        biophys, dataqa = b.get("biophysics_verification", {}), b.get("data_QA", {})
        key_numbers = biophys.get("key_numbers", {}) or {}

        # store per-run canonical metrics for thesis eval
        per_kn = {}
        for k, v in key_numbers.items():
            per_kn[normalize_metric_name(k)] = v
        per_run_keynums.append(per_kn)

        rmsds.append(per_kn.get("best_final_RMSD_A"))
        verdict = (biophys.get("verdict") or "INDETERMINATE").upper()
        conf = ((dataqa.get("confidence") or {}).get("level") or "MEDIUM").upper()
        verdict_counts[verdict] += 1; conf_counts[conf] += 1

        rows.append([
            rid,
            cfg_by_id.get(rid, {}).get("source_folder", ""),
            f"{verdict} — {conf}",
            f"{per_kn.get('best_final_RMSD_A'):.2f}" if per_kn.get('best_final_RMSD_A') is not None else "—",
            f"{per_kn.get('best_final_Rg_A'):.2f}" if per_kn.get('best_final_Rg_A') is not None else "—",
            per_kn.get('runs_count', '—'),
            per_kn.get('failures', '—')
        ])

    rows.sort(key=lambda r: int(r[0]))

    # PATCH #2: robust RMSD summary
    valid_rmsds = [v for v in rmsds if v is not None]
    median_rmsd = statistics.median(valid_rmsds) if valid_rmsds else None
    min_rmsd    = min(valid_rmsds) if valid_rmsds else None
    max_rmsd    = max(valid_rmsds) if valid_rmsds else None

    rmsd_summary_line = "- RMSD summary: Not Available (no valid RMSD values found)"
    if median_rmsd is not None:
        rmsd_summary_line = f"- RMSD summary: median **{median_rmsd:.2f} Å**, min **{min_rmsd:.2f} Å**, max **{max_rmsd:.2f} Å**"

    thesis_eval = evaluate_thesis(thesis_cfg, per_run_keynums)
    exec_summary_md = (
        f"- Runs analyzed: **{len(blocks)}**\n"
        f"- Verdicts — CONFIRMED: **{verdict_counts['CONFIRMED']}**, DEVIATION: **{verdict_counts['DEVIATION']}**, INDETERMINATE: **{verdict_counts['INDETERMINATE']}**\n"
        f"{rmsd_summary_line}\n"
        f"- Thesis check: **{'✅ satisfied' if thesis_eval.get('satisfied') else '❌ not satisfied' if thesis_eval.get('satisfied') is False else '—'}** — {thesis_eval.get('explanation','')}"
    )

    table_md = "| " + " | ".join(headers) + " |\n| " + " | ".join([":--" for _ in headers]) + " |\n" + \
               "\n".join("| " + " | ".join(map(str,r)) + " |" for r in rows)

    # --- PATCH #8: PASS API KEY TO NARRATIVE FUNCTION ---
    narrative = generate_narrative(api_key, MODEL_NAME, project, thesis_cfg, table_md, exec_summary_md)
    md_content = (
        f"# Biophysics Final Report\n\n**Project:** {project}\n\n"
        f"## Executive Summary (deterministic)\n{exec_summary_md}\n\n"
        f"## Run-by-Run Summary\n{table_md}\n\n"
        f"## Expert Narrative (Gemini)\n{narrative}\n"
    )

    with open("Biophysics_Final_Report.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    with open("Biophysics_Final_Report_Summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "verdict_counts": verdict_counts,
            "confidence_counts": conf_counts,
            "median_best_RMSD_A": median_rmsd,
            "thesis_evaluation": thesis_eval
        }, f, indent=2)

    logger.info("✅ Wrote final reports: Biophysics_Final_Report.md and Biophysics_Final_Report_Summary.json")

# ==============================================================================
# SECTION 3: MAIN ORCHESTRATOR
# ==============================================================================

def main():
    if not GITHUB_DATA_REPO_URL or not PROJECT_NAME:
        logger.critical("FATAL: GITHUB_DATA_REPO_URL or PROJECT_NAME is not set.")
        return

    try:
        repo_name = GITHUB_DATA_REPO_URL.split('/')[-1].replace('.git', '')
        logger.info(f"Preparing to clone data repository: {repo_name}")

        if os.path.exists(repo_name):
            logger.info(f"Removing existing directory '{repo_name}'...")
            os.system(f"rm -rf {repo_name}")

        logger.info(f"Cloning from {GITHUB_DATA_REPO_URL}...")
        clone_result = os.system(f"git clone {GITHUB_DATA_REPO_URL}")
        if clone_result != 0:
            raise RuntimeError(f"git clone failed with exit code {clone_result}")
        logger.info("✅ Repository cloned successfully.")

        project_path = os.path.join(repo_name, "Projects", PROJECT_NAME)
        root_dir = project_path
        findings_path = os.path.join(project_path, "findings.json")

        if not os.path.isdir(project_path):
            logger.critical(f"FATAL: Project directory '{PROJECT_NAME}' not found in 'Projects'. Looked for: {project_path}")
            return
        if not os.path.exists(findings_path):
            logger.critical(f"FATAL: 'findings.json' not found in project directory: {findings_path}")
            return

    except Exception as e:
        logger.critical(f"FATAL: Failed to clone and set up the data repository. Error: {e}")
        return

    # API key
    api_key = os.getenv("GEMINI_API_KEY")
    api_key_path = Path("API_KEY.txt")
    if not api_key and api_key_path.exists():
        api_key = api_key_path.read_text().strip()
    if not api_key and IN_COLAB:
        try:
            logger.info("Gemini API Key not found.")
            api_key = input("Please paste your Gemini API key here and press Enter: ")
            if api_key:
                api_key_path.write_text(api_key)
                logger.info("API Key received and saved to API_KEY.txt for this session.")
        except Exception as e:
            logger.warning(f"Could not get API key from user input: {e}")
    if not api_key:
        logger.critical("FATAL: Gemini API key is missing. The audit engine requires the API key for AI peer review.")
        return

    logger.info(f"Gemini model: {MODEL_NAME}")
    logger.info(f"Using findings: {findings_path}")
    logger.info(f"Base data root_dir: {root_dir}")

    # Model init
    if not GEMINI_AVAILABLE:
        logger.critical("FATAL: 'google-generativeai' library not installed.")
        return
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        logger.info("Successfully initialized Gemini model.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize Gemini model: {e}")
        return

    verification_report = run_verification_engine(findings_path=findings_path, root_dir=root_dir, model=model)
    if not verification_report or not verification_report.get("comprehensive_verification_report"):
        logger.critical("Verification engine failed to produce a report. Aborting.")
        return

    with open("Comprehensive_Verification_Report.json", "w", encoding="utf-8") as f:
        json.dump(verification_report, f, indent=2)
    logger.info("✅ Wrote comprehensive verification report: Comprehensive_Verification_Report.json")

    # Re-load original (unrepaired) findings for metadata (title/thesis text), but that’s safe
    with open(findings_path, "r", encoding="utf-8") as f:
        findings_doc = json.load(f)

    write_final_report(findings_doc, verification_report, api_key)

if __name__ == "__main__":
    main()
