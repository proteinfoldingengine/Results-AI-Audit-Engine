# ==============================================================================
# Unified Audit Engine (p53/MECP2-ready)
# v5.7  — Gold Standard (Final Assembly Fix)
# ==============================================================================
#
# WHAT'S NEW (v5.7):
# - BUGFIX: Corrected a NameError caused by the omission of the `evaluate_thesis`
#   function definition in the previous version.
# - This is the definitive, stable, and fully functional version.
# ==============================================================================

import os, sys, re, json, time, hashlib, logging, random, datetime, statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from PIL import Image
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
GITHUB_DATA_REPO_URL = "https://github.com/ProteinFoldingEngine/Results-AI-Audit-Engine.git"
PROJECT_NAME = "MECP2"

# --- SPEC-COMPLETE CONFIGURATIONS ---
METRIC_TOLERANCES = {
    "best_final_RMSD_A": 0.5,
    "best_final_Rg_A": 0.25,
}
STABILITY_STD_MAX = 1.0
STABILITY_SLOPE_ABS_MAX = 0.0005

# ---------------- Canonical vocab, aliases, and role handling -------------
PRIMARY_ROLES = {"comprehensive_png", "diagnostics_png", "diagnostics_csv", "raw_csv", "by_param_csv", "trajectory_pdb", "md_report"}
AUX_ROLES = {"contact_maps_npz": "aux_contact_maps", "audit_log": "aux_audit_log"}
ROLE_ALIASES = {"diagnostics": "diagnostics_csv", "md": "md_report", "markdown": "md_report", "report": "md_report"}
PREFERRED_RMSD_COLS = ["RMSD_interface","rmsd_interface","RMSD","rmsd","Backbone_RMSD","RMSD(Å)","RMSD(AA)","final_RMSD","final_rmsd"]
PREFERRED_RG_COLS = ["Rg", "rg", "Rg_A", "Rg(Å)", "Rg(AA)", "RadiusofGyration", "Radius of Gyration"]
METRIC_ALIASES = {
    "final_RMSD_A": "best_final_RMSD_A", "best_final_RMSD_A": "best_final_RMSD_A",
    "final_Rg_A": "best_final_Rg_A", "best_final_Rg_A": "best_final_Rg_A",
    "failures": "failures", "runs_count": "runs_count",
}

PROMPT_LIBRARY = {
    "md_report": """ROLE: Senior biophysicist. TASK: Extract ONLY the reported numbers from the markdown report. Use ONLY explicit numbers. SCHEMA: {"runs_count": int|null, "failures": int|null, "best_final_RMSD_A": float|null, "best_final_Rg_A": float|null} OUTPUT: JSON only. FILE_CONTENT: {FILE_CONTENT}""",
    "raw_csv": """ROLE: Data auditor. TASK: Extract the single best RMSD and Rg values from the CSV snippet. SCHEMA: {"best_final_RMSD_A": float|null, "best_final_Rg_A": float|null} OUTPUT: JSON only. CSV_CONTENT: {FILE_CONTENT}""",
    "by_param_csv": """ROLE: Computational chemist. TASK: Review the parameter-sweep CSV. Does the best RMSD value reported here align with the definitive value in the LOCAL_SUMMARY? SCHEMA: {"consistency_check": "consistent"|"inconsistent"|"cannot_determine", "notes": "string"} OUTPUT: JSON only. CSV_CONTENT: {FILE_CONTENT} LOCAL_SUMMARY: {LOCAL_SUMMARY}""",
    "diagnostics_csv": """ROLE: Quantitative analyst. TASK: Compare the timeseries CSV snippet to the LOCAL_SUMMARY calculated from its own tail. Note if the final frame value aligns with the more stable tail median. SCHEMA: {"consistency_check": "consistent"|"inconsistent"|"cannot_determine", "notes": "short explanation"} OUTPUT: JSON only. CSV_CONTENT: {FILE_CONTENT} LOCAL_SUMMARY: {LOCAL_SUMMARY}""",
    "trajectory_pdb": """ROLE: Structural biologist. TASK: Assess PDB snippet (focus on CA atoms if present). SCHEMA: {{"frames_sampled":[int,...]|null, "qualitative_compaction":"yes"|"no"|"uncertain", "notes":"string"}} OUTPUT: JSON only. PDB_SNIPPET: {FILE_CONTENT}""",
    "diagnostics_png": """ROLE: Analyst. TASK: Visually estimate the final RMSD value from the plot. SCHEMA: {"estimated_final_RMSD_A": float|null} OUTPUT: JSON only.""",
    "comprehensive_png": """ROLE: Structural biologist. TASK: Visually estimate the final RMSD value from the structure overlay plot caption or title. SCHEMA: {"estimated_final_RMSD_A": float|null} OUTPUT: JSON only."""
}

def normalize_metric_name(metric: Optional[str]) -> Optional[str]:
    return METRIC_ALIASES.get(metric, metric) if metric else metric

def normalize_role_name(role: str) -> Tuple[str, bool]:
    role = ROLE_ALIASES.get(role, role)
    if role.startswith("aux_"): return role, True
    if role in PRIMARY_ROLES: return role, False
    if role in AUX_ROLES: return AUX_ROLES[role], True
    return f"aux_{role}", True

# ==============================================================================
# SECTION 1: VERIFICATION ENGINE LOGIC
# ==============================================================================
def _safe_prompt_format(template: str, **kwargs) -> str:
    esc = template.replace("{", "{{").replace("}", "}}")
    for k in kwargs: esc = esc.replace(f"{{{{{k}}}}}", f"{{{k}}}")
    return esc.format(**kwargs)

def pick_numeric_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty: return None
    col_map = {re.sub(r'\s+', '', c).lower(): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r'\s+', '', cand).lower()
        if key in col_map:
            s = pd.to_numeric(df[col_map[key]], errors="coerce")
            if s.notna().any():
                return col_map[key]
    return None

def best_final_metric_from_timeseries(ts_df: Optional[pd.DataFrame], kind: str = "RMSD") -> Dict[str, Any]:
    out = {"min": np.nan, "median": np.nan, "final": np.nan, "stable": None}
    if ts_df is None or ts_df.empty: return out
    
    col_candidates = PREFERRED_RMSD_COLS if kind == "RMSD" else PREFERRED_RG_COLS
    col = pick_numeric_col(ts_df, col_candidates)
    if not col: return out
    
    logger.info(f"    - Using column '{col}' for {kind}")
    n = len(ts_df)
    tail = pd.to_numeric(ts_df[col], errors="coerce").dropna().tail(max(100, n // 10))
    if tail.empty: return out
    
    out.update({"min": float(tail.min()), "median": float(tail.median()), "final": float(tail.iloc[-1])})
    if len(tail) > 1:
        std = float(tail.std())
        x = np.arange(len(tail), dtype=float)
        slope = float(np.polyfit(x, tail.values, 1)[0]) if len(tail) >= 3 else 0.0
        out["stable"] = bool((std <= STABILITY_STD_MAX) and (abs(slope) <= STABILITY_SLOPE_ABS_MAX))
    else:
        out["stable"] = True
    return out

def verdict_from_constraint(value: Optional[float], op: str, target: float, tol: Optional[float] = None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)): return "INDETERMINATE"
    tol_val = float(tol) if tol is not None else 0.0
    target_val = float(target)
    if op == "<=": return "CONFIRMED" if value <= target_val + tol_val else "DEVIATION"
    if op == ">=": return "CONFIRMED" if value >= target_val - tol_val else "DEVIATION"
    if op == "==": return "CONFIRMED" if abs(value - target_val) <= tol_val else "DEVIATION"
    return "INDETERMINATE"

def _auto_repair_findings(doc: Dict[str, Any]) -> Dict[str, Any]:
    repaired = json.loads(json.dumps(doc))
    for i, run in enumerate(repaired.get("findings", []), 1):
        for a in run.get("artifacts", []):
            if (role := a.get("role")) and (new_role := normalize_role_name(role)[0]) != role:
                a["role"] = new_role
    return repaired

def perform_consistency_triage(analyses: List[Dict], definitive_metrics: Dict, tolerances: Dict, md_claims: Dict) -> Dict[str, str]:
    triage = {}
    def check(extracted, definitive, tol):
        if extracted is None or definitive is None or np.isnan(definitive): return "MISSING_DATA"
        return "CONSISTENT" if abs(float(extracted) - definitive) <= tol else "INCONSISTENT"

    def_rmsd = definitive_metrics.get("best_final_RMSD_A")
    triage["diagnostics_vs_md_rmsd"] = check(md_claims.get("best_final_RMSD_A"), def_rmsd, tolerances.get("best_final_RMSD_A", 0.5))
    def_rg = definitive_metrics.get("best_final_Rg_A")
    triage["diagnostics_vs_md_rg"] = check(md_claims.get("best_final_Rg_A"), def_rg, tolerances.get("best_final_Rg_A", 0.25))

    for analysis in analyses:
        role = analysis.get("role")
        findings = analysis.get("ai_findings", {})
        if "error" in findings: continue

        if role == "raw_csv":
            triage["diagnostics_vs_raw_rmsd"] = check(findings.get("best_final_RMSD_A"), def_rmsd, tolerances.get("best_final_RMSD_A", 0.5))
            triage["diagnostics_vs_raw_rg"] = check(findings.get("best_final_Rg_A"), def_rg, tolerances.get("best_final_Rg_A", 0.25))
        elif role in ["diagnostics_png", "comprehensive_png"]:
            label = "png_diag" if role == "diagnostics_png" else "png_comp"
            triage[f"diagnostics_vs_{label}_rmsd"] = check(findings.get("estimated_final_RMSD_A"), def_rmsd, tolerances.get("best_final_RMSD_A", 0.5))
    return triage

def safe_text_snippet(s: str, max_bytes: int, head: int = 200, tail: int = 200) -> str:
    if not s: return ""
    b = s.encode("utf-8", errors="ignore");
    if len(b) <= max_bytes: return s
    lines = s.splitlines(True)
    return "".join(lines[:head]) + "\n...TRUNCATED...\n" + "".join(lines[-tail:])

def csv_focus_minimal(df: pd.DataFrame, byte_limit: int = MAX_PAYLOAD_SIZE_BYTES) -> str:
    if df is None: return ""
    parts = [",".join(map(str, df.columns)) + "\n"]
    if not df.empty:
        if len(df) > 60:
            parts.append(df.head(30).to_csv(index=False, header=False))
            parts.append("\n...rows truncated...\n")
            parts.append(df.tail(30).to_csv(index=False, header=False))
        else:
            parts.append(df.to_csv(index=False, header=False))
    return safe_text_snippet("".join(parts), byte_limit)

def pdb_stratified_snippet(text: str, max_frames: int = 5) -> Tuple[str, List[int]]:
    if not text: return "", []
    lines = text.splitlines(True); header = [ln for ln in lines if not ln.startswith(("MODEL", "ATOM", "HETATM"))]
    models, cur, in_model = [], [], False; frames = []
    for ln in lines:
        if ln.startswith("MODEL"): in_model = True; cur = [ln]
        elif ln.startswith("ENDMDL"): cur.append(ln); models.append(cur); in_model = False
        elif in_model: cur.append(ln)
    snippet_content = ""
    if not models: snippet_content = "".join(lines[:4000])
    else:
        num = len(models)
        idx = [0] if max_frames < 2 else sorted(set([0, num - 1] + [int(num * (i / (max_frames - 1))) for i in range(1, max_frames - 1)]))
        frames = [i + 1 for i in idx]; out_lines = header + [f"REMARK SAMPLED FRAMES: {frames}\n"]
        for i in idx:
            m = models[i]; ca = [ln for ln in m if " CA " in ln]
            out_lines.extend([m[0]] + ca[:1000] + [m[-1]])
        snippet_content = "".join(out_lines)
    if len(snippet_content.encode("utf-8", errors="ignore")) > MAX_PAYLOAD_SIZE_BYTES:
        snippet_content = safe_text_snippet(snippet_content, MAX_PAYLOAD_SIZE_BYTES)
    return snippet_content, frames

def _read_csv_guard(path: Path) -> Optional[pd.DataFrame]:
    try: return pd.read_csv(path)
    except Exception as e:
        logger.warning(f"     - Could not read CSV at {path}: {e}")
        return None

def call_gemini_with_retry(model, payload, max_retries=3, timeout=300):
    for attempt in range(max_retries):
        try:
            cfg = {"temperature": 0.0, "response_mime_type": "application/json"}
            logger.info(f"     --> Attempting Gemini API call (Attempt {attempt + 1}/{max_retries})...")
            resp = model.generate_content(payload, generation_config=cfg, request_options={"timeout": timeout})
            logger.info(f"     --> API call returned.")
            txt = (resp.text or "").strip().replace("```json","").replace("```","")
            try: return json.loads(txt)
            except json.JSONDecodeError:
                logger.warning("     --> Invalid JSON detected. Attempting one repair pass.")
                repair = f"Fix JSON only:\n{txt}"
                r2 = model.generate_content(repair, generation_config=cfg)
                return json.loads((r2.text or "").strip().replace("```json","").replace("```",""))
        except Exception as e:
            logger.error(f"     --> API error: {e}", exc_info=False)
            time.sleep((2**attempt) + random.uniform(0, 1))
    return {"error": "API failed after all retries"}

def _resolve_path(root_dir: Path, source_folder: str, artifact_path: str) -> Path:
    return (Path(root_dir) / source_folder / artifact_path).resolve()

def _open_artifact(path: Path) -> Tuple[Optional[bytes], Optional[str]]:
    if not path.exists(): return None, None
    if str(path).lower().endswith((".png", ".jpg", ".jpeg")):
        with open(path, "rb") as f: return f.read(), None
    try: return None, path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        with open(path, "rb") as f: return f.read(), None

def _prepare_png_payload(raw_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.thumbnail((PNG_MAX_SIDE, PNG_MAX_SIDE))
        return img
    except Exception as e:
        logger.warning(f"     - PNG prepare failed: {e}")
        return raw_bytes
        
def run_verification_engine(findings_path: str, root_dir: str, model: Any) -> Dict[str, Any]:
    try:
        with open(findings_path, "r", encoding="utf-8") as f: raw_doc = json.load(f)
        doc = _auto_repair_findings(raw_doc)
        findings = doc.get("findings", [])
        root_path = Path(root_dir).resolve()
        logger.info(f"ENGINE: Loaded {len(findings)} runs from {Path(findings_path).name} (schema v{doc.get('schema_version')})")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to load findings file: {e}", exc_info=True); return {}

    report = []
    for finding in findings:
        run_id = str(finding.get("run_id", "UNK"))
        source_folder = finding.get("source_folder", "")
        logger.info(f"\n----- RUN {run_id}: {source_folder} -----")
        artifacts = finding.get("artifacts", []) or []

        # STAGE 1: DETERMINISTIC METRIC CALCULATION (SOT)
        diagnostics_csv_path = next((_resolve_path(root_path, source_folder, art.get("path")) for art in artifacts if normalize_role_name(art.get("role", ""))[0] == "diagnostics_csv"), None)
        ts_df = _read_csv_guard(diagnostics_csv_path) if diagnostics_csv_path and diagnostics_csv_path.exists() else None
        
        ts_rmsd = best_final_metric_from_timeseries(ts_df, kind="RMSD")
        ts_rg = best_final_metric_from_timeseries(ts_df, kind="Rg")
        key_numbers = {
            "best_final_RMSD_A": ts_rmsd["median"], "aux_tail_min_RMSD_A": ts_rmsd["min"],
            "best_final_Rg_A": ts_rg["median"],
            "tail_stable": ts_rmsd["stable"], "failures": None, "runs_count": None
        }
        if not np.isnan(key_numbers['best_final_RMSD_A']):
            logger.info(f"  - Definitive best_final_RMSD_A (tail median): {key_numbers['best_final_RMSD_A']:.2f} Å")
        
        # STAGE 2: HOLISTIC AI CROSS-CHECK & CONTEXT GATHERING
        analyses, md_report_analysis = [], {}
        logger.info("  - Starting holistic AI cross-check of all artifacts...")
        
        for art in artifacts:
            role_name, is_aux = normalize_role_name(art.get("role", ""))
            if is_aux or not art.get("path"): continue
            
            resolved_path = _resolve_path(root_path, source_folder, art.get("path"))
            if not resolved_path.exists(): continue

            if role_name not in PROMPT_LIBRARY: continue
            
            logger.info(f"    - AI analyzing: {art.get('path')} (Role: {role_name})")
            payload, an = None, {"error": "payload_preparation_skipped"}

            try:
                prompt_template = PROMPT_LIBRARY[role_name]
                if role_name == "md_report":
                    _, text = _open_artifact(resolved_path)
                    payload = _safe_prompt_format(prompt_template, FILE_CONTENT=safe_text_snippet(text or "", MAX_PAYLOAD_SIZE_BYTES))
                elif role_name in ["raw_csv", "by_param_csv", "diagnostics_csv"]:
                    df = _read_csv_guard(resolved_path)
                    local_summary_for_ai = json.dumps({"best_final_RMSD_A": key_numbers["best_final_RMSD_A"], "best_final_Rg_A": key_numbers["best_final_Rg_A"]})
                    if df is not None: payload = _safe_prompt_format(prompt_template, FILE_CONTENT=csv_focus_minimal(df), LOCAL_SUMMARY=local_summary_for_ai)
                elif role_name == "trajectory_pdb":
                    _, text = _open_artifact(resolved_path)
                    snippet, _ = pdb_stratified_snippet(text or "")
                    payload = _safe_prompt_format(prompt_template, FILE_CONTENT=snippet)
                elif role_name in ["diagnostics_png", "comprehensive_png"]:
                    raw_bytes, _ = _open_artifact(resolved_path)
                    local_summary_for_ai = json.dumps({"best_final_RMSD_A": key_numbers["best_final_RMSD_A"], "best_final_Rg_A": key_numbers["best_final_Rg_A"]})
                    if raw_bytes: payload = [_safe_prompt_format(prompt_template, LOCAL_SUMMARY=local_summary_for_ai), _prepare_png_payload(raw_bytes)]
                
                if payload:
                    an = call_gemini_with_retry(model, payload)
                    if role_name == "md_report" and "error" not in an:
                        key_numbers["failures"] = an.get("failures")
                        key_numbers["runs_count"] = an.get("runs_count")
                        md_report_analysis = an
            except Exception as e:
                an = {"error": f"Payload preparation failed: {e}"}
            analyses.append({"file": art.get("path"), "role": role_name, "ai_findings": an})
            
        # STAGE 3: CLAIM VERIFICATION LAYER
        claim_integrity = {}
        md_claims = md_report_analysis
        for metric, tolerance in METRIC_TOLERANCES.items():
            claimed_val, verified_val = md_claims.get(metric), key_numbers.get(metric)
            if claimed_val is not None:
                if verified_val is not None and not np.isnan(verified_val):
                    if abs(claimed_val - verified_val) <= tolerance: claim_integrity[metric] = "CLAIM_MATCHES_EVIDENCE"
                    else: claim_integrity[metric] = "CLAIM_CONTRADICTS_EVIDENCE"
                else: claim_integrity[metric] = "CLAIM_UNVERIFIABLE"
            else: claim_integrity[metric] = "CLAIM_NOT_FOUND"

        # STAGE 4: EXPLICIT CONSISTENCY TRIAGE
        consistency_triage = perform_consistency_triage(analyses, key_numbers, METRIC_TOLERANCES, md_claims=md_claims)

        # STAGE 5: FINAL VERDICT & REPORT SYNTHESIS
        constraints = (finding.get("canonical_claim") or {}).get("constraints", [])
        verdict, constraint_results = "INDETERMINATE", []
        if constraints:
            statuses = []
            for c in constraints:
                metric = normalize_metric_name(c.get("metric"))
                val = key_numbers.get(metric)
                status = verdict_from_constraint(val, c.get("op"), c.get("value"), c.get("tolerance"))
                statuses.append(status)
                constraint_results.append({"constraint": c, "status": status, "actual_value": val if val is not None and not (isinstance(val, float) and np.isnan(val)) else None})
            if "INDETERMINATE" in statuses and all(s != "DEVIATION" for s in statuses): verdict = "INDETERMINATE"
            elif any(s == "DEVIATION" for s in statuses): verdict = "DEVIATION"
            elif all(s == "CONFIRMED" for s in statuses): verdict = "CONFIRMED"
        
        logger.info(f"  - Final Deterministic Verdict: {verdict}")
        clean_key_numbers = {k: (v if not (isinstance(v, float) and np.isnan(v)) else None) for k, v in key_numbers.items()}
        report.append({
            "run_id": run_id,
            "biophysics_verification": { "verdict": verdict, "key_numbers": clean_key_numbers, "constraint_evaluations": constraint_results },
            "data_QA": { "claim_integrity": claim_integrity, "consistency_triage": consistency_triage, "ai_artifact_reviews": analyses }
        })
    return {"comprehensive_verification_report": report}

# ==============================================================================
# SECTION 2 & 3: REPORTING & MAIN
# ==============================================================================
def evaluate_thesis(thesis_cfg, per_run_numbers):
    if not thesis_cfg or not (cons := thesis_cfg.get("constraints")): return {"present": False}
    all_results = []
    for constraint in cons:
        agg, where, op, val = constraint.get("aggregation"), constraint.get("where", []), constraint.get("op"), constraint.get("value")
        if agg != "count_runs_meeting":
            all_results.append({"present": True, "satisfied": None, "explanation": f"Unsupported aggregator '{agg}'."}); continue
        def run_meets(kn):
            for w in where:
                m = normalize_metric_name(w.get("metric"))
                if verdict_from_constraint(kn.get(m), w.get("op"), w.get("value"), w.get("tolerance")) != "CONFIRMED":
                    return False
            return True
        count_meeting = sum(1 for kn in per_run_numbers if run_meets(kn))
        sat = verdict_from_constraint(count_meeting, op, val) == "CONFIRMED"
        all_results.append({
            "satisfied": bool(sat), "count_meeting": count_meeting, "required": {"op": op, "value": val},
            "explanation": f"{count_meeting} runs meet filter; require {op} {val}."
        })
    final_satisfied = all(r.get("satisfied", False) for r in all_results)
    explanations = " | ".join(r.get("explanation", "") for r in all_results)
    return { "present": True, "satisfied": final_satisfied, "explanation": explanations, "details": all_results }

def generate_narrative(api_key, model_name, project, thesis, runs_rows_md, exec_summary_md):
    if not api_key or not GEMINI_AVAILABLE: return "_API key not found; skipping narrative generation._"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"""ROLE: Senior structural biophysicist. Write a concise, publication-grade final assessment.
PROJECT: {project}
THESIS: {json.dumps(thesis or {}, indent=2)}
DETERMINISTIC SUMMARY: {exec_summary_md}
RUN-BY-RUN TABLE: {runs_rows_md}
STRICT INSTRUCTIONS: Be precise and non-speculative. Ground statements in the verified numbers.
"""
    try: return (model.generate_content(prompt, generation_config={"temperature":0.2}).text or "").strip()
    except Exception as e: return f"_Generated narrative unavailable (API call failed: {e})._"

def write_final_report(findings_doc, verification_report, api_key, out_md="Biophysics_Final_Report.md", out_json="Biophysics_Final_Report_Summary.json"):
    project = findings_doc.get("project","")
    thesis_cfg = (findings_doc.get("thesis") or {}).get("canonical_field_claim")
    cfg_by_id = {str(r.get("run_id")): r for r in (findings_doc.get("findings", []) or []) if "run_id" in r}
    blocks = verification_report.get("comprehensive_verification_report", [])
    headers = ["run_id", "source_folder", "verdict", "RMSD_Å", "Rg_Å", "tail_stable", "Claim Integrity", "Consistency Triage"]
    rows, verdict_counts, per_run_keynums = [], {"CONFIRMED": 0, "DEVIATION": 0, "INDETERMINATE": 0}, []

    def _rid_sort_key(row):
        try: return int(row[0])
        except (ValueError, TypeError): return float('inf')
    
    def _fmt(v, fallback="—"):
        return fallback if (v is None) else f"{v:.2f}"

    for b in blocks:
        rid = str(b.get("run_id"))
        biophys, dataqa = b.get("biophysics_verification", {}), b.get("data_QA", {})
        key_numbers = biophys.get("key_numbers", {})
        per_run_keynums.append(key_numbers)
        verdict = biophys.get("verdict", "INDETERMINATE")
        verdict_counts[verdict] += 1
        
        claim_integrity = dataqa.get("claim_integrity", {})
        claim_summary = " ".join([f"{k.split('_')[-2]}:{'✅' if 'MATCHES' in v else '❌' if 'CONTRADICTS' in v else '❓'}" for k,v in claim_integrity.items()]) or "N/A"

        triage = dataqa.get("consistency_triage", {})
        triage_summary = " ".join([f"{k.split('_')[-1].upper()}:{'✅' if 'CONSISTENT' in v else '❌' if 'INCONSISTENT' in v else '❓'}" for k,v in triage.items()]) or "N/A"

        rows.append([
            rid, cfg_by_id.get(rid, {}).get("source_folder", ""), verdict,
            _fmt(key_numbers.get('best_final_RMSD_A'), "N/A (no diag)"),
            _fmt(key_numbers.get('best_final_Rg_A'), "N/A (no diag)"),
            "✅" if key_numbers.get('tail_stable') is True else "❌" if key_numbers.get('tail_stable') is False else "—",
            claim_summary, triage_summary,
        ])
    rows.sort(key=_rid_sort_key)
    
    all_rmsds = [r.get('best_final_RMSD_A') for r in per_run_keynums if r.get('best_final_RMSD_A') is not None]
    median_rmsd = statistics.median(all_rmsds) if all_rmsds else None
    rmsd_summary_line = f"- RMSD summary (tail median): {'**%.2f Å**' % median_rmsd if median_rmsd is not None else 'N/A'}"
    
    thesis_eval = evaluate_thesis(thesis_cfg, per_run_keynums)
    exec_summary_md = (f"- Runs analyzed: **{len(blocks)}**\n"
                       f"- Verdicts — CONFIRMED: **{verdict_counts['CONFIRMED']}**, DEVIATION: **{verdict_counts['DEVIATION']}**, INDETERMINATE: **{verdict_counts['INDETERMINATE']}**\n"
                       f"{rmsd_summary_line}\n"
                       f"- Thesis check: **{'✅ satisfied' if thesis_eval.get('satisfied') else '❌ not satisfied' if thesis_eval.get('satisfied') is False else '—'}** — {thesis_eval.get('explanation','')}")
    
    table_md = "| " + " | ".join(headers) + " |\n| " + " | ".join([":--" for _ in headers]) + " |\n" + "\n".join("| " + " | ".join(map(str,r)) + " |" for r in rows)
    
    if any(k.get('best_final_RMSD_A') is not None for k in per_run_keynums):
        narrative = generate_narrative(api_key, MODEL_NAME, project, thesis_cfg, table_md, exec_summary_md)
    else:
        narrative = "_Narrative generation skipped: no runs with definitive metrics were found._"
        
    md_content = (f"# Biophysics Final Report\n\n**Project:** {project}\n\n"
                  f"## Executive Summary\n{exec_summary_md}\n\n"
                  f"## Run-by-Run Verification\n{table_md}\n\n"
                  f"## Expert Narrative (Gemini)\n{narrative}\n")
    with open(out_md, "w", encoding="utf-8") as f: f.write(md_content)
    with open(out_json, "w", encoding="utf-8") as f: json.dump({"verdict_counts": verdict_counts, "median_best_RMSD_A": median_rmsd, "thesis_evaluation": thesis_eval}, f, indent=2)
    logger.info(f"✅ Wrote final reports: {out_md} and {out_json}")

def main():
    if not GITHUB_DATA_REPO_URL or not PROJECT_NAME:
        logger.critical("FATAL: GITHUB_DATA_REPO_URL or PROJECT_NAME is not set."); return
    try:
        repo_name = GITHUB_DATA_REPO_URL.split('/')[-1].replace('.git', '')
        if os.path.exists(repo_name):
            logger.info(f"Removing existing directory '{repo_name}'...")
            os.system(f"rm -rf {repo_name}")
        logger.info(f"Cloning from {GITHUB_DATA_REPO_URL}...")
        if os.system(f"git clone {GITHUB_DATA_REPO_URL}") != 0: raise RuntimeError("git clone failed")
        logger.info("✅ Repository cloned successfully.")
        root_dir = os.path.join(repo_name, "Projects", PROJECT_NAME)
        findings_path = os.path.join(root_dir, "findings.json")
        if not os.path.exists(findings_path): raise FileNotFoundError(f"findings.json not found in {root_dir}")
    except Exception as e:
        logger.critical(f"FATAL: Failed to set up data repository: {e}"); return
    
    api_key = os.getenv("GEMINI_API_KEY") or (Path("API_KEY.txt").read_text().strip() if Path("API_KEY.txt").exists() else None)
    if not api_key and IN_COLAB:
        try: api_key = input("Please paste your Gemini API key: ")
        except: pass
    if not api_key:
        logger.critical("FATAL: Gemini API key is missing."); return
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        logger.info("Successfully initialized Gemini model.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize Gemini model: {e}"); return

    verification_report = run_verification_engine(findings_path=findings_path, root_dir=root_dir, model=model)
    with open("Comprehensive_Verification_Report.json", "w", encoding="utf-8") as f:
        json.dump(verification_report, f, indent=2)
    logger.info("✅ Wrote comprehensive verification report: Comprehensive_Verification_Report.json")
    
    with open(findings_path, "r", encoding="utf-8") as f:
        findings_doc = json.load(f)
    write_final_report(findings_doc, verification_report, api_key)

if __name__ == "__main__":
    main()
