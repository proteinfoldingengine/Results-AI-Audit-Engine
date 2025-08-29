# ==============================================================================
# Independent AI Review Engine & Manuscript Creator
# v25.0 — Final Manuscript Prompt Fix
# ==============================================================================
#
# WHAT'S NEW (v25.0):
# - FINAL BUG FIX: Corrected a critical bug in the `generate_final_manuscript`
#   function that caused the final prompt to be malformed. The function now
#   uses the robust, standard formatting method, ensuring the prompt for the
#   abstract and manuscript body is always generated correctly.
#
# Generated on: Friday, August 29, 2025 at 3:34 PM
# ==============================================================================

# ==============================================================================
# ⚙️ Step 1: Configure and Run the AI Review Engine
# ==============================================================================
#@title Engine Configuration
#@markdown ### 1. Select the AI Model to use for the analysis.
ai_model_choice = "ChatGPT"  #@param ["Gemini", "ChatGPT"]

#@markdown ---
#@markdown ### 2. Provide the necessary API Key.
#@markdown Only the key for the **selected** model is required. Leave the other blank.
gemini_api_key = ""  #@param {type:"string"}
openai_api_key = ""  #@param {type:"string"}

#@markdown ---
#@markdown ### 3. Run the Engine
#@markdown Once configured, ensure this is checked and run the cell to execute the workflow.
run_engine = True  #@param {type:"boolean"}

# ==============================================================================
# 🐍 Main Python Script - DO NOT EDIT BELOW THIS LINE
# ==============================================================================

import os
import sys
import re
import json
import time
import logging
import base64
import io
from pathlib import Path
from typing import Dict, Any, List, Tuple

# --- AI Model Libraries ---
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# --- Data Handling Libraries ---
try:
    import pandas as pd
    from PIL import Image
except ImportError:
    print("Please install required libraries: pip install pandas Pillow google-generativeai openai tiktoken")
    sys.exit(1)

# --- Colab Helpers ---
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# --------------------------- Logging -----------------------------------------
logger = logging.getLogger("independent_review_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "independent_review_engine.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

# ==============================================================================
# ⭐ TPM Governor & Smart Packing Configuration
# ==============================================================================
CHATGPT_TPM_LIMIT = 150000
CHATGPT_REQUEST_TOKEN_BUDGET = 120000
ESTIMATED_IMAGE_TOKENS = 850
RUN_LIMIT = 0
# ==============================================================================

# --------------------------- Constants & Prompts -----------------------------
GEMINI_MODEL_NAME = "gemini-1.5-pro-latest"
CHATGPT_MODEL_NAME = "gpt-4o"
GITHUB_DATA_REPO_URL = "https://github.com/ProteinFoldingEngine/Results-AI-Audit-Engine.git"
PROJECT_NAME = "MECP2"
DATA_REPO_NAME = "Results-AI-Audit-Engine"
REVIEW_REPORTS_DIR = "Holistic_BRR_Reports"

PROMPT_LIBRARY = {
    "executive_summary_synthesis": """
ROLE: Senior computational biophysicist and AI data analyst.
TASK: Perform a high-level review of the provided biophysical simulation data. Your ONLY goal is to formulate a concise **Executive Summary Narrative** and a **Final Verdict** based on how the evidence aligns with the **RUN CLAIM**.
INSTRUCTIONS:
1.  Analyze all provided data (CSV, reports, PDB, images).
2.  Synthesize your findings into a brief, expert-level narrative.
3.  Assign a final verdict: 'CONFIRMED', 'DEVIATION', or 'INDETERMINATE'.
4.  You MUST output ONLY a JSON object containing the `executive_summary` key, like the schema below. Your response must begin with {{ and end with }}.

**RUN CLAIM:** {RUN_CLAIM}

**JSON SCHEMA TO POPULATE:**
{{
  "executive_summary": {{
    "narrative": "string",
    "final_verdict": "CONFIRMED | DEVIATION | INDETERMINATE"
  }}
}}

**EVIDENCE PACKAGE FOR THE RUN:**
""",
    "holistic_review_synthesis": """
ROLE: Senior computational biophysicist and AI data analyst.
TASK: You have already performed a high-level review and formulated an Executive Summary. Your task now is to complete the full Biophysical Review Record (BRR) by providing the detailed, low-level findings that support your initial conclusion.
INSTRUCTIONS:
1.  **Use the Provided Summary:** You MUST incorporate the provided `PRE_COMPUTED_SUMMARY` verbatim into the final JSON. Do not change it.
2.  **Detail the Evidence:** For each artifact group provided in the pre-filled `artifact_analysis` section of the schema, describe what the data shows and conclude how it supports or contradicts the RUN CLAIM.
3.  **Complete the Schema:** Fill in the `findings` and `conclusion` for each pre-filled artifact group. Fill in all other remaining fields of the BRR JSON schema.
4.  **Output JSON only:** Your entire output must be a single, valid JSON object. Your response must begin with {{{{ and end with }}}}.

**PRE_COMPUTED_SUMMARY:**
{PRE_COMPUTED_SUMMARY}

**BRR v1.0 SCHEMA TO POPULATE (with pre-filled artifact groups):** {BRR_SCHEMA_TEMPLATE}

**EVIDENCE PACKAGE FOR THE RUN:**
""",
    "manuscript_synthesis": """
ROLE: You are a senior research scientist and the corresponding author for a high-impact journal publication.

TASK:
Write a comprehensive scientific manuscript that summarizes the results of a large-scale computational review. The analysis for each of the {run_count} simulation runs is provided in the input data package. Your job is to synthesize these individual findings into a cohesive, high-level narrative suitable for publication.

INSTRUCTIONS:
1.  **Title and Abstract:** Create a compelling title and a concise abstract based on the `project_metadata`. The abstract MUST incorporate the `scientific_thesis` and summarize the review's top-level results (verdict counts) and the main conclusion.
2.  **Introduction:** Use the `project_metadata` to introduce the scientific problem (e.g., Rett Syndrome, MECP2), the computational approach, and the purpose of the independent review.
3.  **Results:** Present the overall review statistics from the `overall_statistics` section. Then, create subsections to discuss the findings for logical groups of runs, summarizing the key outcomes.
4.  **Discussion:** Interpret the results. To what extent did the outcomes of the runs support the overall `scientific_thesis`? What does the review reveal about the project's success? What are the implications of any identified deviations?
5.  **Conclusion:** Briefly summarize the main takeaway from the review in relation to the project thesis.

**IMPORTANT: Do NOT create an Appendix section. Generate only the main body of the manuscript, ending with the Conclusion.**

**INPUT DATA PACKAGE (JSON):**
{input_data_package}
"""
}

# --------------------------- Helper Functions --------------------------------
def _safe_prompt_format(template: str, **kwargs) -> str: return template.format(**kwargs)
def _image_to_base64_string(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG", optimize=True, compress_level=6)
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
def story_arc_pdb_sampler(pdb_file_path: Path) -> str:
    try:
        with open(pdb_file_path, 'r', encoding='utf-8') as f: content = f.read()
        frames = [frame.strip() for frame in re.split(r'\s*ENDMDL\s*', content) if frame.strip()]
        if not frames: return ""
        if len(frames) >= 3:
            return "ENDMDL\n".join([frames[0], frames[len(frames) // 2], frames[-1]])
        else:
            return "ENDMDL\n".join(frames)
    except Exception as e: logger.error(f"Error processing PDB file {pdb_file_path.name}: {e}"); return ""
def format_appendix_from_brrs(brr_reports: list) -> str:
    appendix_parts = ["## Appendix: Detailed Per-Run Biophysical Review Records"]
    for i, report in enumerate(brr_reports):
        try:
            subject = report.get("review_metadata", {}).get("subject", f"Unknown Run {i+1}")
            verdict = report.get("executive_summary", {}).get("final_verdict", "INDETERMINATE")
            narrative = report.get("executive_summary", {}).get("narrative", "No narrative provided.")
            recommendation = report.get("synthesis_and_recommendation", "No recommendation provided.")
            artifact_groups = [f"- `{group.get('artifact_group', 'Unnamed Artifact')}`" for group in report.get("artifact_analysis", [])]
            entry = (f"### Review of: {subject}\n\n**Final Verdict:** `{verdict}`\n\n**Executive Summary Narrative:**\n> {narrative}\n\n**Synthesis and Recommendation:**\n> {recommendation}\n\n**Analyzed Artifact Groups:**\n{'\n'.join(artifact_groups) or 'None'}")
            appendix_parts.append(entry)
        except Exception as e: appendix_parts.append(f"### Error processing report for Run {i+1}\n\nCould not format appendix entry: {e}")
    return "\n\n---\n\n".join(appendix_parts)
def _resolve_path(root_dir: Path, source_folder: str, artifact_path: str) -> Path: return (Path(root_dir) / source_folder / artifact_path).resolve()
def get_brr_schema_template(pre_filled_artifacts: List[Dict[str, Any]] = None) -> str:
    schema = {"schema_name": "BiophysicalReviewRecord","schema_version": "1.0","review_metadata": {},"executive_summary": {},"artifact_analysis": [],"synthesis_and_recommendation": "string"}
    if pre_filled_artifacts: schema["artifact_analysis"] = pre_filled_artifacts
    return json.dumps(schema, indent=2)

def call_gemini_api(model_client, payload, expect_json=True, max_retries=3) -> Tuple[Any, Dict[str, int]]:
    for attempt in range(max_retries):
        try:
            cfg = {"temperature": 0.2 if expect_json else 0.4};
            if expect_json: cfg["response_mime_type"] = "application/json"
            response = model_client.generate_content(payload, generation_config=cfg, request_options={"timeout": 900})
            text = response.text.strip().replace("```json", "").replace("```", "")
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            return (json.loads(text) if expect_json else text), usage
        except Exception as e: logger.error(f"Gemini API error on attempt {attempt + 1}: {e}"); time.sleep(5 * (attempt + 1))
    return ({"error": "API call failed after multiple retries."} if expect_json else "# Manuscript Generation Failed"), {}

def call_chatgpt_api(model_client, payload, expect_json=True, max_retries=3) -> Tuple[Any, Dict[str, int]]:
    messages = [{"role": "user", "content": []}]
    for item in payload:
        if isinstance(item, str): messages[0]["content"].append({"type": "text", "text": item})
        elif isinstance(item, Image.Image): messages[0]["content"].append({"type": "image_url", "image_url": {"url": _image_to_base64_string(item)}})
    for attempt in range(max_retries):
        try:
            logger.info(f"      --> Attempting OpenAI API call (Attempt {attempt + 1}/{max_retries})...")
            request_params = {"model": CHATGPT_MODEL_NAME, "messages": messages, "temperature": 0.2 if expect_json else 0.4, "max_tokens": 4096}
            if expect_json: request_params["response_format"] = {"type": "json_object"}
            response = model_client.chat.completions.create(**request_params)
            content = response.choices[0].message.content
            usage = {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
            return (json.loads(content) if expect_json else content.strip()), usage
        except Exception as e: logger.error(f"OpenAI API error on attempt {attempt + 1}: {e}"); time.sleep(5 * (attempt + 1))
    return ({"error": "API call failed after multiple retries."} if expect_json else "# Manuscript Generation Failed"), {}

def run_holistic_review_synthesis(findings_doc: Dict, root_dir: Path, model_client: Any, api_call_function: callable, is_chatgpt: bool):
    project_thesis = findings_doc.get("thesis", {}).get("canonical_field_claim", {}).get("statement", "No project thesis provided.")
    runs = findings_doc.get("findings", [])
    if RUN_LIMIT > 0:
        logger.warning(f"Processing a maximum of {RUN_LIMIT} run(s) as per configuration.")
        runs = runs[:RUN_LIMIT]
    
    tpm_window_start_time = time.time()
    tokens_used_in_window = 0
    tpm_threshold = CHATGPT_TPM_LIMIT * 0.95
    encoder = tiktoken.encoding_for_model(CHATGPT_MODEL_NAME) if is_chatgpt and TIKTOKEN_AVAILABLE else None
    if is_chatgpt and not encoder:
        logger.error("FATAL: `tiktoken` library is required for ChatGPT. Please `pip install tiktoken`."); return

    output_dir = Path(REVIEW_REPORTS_DIR); output_dir.mkdir(exist_ok=True)
    logger.info(f"PHASE 1: Starting Holistic Review Synthesis for {len(runs)} run(s).")
    priority_roles = ["md_report", "trajectory_pdb", "comprehensive_png", "diagnostics_png", "diagnostics_csv", "by_param_csv", "raw_csv"]

    for run_index, run in enumerate(runs):
        run_id = str(run.get("run_id", "UNK")); source_folder = run.get("source_folder", "N/A")
        logger.info(f"\n----- Synthesizing BRR for Run {run_id} ({run_index + 1}/{len(runs)}) -----")
        run_claim = run.get("canonical_claim", {}).get("statement", "No specific claim provided for this run.")
        
        logger.info("  - Step 1: Assembling and packing evidence payload...")
        evidence_payload, included_artifacts_for_schema, prompt_tokens_for_this_run = [], [], 0
        sorted_artifacts = sorted(run.get("artifacts", []), key=lambda x: priority_roles.index(x.get("role")) if x.get("role") in priority_roles else 99)
        base_summary_prompt_cost = len(encoder.encode(PROMPT_LIBRARY["executive_summary_synthesis"])) if is_chatgpt else 0
        base_full_prompt_cost = len(encoder.encode(PROMPT_LIBRARY["holistic_review_synthesis"])) if is_chatgpt else 0
        prompt_tokens_for_this_run = max(base_summary_prompt_cost, base_full_prompt_cost)

        for art in sorted_artifacts:
            role, path = art.get("role"), _resolve_path(root_dir, source_folder, art.get("path", ""))
            if not path.exists(): continue
            try:
                content, estimated_cost = None, 0
                if role in ["comprehensive_png", "diagnostics_png"]:
                    with open(path, "rb") as f: content = Image.open(f); content.thumbnail((1024, 1024))
                    estimated_cost = ESTIMATED_IMAGE_TOKENS if is_chatgpt else 0
                else:
                    content_str = ""
                    if role == "diagnostics_csv": content_str = f"{(pd.read_csv(path).head(5).to_csv(index=False))}\n...\n{(pd.read_csv(path).tail(20).to_csv(index=False))}"
                    elif role == "trajectory_pdb": content_str = story_arc_pdb_sampler(path)
                    else: content_str = path.read_text(encoding='utf-8', errors='ignore')
                    if not content_str: continue
                    content = f"\n--- Snippet from: {path.name} ---\n```\n{content_str}\n```"
                    estimated_cost = len(encoder.encode(content)) if is_chatgpt else 0
                
                if is_chatgpt and (prompt_tokens_for_this_run + estimated_cost > CHATGPT_REQUEST_TOKEN_BUDGET):
                    logger.warning(f"  - SKIPPING BY REQUEST BUDGET ({role}): '{path.name}' (Est. cost: {estimated_cost})")
                    continue
                if role in ["comprehensive_png", "diagnostics_png"]: evidence_payload.extend([f"\n--- Image file: {path.name} ---", content])
                else: evidence_payload.append(content)
                included_artifacts_for_schema.append({"artifact_group": path.name, "findings": [], "conclusion": ""})
                prompt_tokens_for_this_run += estimated_cost
                logger.info(f"  - ATTACHING ({role}): '{path.name}' (Request tokens so far: {prompt_tokens_for_this_run if is_chatgpt else 'N/A'})")
            except Exception as e: logger.warning(f"  - FAILED TO ATTACH: {path.name}: {e}")

        estimated_total_cost_for_run = prompt_tokens_for_this_run * 2
        if is_chatgpt:
            elapsed = time.time() - tpm_window_start_time
            if elapsed > 60:
                logger.info(f"TPM Governor: Resetting token window ({elapsed:.1f}s elapsed).")
                tpm_window_start_time = time.time(); tokens_used_in_window = 0
            if (tokens_used_in_window + estimated_total_cost_for_run) > (CHATGPT_TPM_LIMIT * 0.95):
                wait_time = max(0, 60.0 - elapsed) + 1
                logger.warning(f"TPM Governor: Predicted breach. Pausing for {wait_time:.1f}s to reset window.")
                time.sleep(wait_time)
                tpm_window_start_time = time.time(); tokens_used_in_window = 0
        
        logger.info("  - Step 2: Generating executive summary...")
        summary_prompt_text = _safe_prompt_format(PROMPT_LIBRARY["executive_summary_synthesis"], RUN_CLAIM=run_claim)
        summary_payload = [summary_prompt_text] + evidence_payload
        summary_json, usage1 = api_call_function(model_client, summary_payload, expect_json=True)
        if is_chatgpt and usage1: tokens_used_in_window += usage1.get("prompt_tokens", 0) + usage1.get("completion_tokens", 0)
        if not isinstance(summary_json, dict) or "executive_summary" not in summary_json:
            logger.error(f"  - ❌ FAILED to generate a valid executive summary for Run {run_id}. Response: {summary_json}"); continue
        logger.info(f"  - ✅ Successfully generated executive summary. Verdict: {summary_json['executive_summary'].get('final_verdict')}")
        
        logger.info("  - Step 3: Generating full BRR with detailed findings...")
        dynamic_schema = get_brr_schema_template(pre_filled_artifacts=included_artifacts_for_schema)
        full_report_prompt = _safe_prompt_format(PROMPT_LIBRARY["holistic_review_synthesis"], PRE_COMPUTED_SUMMARY=json.dumps(summary_json, indent=2), BRR_SCHEMA_TEMPLATE=dynamic_schema)
        full_report_payload = [full_report_prompt] + evidence_payload
        final_brr_json, usage2 = api_call_function(model_client, full_report_payload, expect_json=True)
        if is_chatgpt and usage2: tokens_used_in_window += usage2.get("prompt_tokens", 0) + usage2.get("completion_tokens", 0)
        if isinstance(final_brr_json, dict) and "error" in final_brr_json:
            logger.error(f"  - ❌ FAILED to synthesize full BRR for Run {run_id}. API Error: {final_brr_json['error']}"); continue

        output_path = output_dir / f"AI_BRR_Report_Run_{run_id}.json"
        with open(output_path, "w", encoding="utf-8") as f: json.dump(final_brr_json, f, indent=2)
        logger.info(f"  - ✅ Successfully synthesized and saved full BRR report to {output_path}")

def generate_final_manuscript(model_client: Any, findings_doc: Dict, api_call_function: callable):
    logger.info("\nPHASE 2: Starting Final Manuscript Synthesis.")
    brr_dir = Path(REVIEW_REPORTS_DIR);
    if not brr_dir.exists() or not any(brr_dir.iterdir()): logger.critical(f"FATAL: Cannot generate manuscript."); return
    brr_reports = [json.load(open(p, "r", encoding="utf-8")) for p in sorted(list(brr_dir.glob("AI_BRR_Report_Run_*.json")))]
    verdict_counts = {"CONFIRMED": 0, "DEVIATION": 0, "INDETERMINATE": 0}
    for r in brr_reports: verdict_counts[r.get("executive_summary", {}).get("final_verdict", "INDETERMINATE")] += 1
    project_thesis = findings_doc.get("thesis", {}).get("canonical_field_claim", {}).get("statement", "N/A")
    input_data = {"project_metadata": {"project_name": findings_doc.get("project", ""),"scientific_thesis": project_thesis},"overall_statistics": {"total_runs_reviewed": len(brr_reports),"verdict_counts": verdict_counts}}
    
    # NEW (v25.0): Use the robust _safe_prompt_format function
    prompt = _safe_prompt_format(
        PROMPT_LIBRARY["manuscript_synthesis"],
        run_count=len(brr_reports),
        input_data_package=json.dumps(input_data, indent=2)
    )
    
    manuscript_body, _ = api_call_function(model_client, [prompt], expect_json=False)
    with open("Final_Manuscript.md", "w", encoding="utf-8") as f:
        f.write(manuscript_body); f.write("\n\n"); f.write(format_appendix_from_brrs(brr_reports))
    logger.info(f"✅ Final manuscript with full appendix successfully written to: Final_Manuscript.md")

def main():
    logger.info(f"Starting engine with '{ai_model_choice}' model.")
    api_key, model_client, api_call_function = None, None, None
    is_chatgpt = (ai_model_choice == "ChatGPT")
    if is_chatgpt:
        if not OPENAI_AVAILABLE: logger.critical("FATAL: 'openai' not installed."); return
        api_key = openai_api_key
        if not api_key: logger.critical("FATAL: OpenAI API Key not provided."); return
        try: model_client = openai.OpenAI(api_key=api_key); api_call_function = call_chatgpt_api; logger.info("Successfully initialized OpenAI model.")
        except Exception as e: logger.critical(f"FATAL: Failed to initialize OpenAI model: {e}"); return
    else: # Gemini
        if not GEMINI_AVAILABLE: logger.critical("FATAL: 'google-generativeai' not installed."); return
        api_key = gemini_api_key
        if not api_key: logger.critical("FATAL: Gemini API Key not provided."); return
        try: genai.configure(api_key=api_key); model_client = genai.GenerativeModel(GEMINI_MODEL_NAME); api_call_function = call_gemini_api; logger.info("Successfully initialized Gemini model.")
        except Exception as e: logger.critical(f"FATAL: Failed to initialize Gemini model: {e}"); return
    try:
        if not os.path.exists(DATA_REPO_NAME):
            import subprocess
            logger.info(f"Cloning data repository from {GITHUB_DATA_REPO_URL}...")
            result = subprocess.run(["git", "clone", GITHUB_DATA_REPO_URL], capture_output=True, text=True, check=False)
            if result.returncode != 0: raise RuntimeError(f"Git clone failed: {result.stderr}")
        root_dir = Path(DATA_REPO_NAME) / "Projects" / PROJECT_NAME
        findings_path = root_dir / "findings.json"
        if not findings_path.exists(): raise FileNotFoundError(f"FATAL: Main findings file not found at {findings_path}")
        with open(findings_path, "r", encoding="utf-8") as f: findings_doc = json.load(f)
    except Exception as e: logger.critical(f"FATAL: Failed to set up data repository or load findings: {e}"); return
    
    run_holistic_review_synthesis(findings_doc, root_dir, model_client, api_call_function, is_chatgpt)
    
    if RUN_LIMIT == 0:
        generate_final_manuscript(model_client, findings_doc, api_call_function)
    else:
        logger.info(f"\nSkipping final manuscript generation because a run limit of {RUN_LIMIT} is active.")
    logger.info("\nEngine finished. All tasks complete.")

# --- SCRIPT EXECUTION TRIGGER ---
if 'run_engine' in globals() and run_engine:
    main()
elif 'run_engine' in globals() and not run_engine:
    print("Engine execution skipped. Please check the 'run_engine' box and re-run the cell to start.")
