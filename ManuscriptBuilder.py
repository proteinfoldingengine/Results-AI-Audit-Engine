# ==============================================================================
# Independent AI Review Engine & Manuscript Creator
# v16.1 — Robust Appendix Generation
# ==============================================================================
#
# WHAT'S NEW (v16.1 - Patched):
# - AI TASK FOCUSED: The AI is now only responsible for writing the creative
#   manuscript body (Title to Conclusion) based on high-level stats.
# - DETERMINISTIC APPENDIX: A new Python function, `format_appendix_from_brrs`,
#   now parses the JSON reports and builds a detailed, fully-controlled appendix.
# - RELIABLE WORKFLOW: The script combines the AI-generated body and the
#   Python-generated appendix in a final step, guaranteeing a perfect result.
# ==============================================================================

import os
import sys
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

# Gemini
try:
    import google.generativeai as genai
    import google.api_core.exceptions
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Pandas, Numpy, Pillow for data handling
try:
    import pandas as pd
    import numpy as np
    from PIL import Image
except ImportError:
    print("Please install required libraries: pip install pandas numpy Pillow google-generativeai")
    sys.exit(1)

# (Colab helpers)
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
    fh = logging.FileHandler("independent_review_engine.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

# --------------------------- Constants & Prompts -----------------------------
MODEL_NAME = "gemini-1.5-pro-latest"
GITHUB_DATA_REPO_URL = "https://github.com/ProteinFoldingEngine/Results-AI-Audit-Engine.git"
PROJECT_NAME = "MECP2"
DATA_REPO_NAME = "Results-AI-Audit-Engine"
REVIEW_REPORTS_DIR = "Holistic_BRR_Reports" # BRR = Biophysical Review Record
PNG_MAX_SIDE = 1600

PROMPT_LIBRARY = {
    "holistic_review_synthesis": """
ROLE: Senior computational biophysicist and AI data analyst.

TASK:
Perform an independent, holistic review of all provided data artifacts for a single molecular dynamics simulation run. Your goal is to determine how the evidence aligns with the specific **RUN CLAIM**, evaluated in the context of the overarching **PROJECT THESIS**. Synthesize your findings into a complete Biophysical Review Record (BRR) v1.0 JSON object.

INSTRUCTIONS:
1.  **Critically Evaluate Evidence:** Analyze all provided data (CSV, reports, PDB, images).
2.  **Synthesize Findings:** For each artifact, describe what it shows and conclude how it supports or contradicts the **RUN CLAIM**.
3.  **Formulate a Final Verdict:** Based on your holistic analysis of the evidence against the specific **RUN CLAIM**, assign a final verdict.
    -   'CONFIRMED': The collective evidence strongly and consistently supports the **RUN CLAIM**.
    -   'DEVIATION': The collective evidence shows significant contradictions or fails to support the **RUN CLAIM**.
    -   'INDETERMINATE': Crucial data is missing, making a confident conclusion about the **RUN CLAIM** impossible.
4.  **Populate the BRR Schema:** Fill in every field of the provided JSON schema with your concise, expert-level conclusions.
5.  **Output JSON only:** Your entire output must be a single, valid JSON object.

**OVERARCHING PROJECT THESIS:**
{PROJECT_THESIS}

**SPECIFIC CLAIM FOR THIS RUN:**
{RUN_CLAIM}

**BRR v1.0 SCHEMA TO POPULATE:**
{BRR_SCHEMA_TEMPLATE}

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
def _safe_prompt_format(template: str, **kwargs) -> str:
    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", str(value))
    return template

def story_arc_pdb_sampler(pdb_file_path: Path) -> str:
    """Reads a multi-frame PDB trajectory file and extracts a representative sample."""
    logger.info(f"    - Reading trajectory file for sampling: {pdb_file_path.name}")
    try:
        with open(pdb_file_path, 'r', encoding='utf-8') as f: content = f.read()
        frames = re.split(r'\s*ENDMDL\s*', content)
        frames = [frame.strip() for frame in frames if frame.strip()]
        total_frames = len(frames)
        if total_frames == 0:
            logger.warning("    - No frames found in PDB file."); return ""
        logger.info(f"    - Found {total_frames} frames in trajectory.")
        if total_frames < 3:
            sampled_frames = frames
            logger.info("    - Attaching all frames (fewer than 3 found).")
        else:
            sampled_frames = [frames[0], frames[total_frames // 2], frames[-1]]
            logger.info("    - Sampling first, middle, and last frames.")
        return "ENDMDL\n".join(sampled_frames) + "ENDMDL\n"
    except Exception as e:
        logger.error(f"    - Error processing PDB file {pdb_file_path.name}: {e}"); return ""

def format_appendix_from_brrs(brr_reports: list) -> str:
    """
    Parses a list of BRR JSON reports and builds a detailed, well-formatted
    Markdown appendix with specific fields.
    """
    logger.info("  - Building custom appendix from BRR reports...")
    appendix_parts = ["## Appendix: Detailed Per-Run Biophysical Review Records"]

    for i, report in enumerate(brr_reports):
        try:
            # --- Safely extract all the desired values using .get() ---
            subject = report.get("review_metadata", {}).get("subject", f"Unknown Run {i+1}")
            verdict = report.get("executive_summary", {}).get("final_verdict", "INDETERMINATE")
            narrative = report.get("executive_summary", {}).get("narrative", "No narrative provided.")
            recommendation = report.get("synthesis_and_recommendation", "No recommendation provided.")
            
            # Extract the list of artifact groups that were analyzed
            artifact_analysis = report.get("artifact_analysis", [])
            artifact_groups = [
                f"- `{group.get('artifact_group', 'Unnamed Artifact')}`"
                for group in artifact_analysis
            ]
            artifacts_list_str = "\n".join(artifact_groups) if artifact_groups else "None"

            # --- Assemble the Markdown entry for this run ---
            entry = (
                f"### Review of: {subject}\n\n"
                f"**Final Verdict:** `{verdict}`\n\n"
                f"**Executive Summary Narrative:**\n"
                f"> {narrative}\n\n"
                f"**Synthesis and Recommendation:**\n"
                f"> {recommendation}\n\n"
                f"**Analyzed Artifact Groups:**\n"
                f"{artifacts_list_str}"
            )
            appendix_parts.append(entry)

        except Exception as e:
            appendix_parts.append(f"### Error processing report for Run {i+1}\n\n"
                                  f"Could not format appendix entry: {e}")
            
    return "\n\n---\n\n".join(appendix_parts)


def _resolve_path(root_dir: Path, source_folder: str, artifact_path: str) -> Path:
    return (Path(root_dir) / source_folder / artifact_path).resolve()

def call_gemini_api(model, payload, expect_json=True, max_retries=3):
    for attempt in range(max_retries):
        try:
            cfg = {"temperature": 0.2 if expect_json else 0.4}
            if expect_json: cfg["response_mime_type"] = "application/json"
            logger.info(f"    --> Attempting Gemini API call (Attempt {attempt + 1}/{max_retries})...")
            response = model.generate_content(payload, generation_config=cfg, request_options={"timeout": 900})
            logger.info("    --> API call successful.")
            if expect_json:
                text = response.text.strip().replace("```json", "").replace("```", "")
                return json.loads(text)
            else:
                return response.text.strip()
        except Exception as e:
            logger.error(f"    --> API error on attempt {attempt + 1}: {e}")
            time.sleep(5 * (attempt + 1))
    return {"error": "API call failed after multiple retries."} if expect_json else "# Manuscript Generation Failed"

def get_brr_schema_template() -> str:
    schema = {
      "schema_name": "BiophysicalReviewRecord", "schema_version": "1.0",
      "review_metadata": { "title": "string", "reviewer": "Dr. Gemini, Senior Biophysicist", "date": "string", "subject": "string" },
      "executive_summary": { "narrative": "string", "quantitative_analysis_summary": "string", "final_verdict": "CONFIRMED | DEVIATION | INDETERMINATE" },
      "artifact_analysis": [
        { "artifact_group": "diagnostics_timeseries.csv", "findings": [{ "title": "string", "details": "string" }], "conclusion": "string" },
        { "artifact_group": "md_report & raw_csv", "findings": [{ "title": "string", "details": "string" }], "conclusion": "string" },
        { "artifact_group": "Visual & Mechanistic Evidence (.png, .pdb, .npz)", "findings": [{ "title": "string", "details": "string" }], "conclusion": "string" }
      ], "synthesis_and_recommendation": "string"
    }
    return json.dumps(schema, indent=2)

# ==============================================================================
# PHASE 1: HOLISTIC REVIEW SYNTHESIS
# ==============================================================================
def run_holistic_review_synthesis(findings_doc: Dict, root_dir: Path, model: Any):
    project_thesis = findings_doc.get("thesis", {}).get("canonical_field_claim", {}).get("statement", "No project thesis provided.")
    logger.info(f"Project Thesis Loaded: '{project_thesis}'")

    runs = findings_doc.get("findings", [])
    output_dir = Path(REVIEW_REPORTS_DIR)
    output_dir.mkdir(exist_ok=True)
    logger.info(f"PHASE 1: Starting Holistic Review Synthesis for {len(runs)} runs.")
    logger.info(f"AI-synthesized BRR reports will be saved to '{output_dir.resolve()}'")

    for run in runs:
        run_id = str(run.get("run_id", "UNK")); source_folder = run.get("source_folder", "N/A")
        logger.info(f"\n----- Synthesizing BRR for Run {run_id}: {source_folder} -----")
        
        run_claim = run.get("canonical_claim", {}).get("statement", "No specific claim provided for this run.")
        logger.info(f"  - Run-specific Claim: '{run_claim}'")

        artifacts = run.get("artifacts", []); prompt_payload = []
        
        for art in artifacts:
            role = art.get("role"); path = _resolve_path(root_dir, source_folder, art.get("path", ""))
            
            if not path.exists():
                logger.warning(f"  - SKIPPING: Artifact file not found at path: {path}")
                continue
            
            try:
                if role == "diagnostics_csv":
                    df = pd.read_csv(path)
                    csv_snippet = f"{df.head(5).to_csv(index=False)}\n...\n{df.tail(20).to_csv(index=False)}"
                    prompt_payload.append(f"\n--- Snippet from: diagnostics_timeseries.csv ---\n```csv\n{csv_snippet}\n```")
                    logger.info(f"  - ATTACHING: CSV artifact '{path.name}'")
                
                elif role in ["md_report", "markdown", "report", "raw_csv", "by_param_csv"]:
                    prompt_payload.append(f"\n--- Snippet from: {path.name} ---\n```\n{path.read_text(encoding='utf-8', errors='ignore')[:10000]}\n```")
                    logger.info(f"  - ATTACHING: Text artifact '{path.name}'")

                elif role == "trajectory_pdb":
                    sampled_pdb = story_arc_pdb_sampler(path)
                    if sampled_pdb:
                        prompt_payload.append(f"\n--- Sampled PDB Trajectory (First, Middle, Last Frames) from: {path.name} ---\n```pdb\n{sampled_pdb}\n```")
                        logger.info(f"  - ATTACHING: Sampled PDB artifact '{path.name}'")
                    else:
                        logger.warning(f"  - SKIPPING: PDB file '{path.name}' was empty or could not be sampled.")
                
                elif role in ["comprehensive_png", "diagnostics_png"]:
                    # Using an io.BytesIO buffer is more memory efficient for large images
                    with open(path, "rb") as image_file:
                        img = Image.open(image_file)
                        img.thumbnail((PNG_MAX_SIDE, PNG_MAX_SIDE))
                        prompt_payload.append(f"\n--- Image file: {path.name} ---"); prompt_payload.append(img)
                        logger.info(f"  - ATTACHING: Image artifact '{path.name}'")
            
            except Exception as e:
                logger.warning(f"  - SKIPPING: Could not process artifact {path.name}: {e}")
        
        main_instruction = _safe_prompt_format(
            PROMPT_LIBRARY["holistic_review_synthesis"],
            PROJECT_THESIS=project_thesis,
            RUN_CLAIM=run_claim,
            BRR_SCHEMA_TEMPLATE=get_brr_schema_template()
        )
        prompt_payload.insert(0, main_instruction)
        
        ai_generated_brr = call_gemini_api(model, prompt_payload)
        output_path = output_dir / f"AI_BRR_Report_Run_{run_id}.json"
        with open(output_path, "w", encoding="utf-8") as f: json.dump(ai_generated_brr, f, indent=2)
        logger.info(f"  - ✅ Successfully synthesized and saved BRR report to {output_path}")

# ==============================================================================
# PHASE 2: FINAL MANUSCRIPT SYNTHESIS
# ==============================================================================
def generate_final_manuscript(model: Any, findings_doc: Dict):
    logger.info("\nPHASE 2: Starting Final Manuscript Synthesis.")
    brr_dir = Path(REVIEW_REPORTS_DIR)
    if not brr_dir.exists() or not any(brr_dir.iterdir()):
        logger.critical(f"FATAL: Cannot generate manuscript. The '{brr_dir}' directory is missing or empty.")
        return
        
    brr_reports = []
    for report_path in sorted(list(brr_dir.glob("AI_BRR_Report_Run_*.json"))):
        try:
            with open(report_path, "r", encoding="utf-8") as f: brr_reports.append(json.load(f))
        except Exception as e: logger.warning(f"Could not load or parse BRR report {report_path}: {e}")
    logger.info(f"  - Loaded {len(brr_reports)} BRR reports for final synthesis.")

    verdict_counts = {"CONFIRMED": 0, "DEVIATION": 0, "INDETERMINATE": 0}
    for report in brr_reports:
        verdict = report.get("executive_summary", {}).get("final_verdict", "INDETERMINATE")
        verdict_counts[verdict] += 1
        
    project_thesis = findings_doc.get("thesis", {}).get("canonical_field_claim", {}).get("statement", "N/A")
    
    # --- STEP 1: Generate the full appendix text in Python ---
    appendix_text = format_appendix_from_brrs(brr_reports)
    logger.info("  - Successfully pre-formatted the full appendix text.")

    # --- STEP 2: Prepare a smaller data package for the AI (NO APPENDIX) ---
    input_data_package = {
        "project_metadata": {
            "project_name": findings_doc.get("project", "Unknown Project"),
            "scientific_thesis": project_thesis
        },
        "overall_statistics": { "total_runs_reviewed": len(brr_reports), "verdict_counts": verdict_counts }
    }
    
    # --- STEP 3: Call the AI to write ONLY the main body of the manuscript ---
    prompt = _safe_prompt_format(
        PROMPT_LIBRARY["manuscript_synthesis"],
        run_count=len(brr_reports),
        input_data_package=json.dumps(input_data_package, indent=2)
    )
    
    logger.info("  - Requesting AI to generate the main manuscript body...")
    manuscript_body = call_gemini_api(model, prompt, expect_json=False)
    
    # --- STEP 4: Combine the AI body and Python-generated appendix into one file ---
    output_path = Path("Final_Manuscript.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(manuscript_body)
        f.write("\n\n") # Add spacing
        f.write(appendix_text) # Append the full, verbatim appendix
        
    logger.info(f"✅ Final manuscript with full appendix successfully written to: {output_path}")


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def main():
    logger.info(f"Starting Independent AI Review Engine (v16.1 - Patched)...")
    try:
        if not os.path.exists(DATA_REPO_NAME):
            logger.info(f"Cloning data repository from {GITHUB_DATA_REPO_URL}...")
            # Using subprocess for better error handling than os.system
            import subprocess
            result = subprocess.run(["git", "clone", GITHUB_DATA_REPO_URL], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Git clone failed: {result.stderr}")
                raise RuntimeError("git clone failed.")
            logger.info("✅ Repository cloned successfully.")
        else:
            logger.info(f"✅ Data repository '{DATA_REPO_NAME}' already exists.")
        
        root_dir = Path(DATA_REPO_NAME) / "Projects" / PROJECT_NAME
        findings_path = root_dir / "findings.json"
        if not findings_path.exists():
            raise FileNotFoundError(f"FATAL: The main findings file was not found at {findings_path}")
        
        logger.info(f"Loading findings from: {findings_path}")
        with open(findings_path, "r", encoding="utf-8") as f:
            findings_doc = json.load(f)

    except Exception as e:
        logger.critical(f"FATAL: Failed to set up data repository or load findings: {e}")
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        key_path = Path("API_KEY.txt")
        if key_path.exists():
            api_key = key_path.read_text().strip()
    if not api_key and IN_COLAB:
        try:
            from google.colab import userdata
            api_key = userdata.get('GEMINI_API_KEY')
        except (ImportError, ModuleNotFoundError):
            pass # Not in a Colab environment with userdata
        except Exception as e:
            logger.warning(f"Could not retrieve key from Colab secrets: {e}")
            
    if not api_key:
        try:
            api_key = input("GEMINI_API_KEY not found. Please paste your Gemini API key: ")
        except EOFError:
            logger.critical("FATAL: Could not read API key from input.")
            return

    if not api_key:
        logger.critical("FATAL: Gemini API key is missing. Please set the GEMINI_API_KEY environment variable or create an API_KEY.txt file.")
        return

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        logger.info("Successfully initialized Gemini model.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize Gemini model: {e}"); return
    
    # --- Execute the Two-Phase Workflow ---
    run_holistic_review_synthesis(findings_doc, root_dir, model)
    generate_final_manuscript(model, findings_doc)
    
    logger.info("\nEngine finished. All tasks complete.")

if __name__ == "__main__":
    main()
