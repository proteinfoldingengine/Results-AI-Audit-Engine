White-Box Protein Simulation Audit Engine
An open-source audit engine for verifying protein simulation results. It uses a JSON manifest to define scientific claims and deterministically validates them against raw data. An integrated AI peer review provides qualitative analysis, ensuring a fully transparent and reproducible "white-box" assessment of any simulation campaign.

Key Features
Deterministic Verification: Uses a machine-readable "science contract" (findings.json) to numerically validate scientific claims against raw data artifacts.

AI-Powered Peer Review: Leverages a large language model to perform qualitative assessments of data (e.g., interpreting plots, describing structures) that complement the quantitative checks.

"White-Box" Transparency: The entire process is open-source. You get the code, the data, and the claims, allowing for complete reproducibility.

Automated Reporting: Generates a final, human-readable Markdown report summarizing the deterministic verdicts and the AI's narrative assessment.

How It Works
The engine follows a simple, three-step "white-box" workflow:

The Science Contract (findings.json): You define every experiment in a JSON manifest. Each run has a specific scientific purpose, a human-readable claim, and a set of machine-testable constraints (e.g., best_final_RMSD_A <= 20.0).

The Unified Audit Engine (unified_audit_engine.py): The engine parses the contract, finds all the specified data files (CSVs, PDBs, images), and performs two parallel analyses:

Deterministic Checks: It runs local calculations on the data to confirm or refute the numerical constraints.

AI Peer Review: It sends the artifacts to the Gemini API for qualitative interpretation.

The Final Report (Biophysics_Final_Report.md): The engine combines the results into a final report containing a summary table with the deterministic verdicts and a narrative written by the AI, providing a complete picture of the findings.

Project Structure
Your project should be organized with the following structure for the engine to work correctly:

your-project-name/
|
├── unified_audit_engine.py       # The main engine script
├── findings.json                 # Your authoritative science contract
├── README.md                     # This file
├── .gitignore                    # Tells Git to ignore private files
|
└── data/                         # A dedicated folder for all simulation data
    |
    ├── your_run_folder_1/
    │   ├── best_run_details/
    │   │   └── ... (png, csv, pdb files)
    │   └── ... (raw.csv, by_param.csv, report.md)
    |
    └── ... (all other run folders)

Getting Started
1. Set Up the Repository

Follow the instructions in the github_setup_guide artifact to structure your project, initialize Git LFS for large files, and push your repository to GitHub.

2. Running in Google Colab (Recommended)

This is the easiest way to run the engine, as it handles the environment setup and data access seamlessly.

Colab Notebook Setup Code:

# --------------------------------------------------------------------------
# SETUP: CLONE THE GITHUB REPOSITORY
# --------------------------------------------------------------------------
import os

# Replace with your GitHub username and repository name
GITHUB_USERNAME = "YOUR_USERNAME"
REPO_NAME = "your-project-name"

# Clone the repository
!git clone https://github.com/{GITHUB_USERNAME}/{REPO_NAME}.git

# Navigate into the repository directory
%cd {REPO_NAME}

# Install Git LFS and pull the large data files
!git lfs install
!git lfs pull

print("\n✅ Setup complete. Repository and data are ready.")
print("Current working directory:", os.getcwd())
# --------------------------------------------------------------------------

Usage
Add Your API Key: After running the setup in Colab, upload a file named API_KEY.txt containing your Gemini API key. The engine will not run without it.

Execute the Engine: Run the following command in a new Colab cell:

!python unified_audit_engine.py

Review the Output: The engine will process all runs defined in findings.json. Upon completion, three new files will be created in your project directory:

Comprehensive_Verification_Report.json: The detailed, machine-readable output with all AI analyses.

Biophysics_Final_Report.md: The final, human-readable report.

Biophysics_Final_Report_Summary.json: A JSON file with the high-level summary metrics.

License
This project is licensed under the MIT License. See the LICENSE file for details.
