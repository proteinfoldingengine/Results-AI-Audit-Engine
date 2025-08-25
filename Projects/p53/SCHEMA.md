
JSON Schema (machine-readable)
{
  "$id": "https://uqcf-gem.org/schemas/findings-2.2.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "UQCF-GEM Findings Manifest",
  "description": "Canonical, machine-auditable contract linking scientific claims to artifacts and numeric constraints.",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "schema_version": {
      "type": "string",
      "description": "Semantic-ish version of this manifest format.",
      "pattern": "^[0-9]+\\.[0-9]+(\\.[0-9]+)?$"
    },
    "project": { "type": "string", "minLength": 1 },
    "created_utc": { "type": "string", "format": "date-time" },
    "dataset_root": { "type": "string", "description": "Relative path prefix used to resolve artifact paths." },

    "metrics_vocabulary": {
      "type": "object",
      "description": "Declares all metrics that may be referenced by constraints.",
      "minProperties": 1,
      "additionalProperties": false,
      "patternProperties": {
        "^[A-Za-z0-9_]+$": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "unit": { "type": "string" },
            "aggregation": {
              "type": "string",
              "description": "How to compute metric for verification.",
              "enum": [
                "min_across_runs",
                "max_across_runs",
                "count_rows",
                "from_report_or_zero",
                "timeseries_median"
              ]
            },
            "preferred_sources": {
              "type": "array",
              "items": {
                "type": "string",
                "enum": ["md", "raw_csv", "diagnostics_csv"]
              },
              "minItems": 1,
              "uniqueItems": true
            },
            "fallback_ok": { "type": "boolean" }
          },
          "required": ["unit", "aggregation", "preferred_sources"]
        }
      }
    },

    "findings": {
      "type": "array",
      "minItems": 1,
      "items": { "$ref": "#/$defs/RunFinding" }
    },

    "thesis": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "canonical_field_claim": { "$ref": "#/$defs/ThesisClaim" },
        "x_notes": { "type": "string" }
      },
      "required": ["canonical_field_claim"]
    },

    "x_metadata": { "type": "object", "description": "Freeform extension block." }
  },
  "required": ["schema_version", "project", "metrics_vocabulary", "findings"],

  "$defs": {
    "Constraint": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "metric": { "type": "string", "pattern": "^[A-Za-z0-9_]+$" },
        "op": {
          "type": "string",
          "enum": ["<", "<=", "==", ">=", ">", "!="]
        },
        "value": { "type": "number" },
        "tolerance": {
          "type": "number",
          "description": "Optional symmetric tolerance band.",
          "default": 0.0
        }
      },
      "required": ["metric", "op", "value"]
    },

    "Artifact": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "role": {
          "type": "string",
          "enum": [
            "comprehensive_png",
            "diagnostics_png",
            "diagnostics_csv",
            "raw_csv",
            "by_param_csv",
            "trajectory_pdb",
            "md_report"
          ]
        },
        "path": {
          "type": "string",
          "minLength": 1,
          "description": "Relative to dataset_root/source_folder unless absolute or URL."
        }
      },
      "required": ["role", "path"]
    },

    "CanonicalClaim": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "type": { "type": "string", "minLength": 1 },
        "statement": { "type": "string", "minLength": 1 },
        "constraints": {
          "type": "array",
          "minItems": 1,
          "items": { "$ref": "#/$defs/Constraint" }
        },
        "tags": { "type": "array", "items": { "type": "string" }, "uniqueItems": true },
        "notes": { "type": "array", "items": { "type": "string" } },
        "baseline_key": { "type": "string" }
      },
      "required": ["type", "statement", "constraints"]
    },

    "RunFinding": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "run_id": { "type": "integer", "minimum": 1 },
        "source_folder": { "type": "string", "minLength": 1 },
        "purpose": { "type": "string" },
        "canonical_claim": { "$ref": "#/$defs/CanonicalClaim" },
        "artifacts": {
          "type": "array",
          "minItems": 1,
          "items": { "$ref": "#/$defs/Artifact" }
        },
        "x_extra": { "type": "object" }
      },
      "required": ["run_id", "source_folder", "canonical_claim", "artifacts"]
    },

    "ThesisWhereClause": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "metric": { "type": "string", "pattern": "^[A-Za-z0-9_]+$" },
        "op": { "type": "string", "enum": ["<", "<=", "==", ">=", ">", "!="] },
        "value": { "type": "number" },
        "tolerance": { "type": "number" }
      },
      "required": ["metric", "op", "value"]
    },

    "ThesisConstraint": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "aggregation": {
          "type": "string",
          "enum": ["count_runs_meeting"]
        },
        "where": {
          "type": "array",
          "minItems": 1,
          "items": { "$ref": "#/$defs/ThesisWhereClause" }
        },
        "op": { "type": "string", "enum": ["<", "<=", "==", ">=", ">", "!="] },
        "value": { "type": "number" }
      },
      "required": ["aggregation", "where", "op", "value"]
    },

    "ThesisClaim": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "statement": { "type": "string", "minLength": 1 },
        "constraints": {
          "type": "array",
          "minItems": 1,
          "items": { "$ref": "#/$defs/ThesisConstraint" }
        },
        "implications": { "type": "array", "items": { "type": "string" } },
        "future_directions": { "type": "array", "items": { "type": "string" } }
      },
      "required": ["statement", "constraints"]
    }
  }
}

Technical Reference (human-readable)
Purpose
findings.json is the science contract for a campaign: it encodes each run’s claim as numeric constraints and binds those claims to concrete artifacts (PNGs, CSVs, PDBs, MD). It also encodes a thesis that can be machine-evaluated across runs.
Top-Level Fields
schema_version — format version (e.g., "2.2").


project — free text name.


created_utc — ISO 8601 timestamp.


dataset_root — base path (relative) for resolving artifacts[].path.


metrics_vocabulary — declares every metric you’ll reference in constraints, with:


aggregation (how to compute or choose values),


preferred_sources (ordered list among md, raw_csv, diagnostics_csv),


optional fallback_ok (allow non-preferred).


findings[] — one entry per run (see below).


thesis.canonical_field_claim — a global, machine-testable claim (e.g., “≥1 run meets rescue criteria”), expressed with constraints using aggregation: "count_runs_meeting" and a where filter of per-run constraints.


Per-Run (
findings[]
) Required Fields
run_id — integer ≥ 1.


source_folder — directory containing the run artifacts (resolved under dataset_root unless absolute or URL).


canonical_claim — the testable claim for this run:


type — free text classifier (e.g., rescue_protocol, ligand_partial).


statement — human text.


constraints[] — list of {metric, op, value, tolerance?}.


artifacts[] — each with:


role ∈ {comprehensive_png, diagnostics_png, diagnostics_csv, raw_csv, by_param_csv, trajectory_pdb, md_report}


path — file path or URL (absolute paths allowed but discouraged).


Operators & Tolerance
op ∈ <, <=, ==, >=, >, !=.


tolerance (optional, number) widens comparisons symmetrically (e.g., <= 20.0 with tolerance: 0.5 means pass if value ≤ 20.5).


Thesis Constraints
Only the aggregator count_runs_meeting is standardized here.


where[] uses the same constraint shape as runs (metric/op/value[/tolerance]).


Example: count of runs meeting {RMSD <= 20.0, failures == 0} must be >= 1.


Extensibility
The schema forbids unknown fields except:


x_metadata (top-level), x_extra (per-run), and any notes/tags.


Add new artifact roles or aggregations by proposing schema updates; meanwhile, stash experimental additions under x_*.


Conformance Levels
Strict: validate against this schema (recommended for publication).


Lenient: allow additional roles/aggregations but keep findings[].artifacts[].path resolvable and all referenced metrics declared in metrics_vocabulary.



Minimal Valid Example
{
  "schema_version": "2.2",
  "project": "Example Campaign",
  "created_utc": "2025-08-25T20:00:00Z",
  "dataset_root": "_analysis",
  "metrics_vocabulary": {
    "best_final_RMSD_A": {
      "unit": "Å",
      "aggregation": "min_across_runs",
      "preferred_sources": ["md", "raw_csv", "diagnostics_csv"]
    },
    "failures": {
      "unit": "count",
      "aggregation": "from_report_or_zero",
      "preferred_sources": ["md"]
    }
  },
  "findings": [
    {
      "run_id": 1,
      "source_folder": "WT_RUN_001",
      "canonical_claim": {
        "type": "baseline",
        "statement": "WT collapses (RMSD ≥ 20 Å).",
        "constraints": [{ "metric": "best_final_RMSD_A", "op": ">=", "value": 20.0 }]
      },
      "artifacts": [
        { "role": "md_report", "path": "WT_RUN_001_report.md" },
        { "role": "raw_csv", "path": "WT_RUN_001_raw.csv" }
      ]
    }
  ],
  "thesis": {
    "canonical_field_claim": {
      "statement": "At least one run rescues (RMSD ≤ 20 & failures == 0).",
      "constraints": [
        {
          "aggregation": "count_runs_meeting",
          "where": [
            { "metric": "best_final_RMSD_A", "op": "<=", "value": 20.0 },
            { "metric": "failures", "op": "==", "value": 0 }
          ],
          "op": ">=",
          "value": 1
        }
      ]
    }
  }
}


