# causal-agent-mvp

An agent-based causal inference pipeline that automatically selects and executes appropriate causal analysis tools (e.g., Average Treatment Effect estimation and survival adjusted curves) based on user requests and dataset structure.

The system combines:

rule-based checks

LLM-assisted routing

deterministic statistical backends

and produces both human-readable outputs and structured JSON artifacts.

The framework is designed to ensure that causal analyses are executed reproducibly and transparently, while preventing inappropriate use of causal methods.


---

# Features

Agentic workflow for causal inference

Automatic capability selection via an LLM router

Deterministic execution of statistical tools

Structured JSON outputs for downstream use

Dynamic Add Tool capability for extending the system

End-to-end reproducible demos using real datasets

Currently supported causal capabilities include:

Average Treatment Effect (ATE) estimation

Confounder-adjusted survival curves

---

# Repository Structure

```bash

causal-agent-mvp/

├── data/                # Example datasets (PBC, GBSG2)

├── scripts/             # Demo and helper scripts

├── src/agent/           # Agent logic, router, schemas, tools

├── out/                 # Runtime outputs (gitignored)

├── README.md
```

---

# Requirements

Python 3.9+

R (required for survival adjusted curves)

Required R packages:

```bash
adjustedCurves
WeightIt
survival

```

--- 

# Quickstart: End-to-End Demos

Below are two fully tested demo commands.
Both have been run successfully end-to-end and generate structured JSON outputs under:
```bash
out/api/
```

---

# Demo 1: Average Treatment Effect (ATE)

Estimate the causal effect of treatment on a binary 5-year outcome using doubly robust estimation.

```bash
curl -s -X POST "http://127.0.0.1:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "csv": "data/PBC_ate5y_cc.csv",
    "request": "Estimate the causal effect (ATE) of treatment on 5-year survival",
    "use_llm_router": true,
    "treatment": "trt01",
    "outcome": "Y5y",
    "covariates": ["age","bili","albumin","protime","edema","platelet","ast"]
  }'
```
Expected behavior:

Selected capability:

```bash

causal_ate
```

Backend method:
```bash

Doubly robust ATE estimation
```

Output includes:

ATE

Standard error

95% confidence interval

Structured JSON output written to:
```bash

out/api/causalmodels.summary.json
```

# Demo 2: Survival Adjusted Curves

Compare survival between treatment groups using IPTW-adjusted Kaplan–Meier curves.

```bash
curl -s -X POST "http://127.0.0.1:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "csv": "data/GBSG2_agent01.csv",
    "request": "Compare survival between groups",
    "use_llm_router": true,
    "time": "time",
    "event": "event",
    "group": "horTh01",
    "covariates": []
  }'
```

Expected behavior:

Selected capability:
```bash

survival_adjusted_curves
```

Backend method:
```bash

IPTW-adjusted Kaplan–Meier
```

JSON summary written to:
```bash

out/api/adjustedcurves.summary.json
```

---

# Output Format

Each API run returns structured JSON.

Example response:

```json

{
  "status": "ok",
  "selected_tool": "causalmodels",
  "stdout": "...",
  "stderr": "...",
  "artifacts": {
    "capability_id": "causal_ate",
    "summary_json": "out/api/causalmodels.summary.json",
    "router_reason": "LLM selected causal_ate"
  }
}

```
Key fields:

Field	Description
status	execution status
selected_tool	executed backend tool
stdout	human-readable logs
stderr	warnings or messages
artifacts	structured output metadata

---

# Input Data Requirements

The framework assumes that the input dataset satisfies the following requirements.

As long as these conditions are met, the pipeline can be executed end-to-end without modification to the core codebase.

---

# CSV File Requirements

Input data must be provided as a CSV file.

General rules:

UTF-8 encoding

comma-separated

header row required

each row represents one observational unit

Example:

```bash
age,bili,albumin,protime,trt01,Y5y
58,1.2,3.5,10.2,1,0
62,2.1,3.1,11.0,0,1
```

---

# Column Naming Rules

Column names must:

be unique

contain no empty values

avoid special characters

avoid trailing spaces

Recommended naming style:

```bash
age
treatment
outcome
covariate1
covariate2
```
Avoid:

```bash
Age (years)
treatment group
Outcome %
```
---

# Required Variables

Required variables depend on the causal task.

---

## Treatment / Exposure

A treatment variable must be provided.

Treatment must be binary, represented as:

```bash
0 / 1
```
Example:
```bash
trt01
0 = control
1 = treated

```

--- 

## Outcome

For ATE estimation

Outcome may be:

binary

continuous

Example:
```bash

Y5y
```
For survival analysis

Two variables are required:
```bash
time
event
```
Where:

```bash
event = 1  event occurred
event = 0  censored
```
---

## Covariates (Confounders)

Covariates may be provided to adjust for confounding.

Requirements:

measured before treatment

numeric or numerically encoded

no missing values

Example:
```bash
age
bili
albumin
protime
```

---

# Structural and Causal Assumptions

The framework relies on standard causal inference assumptions:

### Consistency

Observed outcomes correspond to the assigned treatment.

### Positivity

Each covariate pattern has non-zero probability of receiving each treatment.

### No unmeasured confounding

All relevant confounders are included in the covariates.

These assumptions are not automatically verified and must be justified by the user.

---

# Preprocessing Expectations

To ensure stable execution:

all required variables must be present

missing values must be removed or imputed

categorical variables should be numerically encoded

extremely rare treatment groups may lead to unstable estimates

---
# Add Tool: Extending the System

The framework allows users to dynamically add new analysis tools.

Each tool requires:

a Python tool file

a capability JSON file

---

# Capability JSON Format

File naming convention:
```bash

cap_<tool_name>.json
```

Example:

```json

{
  "capability_id": "hello_world",
  "description": "A simple demo tool that prints hello world.",
  "required_fields": [],
  "optional_fields": [],
  "tool": "tool_hello_world"
}
```
Field definitions:

Field	Description
capability_id	unique capability identifier
description	description used by the LLM router
required_fields	required request parameters
optional_fields	optional parameters
tool	python module name
Python Tool File Format

File naming convention:

```bash

tool_<tool_name>.py

```

Each tool must inherit from BaseTool.

Example template:

```python
from typing import Tuple
from src.agent.schemas_io import RunRequest
from src.agent.tools.base import BaseTool


class HelloWorldTool(BaseTool):

    @property
    def name(self):
        return "hello_world"

    @property
    def capability_id(self):
        return "hello_world"

    def validate(self, req: RunRequest) -> Tuple[bool, str]:
        return True, ""

    def run(self, req: RunRequest):
        return {
            "status": "ok",
            "stdout": "Hello World",
            "stderr": "",
            "artifacts": {"message": "tool executed"},
            "error": None
        }

```
---

# Tool Output Requirements

The run() method must return a JSON-serializable dictionary containing:

```json
status
stdout
stderr
artifacts
error
```

---
# Tool Validation Pipeline

When a tool is uploaded, the system performs:

JSON schema validation

Python interface validation

capability_id consistency check

runtime execution validation

If any stage fails, a structured error will be returned.

---
# Notes

Informational messages from R packages may appear in stderr and can be safely ignored.

Runtime outputs under out/ are not tracked by git.
