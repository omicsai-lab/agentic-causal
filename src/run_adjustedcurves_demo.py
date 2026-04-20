# src/run_adjustedcurves_demo.py
import argparse
import json
from pathlib import Path

from rpy2 import robjects as ro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--group", default="group", help="Treatment/group column name")
    parser.add_argument("--time", default="time", help="Time-to-event column name")
    parser.add_argument("--event", default="event", help="Event indicator column name (0/1)")
    parser.add_argument(
        "--covariates",
        default="",
        help="Comma-separated covariates for propensity model, e.g. X1,X2",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve().as_posix()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    group = args.group.strip()
    time_col = args.time.strip()
    event_col = args.event.strip()
    covariates = [c.strip() for c in args.covariates.split(",") if c.strip()]

    rhs = "1" if len(covariates) == 0 else " + ".join(covariates)
    tm_formula = f"{group} ~ {rhs}"

    plot_path = (out_dir / "survival_adjusted_curve.png").resolve()
    artifacts_path = (out_dir / "artifacts.json").resolve()

    if plot_path.exists():
        plot_path.unlink()

    R_CODE = f"""
suppressPackageStartupMessages({{
  library(survival)
  library(WeightIt)
  library(adjustedCurves)
}})

dat <- read.csv("{csv_path}")

dat${group} <- as.factor(dat${group})
dat${time_col} <- as.numeric(dat${time_col})
dat${event_col} <- as.numeric(dat${event_col})

adj <- adjustedsurv(
  data = dat,
  variable = "{group}",
  ev_time = "{time_col}",
  event = "{event_col}",
  method = "iptw_km",
  treatment_model = as.formula("{tm_formula}"),
  weight_method = "ps"
)

# Capture textual output and return it to Python
txt <- capture.output(print(adj))

# Save plot more robustly in non-interactive mode
png("{plot_path.as_posix()}", width = 1000, height = 700, res = 120)
p <- plot(adj)
if (!is.null(p)) {{
  print(p)
}}
dev.off()

txt
"""

    out = ro.r(R_CODE)

    stdout_lines = [str(x) for x in out]
    stdout_text = "\n".join(stdout_lines)
    if stdout_text.strip():
        print(stdout_text)

    result = {
        "method": "iptw_km",
        "treatment_model": tm_formula,
        "n_covariates": len(covariates),
    }

    if plot_path.exists():
        result["plot_path"] = plot_path.as_posix()
        result["graph_paths"] = [plot_path.as_posix()]
        print(f"DEBUG: plot created at {plot_path.as_posix()}")
    else:
        result["plot_error"] = "Plot file was not created by R."
        print("DEBUG: plot file was NOT created.")

    artifacts_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()