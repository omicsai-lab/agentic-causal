# Environment snapshot (STEP1_BENCHMARK_FREEZE)

Captured: 2026-08-18T22:11:00.254300+00:00

## Git
- commit: `150f50ed8b555a896378f25b13d9cf40bdec6e1d`
- branch: `paper_evaluation`
- dirty: `'?? evaluation/'` (empty string means clean)

## Interpreter
- python: `3.12.14 (main, Aug 14 2026, 15:10:43) [Clang 21.0.0 (clang-2100.1.1.101)]`
- executable: `/Users/jim/.pyenv/versions/3.12.14/bin/python3`
- platform: `macOS-26.5.2-arm64-arm-64bit`
- machine: `arm64`

## Package versions (installed vs. requirements.txt pin)

| Package | Installed | Pinned in requirements.txt |
|---|---|---|
| openai | 3.3.0 | 1.52.2 |
| fastapi | NOT INSTALLED | 0.115.2 |
| uvicorn | NOT INSTALLED | 0.32.0 |
| pydantic | 2.13.4 | 2.9.2 |
| gradio | NOT INSTALLED | 5.5.0 |
| huggingface_hub | NOT INSTALLED | 0.25.2 |
| requests | NOT INSTALLED | 2.32.3 |
| numpy | 2.5.2 | 2.1.2 |
| pandas | NOT INSTALLED | 2.2.3 |
| scipy | 1.18.0 | 1.14.1 |
| scikit-learn | NOT INSTALLED | 1.5.2 |
| matplotlib | NOT INSTALLED | 3.9.2 |
| reportlab | NOT INSTALLED | 4.2.5 |
| rpy2 | NOT INSTALLED | 3.5.16 |
| pytest | 9.1.1 | n/a |

## Notes
- `openai` is intentionally excluded from the version-mismatch warning list below if it matches; see AUDIT.md Finding 6 for why the installed version can legitimately differ from the requirements.txt pin in an evaluation-only environment.
- Version mismatches vs. requirements.txt:
  - openai: installed=3.3.0, pinned=1.52.2
  - pydantic: installed=2.13.4, pinned=2.9.2
  - numpy: installed=2.5.2, pinned=2.1.2
  - scipy: installed=1.18.0, pinned=1.14.1

---

# Environment snapshot (STEP2_LIVE_EVAL_FINALIZED)

Captured: 2026-08-18T23:08:58.421774+00:00

## Git
- commit: `3a78439786b5e68e2f5c8306dd6c0500df7f8456`
- branch: `paper_evaluation`
- dirty: `'M evaluation/semantic_orchestration/analysis/figures.py\n M evaluation/semantic_orchestration/analysis/run_all.py\n M evaluation/semantic_orchestration/analysis/tables.py\n?? evaluation/semantic_orchestration/analysis/baseline_comparison.py\n?? evaluation/semantic_orchestration/analysis/challenge_descriptive.py\n?? evaluation/semantic_orchestration/analysis/publication_tables.py\n?? evaluation/semantic_orchestration/analysis/stratified.py\n?? evaluation/semantic_orchestration/figures/\n?? evaluation/semantic_orchestration/results/processed/analysis_summary.json\n?? evaluation/semantic_orchestration/results/raw/\n?? evaluation/semantic_orchestration/tables/\n?? evaluation/semantic_orchestration/tests/test_baseline_comparison.py\n?? evaluation/semantic_orchestration/tests/test_challenge_descriptive.py\n?? evaluation/semantic_orchestration/tests/test_figures.py\n?? evaluation/semantic_orchestration/tests/test_publication_outputs.py\n?? evaluation/semantic_orchestration/tests/test_stratified.py'` (empty string means clean)

## Interpreter
- python: `3.12.14 (main, Aug 14 2026, 15:10:43) [Clang 21.0.0 (clang-2100.1.1.101)]`
- executable: `/Users/jim/.pyenv/versions/3.12.14/bin/python3`
- platform: `macOS-26.5.2-arm64-arm-64bit`
- machine: `arm64`

## Package versions (installed vs. requirements.txt pin)

| Package | Installed | Pinned in requirements.txt |
|---|---|---|
| openai | 3.3.0 | 1.52.2 |
| fastapi | NOT INSTALLED | 0.115.2 |
| uvicorn | NOT INSTALLED | 0.32.0 |
| pydantic | 2.13.4 | 2.9.2 |
| gradio | NOT INSTALLED | 5.5.0 |
| huggingface_hub | NOT INSTALLED | 0.25.2 |
| requests | NOT INSTALLED | 2.32.3 |
| numpy | 2.5.2 | 2.1.2 |
| pandas | NOT INSTALLED | 2.2.3 |
| scipy | 1.18.0 | 1.14.1 |
| scikit-learn | NOT INSTALLED | 1.5.2 |
| matplotlib | 3.11.1 | 3.9.2 |
| reportlab | NOT INSTALLED | 4.2.5 |
| rpy2 | NOT INSTALLED | 3.5.16 |
| pytest | 9.1.1 | n/a |

## Notes
- `openai` is intentionally excluded from the version-mismatch warning list below if it matches; see AUDIT.md Finding 6 for why the installed version can legitimately differ from the requirements.txt pin in an evaluation-only environment.
- Version mismatches vs. requirements.txt:
  - openai: installed=3.3.0, pinned=1.52.2
  - pydantic: installed=2.13.4, pinned=2.9.2
  - numpy: installed=2.5.2, pinned=2.1.2
  - scipy: installed=1.18.0, pinned=1.14.1
  - matplotlib: installed=3.11.1, pinned=3.9.2
