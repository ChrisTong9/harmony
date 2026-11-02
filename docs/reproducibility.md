# Reproduction scope and provenance

## What can be reproduced offline

`python -m harmony tables --out outputs/tables` recomputes the task-level and aggregate tables from `results/<configuration>/<task>/per_sample.json` without calling an editor or evaluator. It verifies all 72 displayed metric values against `configs/expected_tables.json`.

The cohorts contain 115 addition, 149 removal, 79 replacement, and 77 volume examples, totaling 420. The four configurations contain 1,680 numerical records. Macro averages weight tasks equally; pooled averages weight examples equally. Scores use their unrounded values until final display. The same sample IDs must appear across all configurations within each task.

The numerical results were copied unchanged from the v377 paper package. This reorganization produces no new audio, votes, or benchmark scores.

## What the snapshot measures

The frozen results include checkpoint continuation and task repairs. A1B1 includes later full-system revisions, including the v363 replacement integration. Other rows use the corresponding earlier configuration views. They are not all evaluations of a single identical implementation with only A and B toggled. Consequently, the differences between the recorded rows also include version changes.

Historical outputs, first evaluator votes, and full execution logs remain in the original local result archive. This public package contains per-example numerical scores and source identity records, not that full audio/log archive. The public demo contains its existing selected audio examples; it is not the full benchmark corpus.

## New-input execution

`python -m harmony run` uses the selected primary backend plus the common execution service. It does not look up a benchmark sample ID, select a historical repair by score, or replay stored predictions. The history directory is not consulted by the runtime driver.

A fresh stochastic edit is not guaranteed to match a historical waveform or evaluator score. Model versions, sampling, service behavior, and numerical environments can affect a rerun. The reorganization has been checked with source hashes, compilation, all 16 profile dry runs, CPU tests, and offline table reconstruction. It has not been evaluated with a fresh full GPU benchmark.

## Files

- `SOURCES.json`: current locations, cleaned source digests, and original paper source identities.
- `layout.json`: mapping from the earlier archive layout to this repository.
- `original_MANIFEST.json`: the manifest of the earlier code archive; its paths describe that earlier layout.
- `original_CODE_CLEANUP.json`: prior comment/console cleanup and source hash lineage.
- `original_profiles.json`: original profile values, including historical workspace paths.
- `release_manifest.json`: hashes for this reorganized release, checked by `python -m harmony verify`.
- `environments/`: package observations and external source commit IDs, not a retrospective dependency lockfile.
- `report.json`, `summary.csv`, `per_sample.csv`: historical timing/call observations.
- `correction_budgets.json`: stage-specific correction limits.

Paths above are relative to `reproducibility/`. Historical environment records may include old workspace paths for traceability; the new runtime profiles use a configurable resource root.

To intentionally make a new packaging revision, run `python scripts/build_release_manifest.py` after reviewing the changes, then rerun the checks. Regenerating a manifest is not evidence that editing behavior or published scores were re-evaluated.
