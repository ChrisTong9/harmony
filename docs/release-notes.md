# Source release preparation

This preparation combines the existing Harmony demo with the v377 code/result package. The public website paths and original demo/audio bytes are retained. The personal GitHub profile and anonymous review website are outside this change.

Changes in this preparation:

- A package-style CLI (`python -m harmony`) and compatibility script (`python harmony.py`).
- Seven named runtime backend directories; 20 historical source trees under `reproducibility/history/`.
- Configurable resource/checkpoint directories, separate from historical workspace paths.
- English method, setup, reproduction, and example documentation.
- An arXiv placeholder without a fabricated identifier or acceptance claim.
- CPU-only release tests and a GitHub Actions workflow.
- Resource-cache checks before creating an inference output directory.

No backend source or model prompt is edited in this step. Numerical records and evaluation outcomes are unchanged. The repository badge links to the actual GitHub Actions workflow status.

## Publication links

After the preprint is public, replace the README's `arXiv — link forthcoming` link and `ARXIV_URL` comment with the actual URL, and add verified citation metadata. No publication venue or acceptance status is inferred.

A source license remains to be selected by the copyright holders. See `LICENSE-NOTICE.md`.
