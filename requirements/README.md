# Dependency records

The CLI, table reconstruction, and tests use only Python's standard library. `planner.txt` lists direct planner dependencies with versions observed in the archived mixer environment. It is a starting point for that environment, not a complete installation for SAM Audio, Stable Audio, Qwen serving, or music rendering.

See `docs/setup.md` for the separate environment layout. Full observed package inventories and Python versions are retained under `reproducibility/environments/`. Model weights, third-party source code, and credentials are not installed by the requirements file.
