# Dependencies

The CLI and `--dry-run` use the standard library. Install the checkout with `python -m pip install -e .`, or run `python -m harmony` from its root.

- `planner.txt` records the planner's direct dependencies for Python 3.11. Install PyTorch and Torchaudio separately for the selected CUDA runtime.
- `music-render.txt` records direct dependencies for optional Basic Pitch and SoundFont rendering in a separate Python 3.11 environment. The renderer explicitly selects the ONNX model and CPU provider. A normal installation also installs package-declared dependencies; PyAudio may require PortAudio development headers.
- SAM Audio, Stable Audio 3, and the Qwen service use separate environments. Their installation commands and source revisions are in the [setup guide](../docs/setup.md).

These files are not complete dependency lockfiles. The direct versions were checked against the development environments; a fresh installation of every model environment has not yet been validated. Model weights, external repositories, SoundFont resources, and credentials must be supplied separately.

Use `python -m harmony doctor --resource-root /path/to/resources` to check the installation before starting a model run. All four operations share the same configuration; the planner selects the operation from the instruction.
