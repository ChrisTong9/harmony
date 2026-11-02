# Contributing

Bug reports and small, reproducible improvements are welcome. Before opening an issue, check the [setup guide](docs/setup.md) and [reproduction scope](docs/reproducibility.md).

For a bug report, include the task, command, Python version, expected behavior, and relevant error. Remove credentials, private audio, and participant information from any attachments. A dry-run report is useful for configuration problems.

For code changes, keep the public CLI and resource adapters separate from the frozen backend sources. Changes to editing behavior require a new source version and separate evaluation; they must not silently replace the recorded paper snapshot.

Run the following CPU-only checks:

```bash
python -m unittest discover -s tests -v
```

When intentionally preparing a new release, review the changes, regenerate its manifest, and verify the complete checkout:

```bash
python scripts/build_release_manifest.py
python scripts/check_release.py
```

Describe what changed and how it was checked in the pull request. The current source-license status is documented in [LICENSE-NOTICE.md](LICENSE-NOTICE.md).
