# Contributing

Bug reports and focused improvements are welcome. Start with the [setup guide](docs/setup.md) and [code map](docs/method.md).

For a bug report, include the instruction, command, Python version, expected behavior, and relevant error. Remove credentials and private audio from attachments. A `--dry-run` report and `harmony config` output are useful for configuration problems.

Keep changes in the shared `audio_edit/` implementation. Place evidence extraction in `analyzers/`, audio operations in `tools/`, and execution decisions in `policies/`.

To add a policy:

1. Add a strategy to the appropriate removal, material, or rendering stage. Define its evidence requirements, supported inputs, and permitted changes explicitly.
2. Register a [`PolicySpec`](audio_edit/policies/core.py) in the stage's `POLICIES` tuple and connect the strategy at the intended position. Record decisions with `record_policy()` and preserve stopping conditions; metadata alone does not enforce a call limit.
3. Test both an eligible input and a nearby ineligible input, plus any failure path. Update the [policy guide](docs/corrections.md) when conditions or budgets change.

Use synthetic fixtures for regression checks. Never select a policy by dataset ID or evaluation score. A newly supported sound class should be tested on examples and wording beyond those that motivated the change. Distinguish CPU checks from actual model execution in the pull request.

The project website is maintained on `gh-pages`. The source-license status is documented in [LICENSE-NOTICE.md](LICENSE-NOTICE.md).
