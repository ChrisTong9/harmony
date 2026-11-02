# Contributing

Bug reports and focused improvements are welcome. Start with the [setup guide](docs/setup.md) and [code map](docs/method.md).

For a bug report, include the task, command, Python version, expected behavior, and relevant error. Remove credentials and private audio from attachments. A `--dry-run` report is useful for configuration problems.

Keep changes in the shared `audio_edit/` implementation. Use task parameters and mechanism switches where behavior differs; avoid adding separate version directories. Include how the change was checked and distinguish CPU checks from actual model execution.

The project website is maintained on `gh-pages`. The source-license status is documented in [LICENSE-NOTICE.md](LICENSE-NOTICE.md).
