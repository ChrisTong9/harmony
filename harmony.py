#!/usr/bin/env python3
"""Compatibility entry point; prefer python -m harmony."""

from harmony.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
