"""Catch dangling internal imports without loading model dependencies."""

import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"audio_edit", "harmony"}


def module_exists(name):
    path = ROOT.joinpath(*name.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


class PackageImportsTests(unittest.TestCase):
    def test_internal_import_targets_exist(self):
        failures = []
        for package in PACKAGES:
            for path in (ROOT / package).rglob("*.py"):
                relative = path.relative_to(ROOT).with_suffix("")
                parent = ".".join(relative.parts[:-1])
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    targets = []
                    if isinstance(node, ast.Import):
                        targets = [item.name for item in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        if node.level:
                            target = importlib.util.resolve_name(
                                "." * node.level + (node.module or ""), parent
                            )
                        else:
                            target = node.module or ""
                        targets = [target]
                    for target in targets:
                        if target.split(".")[0] in PACKAGES and not module_exists(target):
                            failures.append(f"{path.relative_to(ROOT)}:{node.lineno}: {target}")
        self.assertEqual(failures, [])

    def test_registered_operations_have_executable_modules(self):
        from audio_edit.scripts.registry import ScriptRegistry

        registry = ScriptRegistry().list_scripts()
        self.assertEqual(set(registry), {"add_sound", "remove_sound", "replace", "volume_change"})
        for entry in registry.values():
            path = ROOT.joinpath(*entry["module"].split(".")).with_suffix(".py")
            tree = ast.parse(path.read_text())
            self.assertTrue(
                any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body)
            )


if __name__ == "__main__":
    unittest.main()
