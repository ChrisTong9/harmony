"""Policy records remain inspectable without executing a strategy."""

import json
import unittest
from types import SimpleNamespace

from audio_edit.policies import PolicySpec, catalog, record_policy


class PolicyRecordTests(unittest.TestCase):
    def test_catalog_has_unique_descriptions_for_all_stages(self):
        rows = json.loads(json.dumps(catalog()))
        self.assertEqual({row["stage"] for row in rows}, {"removal", "material", "rendering"})
        self.assertEqual(len(rows), len({row["policy_id"] for row in rows}))
        for row in rows:
            self.assertEqual(
                set(row), {"policy_id", "stage", "condition", "evidence", "action", "budget"}
            )
            self.assertTrue(all(isinstance(value, str) and value.strip() for value in row.values()))

    def test_record_snapshots_decision_without_changing_existing_audit(self):
        audit = {"status": "not_applicable", "reasons": ["missing evidence"]}
        ctx = SimpleNamespace(params={"source_audit": audit})
        spec = PolicySpec("material.example", "material", "condition", "evidence", "action", "0")
        record_policy(ctx, spec, "not_applicable", subject="asset.wav", audit=audit)
        audit["reasons"].append("later update")
        result = json.loads(json.dumps(ctx.params["policy_trace"]))
        self.assertEqual(result[0]["details"]["audit"]["reasons"], ["missing evidence"])
        self.assertEqual(result[0]["subject"], "asset.wav")
        self.assertIs(ctx.params["source_audit"], audit)

    def test_receipts_append_in_order_and_omit_empty_optional_fields(self):
        ctx = SimpleNamespace(params={})
        spec = PolicySpec("removal.example", "removal", "condition", "evidence", "action", "0")
        record_policy(ctx, spec, "selected")
        record_policy(ctx, spec, "applied")
        self.assertEqual(
            [row["status"] for row in ctx.params["policy_trace"]], ["selected", "applied"]
        )
        self.assertEqual(set(ctx.params["policy_trace"][0]), {"policy", "stage", "status"})


if __name__ == "__main__":
    unittest.main()
