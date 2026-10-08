"""Test the real parser methods without loading API clients."""
import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


source = Path(__file__).resolve().parents[1] / "MLAgentBench/agents/agent.py"
tree = ast.parse(source.read_text())
agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Agent")
agent.body = [node for node in agent.body if isinstance(node, ast.FunctionDef) and
              node.name in {"parse_entries", "parse_action_input", "parse_action_input_by_matching", "sanitize_json_string"}]
namespace = {"json": json, "re": re}
exec(compile(ast.Module(body=[agent], type_ignores=[]), str(source), "exec"), namespace)
Agent = namespace["Agent"]


class ActionInputTests(unittest.TestCase):
    def test_action_only_response_and_research_missing_fields(self):
        response = 'Action: List Files\nAction Input:\n```json\n{"dir_path": "."}\n```'
        parsed = Agent.parse_entries(response, ["Action", "Action Input"])
        self.assertEqual(parsed["Action"].strip(), "List Files")
        self.assertEqual(Agent.parse_action_input(parsed["Action Input"], SimpleNamespace(usage={"dir_path": ""})), {"dir_path": "."})
        with self.assertRaisesRegex(Exception, "Invalid:"):
            Agent.parse_entries(response, ["Reflection", "Research Plan and Status", "Fact Check", "Thought", "Action", "Action Input"])

    def test_fenced_dataset_mapping_preserves_escaping(self):
        expected = {"datasets": {"train": "train.csv", "validation": "validation.csv"},
                    "objective": "Analyze sales",
                    "script_name": "eda_sales.py"}
        info = SimpleNamespace(usage={key: "" for key in expected})
        for opening in ("", "```json\n", "```\n"):
            text = opening + json.dumps(expected) + ("\n```" if opening else "")
            with self.subTest(opening=opening):
                parsed = Agent.parse_action_input(text, info)
                self.assertEqual(parsed, expected)
                self.assertEqual(len(parsed["datasets"]), 2)

    def test_fenced_newlines_and_backslashes(self):
        expected = {"datasets": {"train": "data/train.csv"},
                    "objective": 'Analyze sales.\nKeep \\n literal.',
                    "script_name": "eda.py"}
        info = SimpleNamespace(usage={key: "" for key in expected})
        self.assertEqual(Agent.parse_action_input("```json\n" + json.dumps(expected) + "\n```", info), expected)


if __name__ == "__main__":
    unittest.main()
