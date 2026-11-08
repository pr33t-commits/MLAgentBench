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
        expected = {"datasets": {"train": "train.csv", "test": "test.csv"},
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

    def test_trajectory_suffix_preserves_boolean_and_nested_values(self):
        expected = {"datasets": {"train": "train.csv"}, "objective": 'Keep {braces}, "quotes", and \\n.',
                    "script_name": "baseline.py", "use_demand_engine": False}
        info = SimpleNamespace(usage=dict.fromkeys(expected))
        for suffix in ('\nuser\nObservation:', '\nuser\n<think>\n\n</think>\nObservation:',
                       '\nI will now run this script.', '\nObservation: {"invented": true}'):
            for fenced in (False, True):
                text = json.dumps(expected)
                if fenced:
                    text = '```json\n' + text + '\n```'
                with self.subTest(suffix=suffix, fenced=fenced):
                    actual = Agent.parse_action_input(text + suffix, info)
                    self.assertEqual(actual, expected)
                    self.assertIs(actual['use_demand_engine'], False)

    def test_malformed_or_ambiguous_input_is_not_repaired(self):
        info = SimpleNamespace(usage={'flag': ''})
        for text in ('{"flag": false', '{"flag": False}', '{"flag": false,}',
                     '{"flag": false, "flag": true}', '{"flag": NaN}',
                     '[{"flag": false}]', '{"flag": false}\n{"flag": true}',
                     '{"flag": false}\nAction: Delete File\nAction Input: {}',
                     'Here is JSON: {"flag": false}', '{}', '{"flag": false, "extra": 1}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                Agent.parse_action_input(text, info)
        with self.assertRaises(ValueError):
            Agent.parse_action_input('garbage', SimpleNamespace(usage={}))
        self.assertEqual(Agent.parse_action_input('{}\nuser', SimpleNamespace(usage={})), {})

    def test_first_action_is_not_replaced_by_trailing_action(self):
        text = ('Thought: inspect\nAction: Read File\nAction Input: {"file_name": "a"}\n'
                'Action: Delete File\nAction Input: {"file_name": "b"}')
        entries = Agent.parse_entries(text, ['Thought', 'Action', 'Action Input'])
        self.assertEqual(entries['Action'].strip(), 'Read File')
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            Agent.parse_action_input(entries['Action Input'], SimpleNamespace(usage={'file_name': ''}))

    def test_header_words_inside_json_are_not_fields(self):
        expected = {'content': 'Action: example\nAction Input: {"a": 1}'}
        text = 'Action: Write File\nAction Input: ' + json.dumps(expected) + '\nuser'
        entries = Agent.parse_entries(text, ['Action', 'Action Input'])
        self.assertEqual(entries['Action'].strip(), 'Write File')
        self.assertEqual(Agent.parse_action_input(entries['Action Input'],
                                                SimpleNamespace(usage={'content': ''})), expected)


if __name__ == "__main__":
    unittest.main()
