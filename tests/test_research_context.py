import ast
import copy
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import uuid

from MLAgentBench.research_context import build_messages, fit_messages, flatten_messages
from MLAgentBench.qwen_tools import uses_native_tools, tool_definitions, tool_alias, parse_tool_response


def step(number=0):
    return {"action": {"Thought": "inspect", "Action": "Execute Script", "Action Input": '{}'},
            "raw_assistant_response": "Thought: inspect\nAction: Execute Script\nAction Input: {}",
            "action_input": {}, "tool_call_id": f"call_{number}", "observation": "done"}


class ContextTests(unittest.TestCase):
    def test_roles_pairs_and_original_response(self):
        history = [step(0), step(1)]
        history[0]["action"]["Thought"] = "normalized"
        messages = build_messages("rules", "task", history, 3, 3)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "tool", "assistant", "tool"])
        for i in (2, 4):
            self.assertEqual(messages[i]["tool_calls"][0]["id"], messages[i+1]["tool_call_id"])
            self.assertEqual(messages[i]["tool_calls"][0]["function"]["arguments"], {})
        self.assertIn("inspect", messages[2]["content"])
        self.assertNotIn("normalized", messages[2]["content"])

    def test_summary_memory_and_omission_are_explicit(self):
        history = [step(0), step(1)]
        history[1]["observation_provenance"] = {"kind": "summary", "model": "helper"}
        messages = build_messages("rules", "task", history, 2, 1, "old evidence", "helper")
        self.assertEqual(messages[2]["role"], "user")
        self.assertIn("not a new user instruction", messages[2]["content"])
        self.assertIn("omitted", messages[4]["content"])
        self.assertIn("Derived tool result", messages[6]["content"])
        self.assertIn("helper", messages[6]["content"])

    def test_failed_arguments_and_legacy_checkpoint(self):
        old = {"action": {"Action": "Execute Script", "Action Input": '{}'}, "observation": "old"}
        messages = build_messages("rules", "task", [old], 3, 3)
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertIn("Legacy stored", messages[-1]["content"])
        failed = step()
        failed.update(parsing_error="Invalid JSON", action_input="bad")
        messages = build_messages("rules", "task", [failed], 3, 3)
        self.assertNotIn("tool_calls", messages[-2])
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("framework feedback", messages[-1]["content"])

    def test_token_trimming_preserves_latest_pair_and_input(self):
        messages = build_messages("rules", "task", [step(0), step(1)], 3, 3, "memory")
        original = copy.deepcopy(messages)
        result = fit_messages(messages, len, 4)
        self.assertEqual([m["role"] for m in result], ["system", "user", "assistant", "tool"])
        self.assertEqual(result[-1]["tool_call_id"], "call_1")
        self.assertEqual(messages, original)
        with self.assertRaisesRegex(ValueError, "context budget"):
            fit_messages(messages, len, 3)
        self.assertEqual(len(build_messages("rules", "task", [step()], 0, 0)), 2)


class ResearchLoopTests(unittest.TestCase):
    """Execute real agent methods with fake inference/environment, no API imports."""
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "MLAgentBench/agents"
        tree = ast.parse((root / "agent.py").read_text())
        base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Agent")
        base.body = [n for n in base.body if isinstance(n, ast.FunctionDef) and n.name in {
            "print_action", "parse_entries", "parse_action_input", "parse_action_input_by_matching",
            "sanitize_json_string", "save", "restore"}]
        self.ns = dict(os=os, sys=sys, re=re, json=json, uuid=uuid, Namespace=SimpleNamespace,
                       EnhancedJSONEncoder=json.JSONEncoder, AI_PROMPT="\n\nAssistant:",
                       build_messages=build_messages, flatten_messages=flatten_messages,
                       uses_native_tools=uses_native_tools, tool_definitions=tool_definitions,
                       tool_alias=tool_alias, parse_tool_response=parse_tool_response,
                       Action=lambda name, arguments: SimpleNamespace(name=name, arguments=arguments))
        exec(compile(ast.Module(body=[base], type_ignores=[]), str(root / "agent.py"), 'exec'), self.ns)
        tree = ast.parse((root / "agent_research.py").read_text())
        assignments = [n for n in tree.body if isinstance(n, ast.Assign)]
        exec(compile(ast.Module(body=assignments, type_ignores=[]), '<prompts>', 'exec'), self.ns)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(root / "agent_research.py"), 'exec'), self.ns)
        self.agent = self.ns['ResearchAgent'].__new__(self.ns['ResearchAgent'])
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.agent.log_dir = self.tmp.name
        self.agent.system_prompt = "general rules"
        self.agent.task_prompt = "task data"
        self.agent.history_steps = []
        self.agent.args = SimpleNamespace(max_steps_in_context=1, max_observation_steps_in_context=1,
            retrieval=True, agent_max_steps=3, max_retries=2, llm_name="huggingface/Qwen/Qwen3.5-4B",
            fast_llm_name="helper")
        self.agent.valid_format_entires = ["Research Plan and Status", "Thought", "Action", "Action Input"]
        self.agent.all_tool_names = ["Execute Script", "Inspect Script Lines"]
        self.agent.action_infos = {name: SimpleNamespace(usage={"script_name": "str"}) for name in self.agent.all_tool_names}
        self.agent.save = Mock()
        self.agent.summarize_observation = Mock(return_value="metric=1.25")
        self.agent.summarize_action_and_observation = Mock(return_value="[Reasoning]: summary")
        self.prompts = []
        self.responses = []
        def complete(prompt, *args, **kwargs):
            self.prompts.append(copy.deepcopy(prompt))
            return self.responses.pop(0) if self.responses else self.response()
        self.ns['complete_text'] = complete
        self.calls = []
        self.env = SimpleNamespace(is_final=lambda: False, trace=SimpleNamespace(steps=[]))
        def execute(action):
            self.calls.append(action)
            self.env.trace.steps.append(action)
            return "x" * 5100 if action.name in self.agent.all_tool_names else "retrieved summary"
        self.env.execute = execute

    def response(self, arguments='{"script_name": "run.py"}', action='Execute Script'):
        return f"Research Plan and Status: **verified plan**\nThought: inspect\nAction: {action}\nAction Input: {arguments}"

    def test_loop_summary_retrieval_retry_and_roundtrip(self):
        self.responses = ["malformed", self.response()]
        self.agent.run(self.env)
        self.assertEqual([m['role'] for m in self.prompts[0]], ['system', 'user'])
        self.assertEqual(self.prompts[1][-2]['content'], 'malformed')
        self.assertIn('validation feedback', self.prompts[1][-1]['content'])
        self.assertEqual(self.prompts[2][-1]['role'], 'tool')
        self.assertIn('Derived tool result', self.prompts[2][-1]['content'])
        saved = self.agent.history_steps[0]
        self.assertEqual(saved['raw_observation'], 'x' * 5100)
        self.assertEqual(saved['observation'], 'metric=1.25')
        self.assertIn('**verified plan**', saved['raw_assistant_response'])
        retrieval = next(c for c in self.calls if c.name == 'Retrieval from Research Log')
        self.assertIn('verified plan', retrieval.arguments['current_plan'])
        # Exercise actual save/restore methods with JSON-serializable state.
        restored = self.ns['ResearchAgent'].__new__(self.ns['ResearchAgent'])
        restored.log_dir = self.tmp.name
        restored.args = vars(self.agent.args)
        restored.history_steps = self.agent.history_steps
        path = os.path.join(self.tmp.name, 'checkpoint.json')
        restored.save(path)
        restored.history_steps = []
        restored.restore(path)
        self.assertEqual(restored.history_steps, self.agent.history_steps)
        self.assertEqual(build_messages('s', 't', restored.history_steps, 1, 1)[-1]['role'], 'tool')

    def test_summary_failure_is_labeled_and_raw_result_survives(self):
        self.agent.args.agent_max_steps = 1
        self.agent.summarize_observation.side_effect = RuntimeError('offline')
        self.agent.run(self.env)
        saved = self.agent.history_steps[0]
        self.assertEqual(saved['observation_provenance']['kind'], 'truncated')
        self.assertEqual(len(saved['raw_observation']), 5100)
        self.assertIn('Truncated', saved['observation'])

    def test_multi_block_summary_labels_sources_and_merges(self):
        helper = Mock(side_effect=['[Observation]: first', '[Observation]: second',
                                   '[Observation]: merged'])
        self.ns['complete_text_fast'] = helper
        result = self.ns['ResearchAgent'].summarize_observation(
            self.agent, 'Action: inspect', 'abcdefgh', 'summary.log', bs=4)
        self.assertEqual(result.strip(), 'merged')
        self.assertIn('character 1 to character 4', helper.call_args_list[0].args[0])
        self.assertIn('character 5 to character 8', helper.call_args_list[1].args[0])
        self.assertIn('Segment 0:', helper.call_args_list[2].args[0])
        self.assertIn('Segment 1:', helper.call_args_list[2].args[0])

    def test_native_init_loop_retry_and_single_call_history(self):
        from unittest.mock import patch
        self.agent.args.valid_format_entires = ['Research Plan and Status', 'Thought', 'Action', 'Action Input']
        self.agent.prompt_tool_names = self.agent.all_tool_names
        self.agent.tools_prompt = 'LEGACY ACTION EXAMPLES MUST NOT REACH NATIVE PROMPT'
        self.agent.data_context = 'data context'
        self.env.research_problem = 'forecast sales'
        for info in self.agent.action_infos.values():
            info.description = 'Execute or inspect a script'
        with patch.object(self.ns['Agent'], '__init__', lambda *a: None):
            self.ns['ResearchAgent'].__init__(self.agent, self.agent.args, self.env)
        self.assertTrue(self.agent.native_tools)
        self.assertNotIn('LEGACY ACTION EXAMPLES', self.agent.system_prompt)
        self.assertNotIn('Stop after Action Input.', self.agent.system_prompt)
        self.assertNotIn('forecast sales', self.agent.system_prompt)
        self.assertIn('forecast sales', self.agent.task_prompt)
        good = ('Research Plan and Status: verified plan\nThought: inspect\n'
                '<tool_call><function=execute_script><parameter=script_name>run.py</parameter>'
                '</function></tool_call>')
        self.responses = ['malformed duplicate\nAction: Run\nAction: Run', good, good]
        self.agent.args.agent_max_steps = 2
        self.agent.run(self.env)
        retry = self.prompts[1]
        self.assertNotIn('malformed duplicate', str(retry))
        self.assertEqual(retry[-1]['role'], 'user')
        self.assertIn('No tool executed', retry[-1]['content'])
        assistant = self.prompts[2][-2]
        self.assertNotIn('<tool_call>', assistant['content'])
        self.assertNotIn('Action:', assistant['content'])
        self.assertEqual(len(assistant['tool_calls']), 1)
        self.assertEqual(assistant['tool_calls'][0]['function']['name'], 'execute_script')
        self.assertEqual(self.agent.history_steps[0]['raw_assistant_response'], good)
        self.assertEqual(self.agent.history_steps[0]['action_input'], {'script_name': 'run.py'})

    def test_bad_arguments_retained_and_inspected_source_not_summarized(self):
        self.responses = [self.response('not json'), self.response(action='Inspect Script Lines')]
        self.agent.args.agent_max_steps = 2
        self.agent.run(self.env)
        self.assertEqual(len(self.agent.history_steps), 2)
        self.assertTrue(self.agent.history_steps[0]['parsing_error'])
        self.assertEqual(self.prompts[1][-1]['role'], 'user')
        self.assertNotIn('tool_calls', self.prompts[1][-2])
        self.agent.summarize_observation.assert_not_called()
        self.assertEqual(len(self.agent.history_steps[1]['observation']), 5100)


if __name__ == '__main__':
    unittest.main()
