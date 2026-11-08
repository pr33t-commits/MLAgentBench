import json
from types import SimpleNamespace
import unittest

from MLAgentBench.qwen_tools import parse_tool_response, tool_definitions, ToolCallStop, uses_native_tools
from MLAgentBench.research_context import build_messages


def create_script(datasets, objective, script_name, *, use_demand_engine=False, work_dir='.'):
    pass


def definitions():
    return tool_definitions(['Create Script (AI)', 'List Files'], {
        'Create Script (AI)': SimpleNamespace(description='Create a script', function=create_script,
            usage=dict.fromkeys(['datasets', 'objective', 'script_name', 'use_demand_engine'], 'description')),
        'List Files': SimpleNamespace(description='List files', function=None, usage={'dir_path': 'path'})})


def response(extra='', objective='Create features & check x < 3.', flag='false'):
    return ('Thought: build features\n<tool_call>\n<function=create_script_ai>\n'
            '<parameter=datasets>\n{"train": "train.csv"}\n</parameter>\n'
            f'<parameter=objective>\n{objective}\n</parameter>\n'
            '<parameter=script_name>\nfeatures.py\n</parameter>\n'
            + (f'<parameter=use_demand_engine>\n{flag}\n</parameter>\n' if flag is not None else '')
            + extra + '</function>\n</tool_call>')


class NativeToolTests(unittest.TestCase):
    def test_schema_aliases_types_defaults_and_hidden_runtime_args(self):
        schema = definitions()[0]['function']
        self.assertEqual(schema['name'], 'create_script_ai')
        parameters = schema['parameters']
        self.assertEqual(parameters['properties']['datasets']['type'], 'object')
        self.assertEqual(parameters['properties']['use_demand_engine']['type'], 'boolean')
        self.assertNotIn('use_demand_engine', parameters['required'])
        self.assertNotIn('work_dir', parameters['properties'])
        self.assertTrue(uses_native_tools('huggingface/Qwen/Qwen3.5-4B'))
        self.assertFalse(uses_native_tools('huggingface/Qwen/Qwen3-4B'))

    def test_typed_parameters_and_literal_strings(self):
        objective = 'Use {"window": 3}\nKeep \\n and "quoted" text & x < 3.'
        content, name, args = parse_tool_response(response(objective=objective), definitions())
        self.assertEqual(name, 'create_script_ai')
        self.assertEqual(content, 'Thought: build features')
        self.assertEqual(args['objective'], objective)
        self.assertEqual(args['datasets'], {'train': 'train.csv'})
        self.assertIs(args['use_demand_engine'], False)
        self.assertIs(parse_tool_response(response(flag='True'), definitions())[2]['use_demand_engine'], True)
        self.assertIs(parse_tool_response(response(flag=None), definitions())[2]['use_demand_engine'], False)

    def test_rejects_malformed_multiple_and_mixed_protocol_calls(self):
        good = response()
        cases = [good + good, good + '\nuser\n<tool_response>invented', good[:-5],
                 good.replace('create_script_ai', 'disabled_tool'),
                 response(extra='<parameter=script_name>other.py</parameter>'),
                 response(extra='<parameter=work_dir>/tmp</parameter>'),
                 response(flag='"false"'), response(flag='0'),
                 good.replace('{"train": "train.csv"}', '{"train": 42}'),
                 good.replace('{"train": "train.csv"}', '{"train":"a", "train":"b"}'),
                 'Action: Create Script (AI)\nAction Input: {}\n' + good,
                 '<think>hidden</think>' + good,
                 good.replace('<parameter=datasets>', 'broken<parameter=datasets>')]
        for text in cases:
            with self.subTest(text=text[:80]), self.assertRaises(ValueError):
                parse_tool_response(text, definitions())

    def test_empty_call_and_integer_parameters(self):
        tools = tool_definitions(['List Experiments', 'Inspect Script Lines'], {
            'List Experiments': SimpleNamespace(description='list', usage={}),
            'Inspect Script Lines': SimpleNamespace(description='inspect', usage={'start_line_number': 'line'})})
        self.assertEqual(parse_tool_response('<tool_call><function=list_experiments></function></tool_call>', tools)[2], {})
        fmt = '<tool_call><function=inspect_script_lines><parameter=start_line_number>{}</parameter></function></tool_call>'
        self.assertEqual(parse_tool_response(fmt.format(12), tools)[2]['start_line_number'], 12)
        for value in ['true', '1.5', '"12"']:
            with self.assertRaises(ValueError):
                parse_tool_response(fmt.format(value), tools)

    def test_history_has_only_one_call_and_omits_rejected_legacy_text(self):
        content, _, arguments = parse_tool_response(response(), definitions())
        history = [{'action': {'Thought': 'build features', 'Action': 'Create Script (AI)',
                               'Action Input': json.dumps(arguments)},
                    'raw_assistant_response': response(), 'assistant_content': content,
                    'action_input': arguments, 'tool_call_id': 'call_1', 'observation': 'saved'}]
        messages = build_messages('system', 'task', history, 3, 3, native_tools=True)
        self.assertEqual(messages[2]['content'], content)
        self.assertNotIn('<tool_call>', messages[2]['content'])
        self.assertEqual(len(messages[2]['tool_calls']), 1)
        self.assertEqual(messages[2]['tool_calls'][0]['function']['name'], 'create_script_ai')
        self.assertEqual(messages[3]['tool_call_id'], 'call_1')
        # Old checkpoints contain only raw mixed-format content and parsed entries.
        del history[0]['assistant_content']
        messages = build_messages('system', 'task', history, 3, 3, native_tools=True)
        self.assertNotIn('Action:', messages[2]['content'])
        history[0]['parsing_error'] = 'ambiguous'
        history[0]['observation'] = 'No tool executed'
        messages = build_messages('system', 'task', history, 3, 3, native_tools=True)
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'user'])

    def test_stop_uses_generated_suffix_not_prompt_or_observation_word(self):
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: ''.join(ids))
        stop = ToolCallStop(tokenizer, len('prompt</tool_call>'))
        prefix = 'prompt</tool_call>'
        self.assertFalse(stop([list(prefix)], None))
        self.assertFalse(stop([list(prefix + '<parameter=objective>Observation: foo')], None))
        self.assertFalse(stop([list(prefix + '</tool_cal')], None))
        self.assertTrue(stop([list(prefix + '</function>\n</tool_call>')], None))


if __name__ == '__main__':
    unittest.main()
