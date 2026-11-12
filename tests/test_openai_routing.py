import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

class OpenAIRoutingTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'MLAgentBench/LLM.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'complete_text_openai')
        self.api = SimpleNamespace(ChatCompletion=SimpleNamespace(create=Mock(return_value={
            'choices': [{'message': {'content': 'answerObservation: invented'}, 'finish_reason': 'stop'}]})),
            Completion=SimpleNamespace(create=Mock(return_value={'choices': [{'text': 'legacy'}]})))
        self.responses = Mock(return_value={'id': 'resp_test', 'status': 'completed',
            'usage': {'input_tokens': 20, 'output_tokens': 30, 'total_tokens': 50,
                      'output_tokens_details': {'reasoning_tokens': 25}},
            'output': [{'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': 'A summary'}]},
                       {'type': 'message', 'role': 'assistant', 'content': [
                           {'type': 'output_text', 'text': 'answerObservation: invented'}]}]})
        self.logger = Mock()
        ns = {'request_openai_responses': self.responses, 'openai': self.api, 'LLMError': RuntimeError, 'log_to_file': self.logger}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        self.call = ns['complete_text_openai']
        self.default_limit = self.call.__defaults__[2]

    def test_reasoning_routes_and_parameters(self):
        for model in ('gpt-5', 'gpt-5-mini', 'gpt-5.6-luna', 'o3-mini'):
            self.assertEqual(self.call('prompt', model=model, stop_sequences=['Observation:']), 'answer')
            request = self.responses.call_args.args[0]
            self.assertEqual(request['model'], model)
            self.assertEqual(request['max_output_tokens'], self.default_limit)
            self.assertFalse({'temperature','stop','max_tokens'} & request.keys())
        self.api.Completion.create.assert_not_called()
        self.api.ChatCompletion.create.assert_not_called()

    def test_existing_chat_parameters(self):
        self.call('prompt', model='gpt-4.1-mini', stop_sequences=['Observation:'])
        request = self.api.ChatCompletion.create.call_args.kwargs
        self.assertEqual(request['max_tokens'], self.default_limit)
        self.assertEqual(request['stop'], ['Observation:'])
        self.assertEqual(request['temperature'], .2)

    def test_legacy_completion(self):
        for model in ('text-davinci-003', 'gpt-3.5-turbo-instruct'):
            self.assertEqual(self.call('prompt', model=model), 'legacy')
        self.api.ChatCompletion.create.assert_not_called()

    def test_token_override_and_local_stop(self):
        self.assertEqual(self.call('prompt', model='gpt-5', max_completion_tokens=8000,
                                  stop='Observation:', top_p=.9), 'answer')
        request = self.responses.call_args.args[0]
        self.assertEqual(request['max_output_tokens'], 8000)
        self.assertNotIn('top_p', request)

    def test_empty_reasoning_output(self):
        self.responses.return_value = {'status': 'incomplete', 'output': [],
            'incomplete_details': {'reason': 'max_output_tokens'},
            'usage': {'output_tokens_details': {'reasoning_tokens': 1500}}}
        with self.assertRaisesRegex(RuntimeError, 'budget includes reasoning'):
            self.call('prompt', model='gpt-5', log_file='test.log')
        self.assertEqual(self.logger.call_args.kwargs['api_metadata']['usage']
                         ['output_tokens_details']['reasoning_tokens'], 1500)

    def test_summary_and_actual_usage_do_not_enter_completion(self):
        self.assertEqual(self.call('prompt', model='gpt-5', log_file='test.log',
                                  reasoning_effort='low', stop='Observation:'), 'answer')
        request = self.responses.call_args.args[0]
        self.assertEqual(request['reasoning'], {'summary': 'auto', 'effort': 'low'})
        self.assertFalse(request['store'])
        metadata = self.logger.call_args.kwargs['api_metadata']
        self.assertEqual(metadata['reasoning_summaries'], ['A summary'])
        self.assertEqual(metadata['usage']['output_tokens_details']['reasoning_tokens'], 25)
        self.assertEqual(metadata['response_id'], 'resp_test')

class UsageLoggingTests(unittest.TestCase):
    def test_logs_api_usage_and_summary_separately_from_estimates(self):
        import json
        import tempfile
        path = Path(__file__).resolve().parents[1] / 'MLAgentBench/LLM.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'log_to_file')
        ns = {'enc': SimpleNamespace(encode=lambda text, **kwargs: list(text)),
              'HUMAN_PROMPT': 'Human:', 'AI_PROMPT': 'Assistant:'}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        metadata = {'usage': {'input_tokens': 7, 'output_tokens': 10,
                    'output_tokens_details': {'reasoning_tokens': 8}},
                    'reasoning_summaries': ['Summary: cafe']}
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'request.log'
            ns['log_to_file'](log, 'prompt', 'answer', 'gpt-5', 100, api_metadata=metadata)
            text = log.read_text(encoding='utf-8')
            self.assertIn('Estimated visible completion tokens', text)
            self.assertIn('Summary: cafe', text)
            self.assertEqual(json.loads(Path(str(log) + '.usage.jsonl').read_text(encoding='utf-8')), metadata)

    def test_logs_special_tokens_in_prompt_and_completion(self):
        import tempfile
        import tiktoken
        path = Path(__file__).resolve().parents[1] / 'MLAgentBench/LLM.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'log_to_file')
        encoding = tiktoken.Encoding(name='test', pat_str=r'(?s:.)',
                                     mergeable_ranks={bytes([i]): i for i in range(256)},
                                     special_tokens={'<|endoftext|>': 256})
        ns = {'enc': encoding,
              'HUMAN_PROMPT': 'Human:', 'AI_PROMPT': 'Assistant:'}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'request.log'
            ns['log_to_file'](log, 'prompt <|endoftext|>', 'answer <|endoftext|>', 'qwen', 100)
            text = log.read_text(encoding='utf-8')
            self.assertIn('prompt <|endoftext|>', text)
            self.assertIn('answer <|endoftext|>', text)
            self.assertIn('Estimated visible completion tokens', text)

if __name__ == '__main__':
    unittest.main()
