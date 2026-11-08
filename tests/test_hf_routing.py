"""Check HF loader selection without downloading weights or calling APIs."""
import ast
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import json
import tempfile
import unittest
from unittest.mock import Mock, patch


class Inputs(dict):
    @property
    def input_ids(self):
        return self['input_ids']

    def to(self, device):
        return self


class HFRoutingTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'MLAgentBench/LLM.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'complete_text_hf')
        self.tokenizer = Mock()
        self.tokenizer.side_effect = lambda value, **kwargs: (
            SimpleNamespace(input_ids=[]) if isinstance(value, list)
            else Inputs(input_ids=[[1, 2]]))
        self.tokenizer.batch_decode.return_value = ['answer']
        self.model = Mock()
        self.model.device = 'cuda'
        self.model.generate.return_value = SimpleNamespace(sequences=[[1, 2, 3]])
        self.causal = Mock()
        self.conditional = Mock()
        for loader in (self.causal, self.conditional):
            loader.from_pretrained.return_value.to.return_value = self.model
        self.config = Mock()
        self.config.from_pretrained.return_value = SimpleNamespace(model_type='qwen3_5')
        self.ns = dict(__package__='MLAgentBench', AutoConfig=self.config, AutoModelForCausalLM=self.causal,
                       AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=self.tokenizer)),
                       torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
                                                                 is_bf16_supported=lambda: True),
                                             bfloat16='bf16', float16='fp16', float32='fp32'),
                       loaded_hf_models={}, StoppingCriteriaList=list)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), self.ns)
        module = ModuleType('transformers')
        module.AutoModelForImageTextToText = self.conditional
        patcher = patch.dict(sys.modules, {'transformers': module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_qwen35_uses_conditional_loader_and_reuses_cached_model(self):
        for _ in range(2):
            self.assertEqual(self.ns['complete_text_hf']('hello', model='huggingface/Qwen/Qwen3.5-4B'), 'answer')
        self.conditional.from_pretrained.assert_called_once_with('Qwen/Qwen3.5-4B', torch_dtype='bf16')
        self.causal.from_pretrained.assert_not_called()
        self.assertFalse(self.model.generate.call_args.kwargs['output_scores'])
        self.assertFalse(self.tokenizer.apply_chat_template.call_args.kwargs['enable_thinking'])

    def test_other_models_keep_causal_loader(self):
        self.config.from_pretrained.return_value.model_type = 'qwen3'
        self.ns['complete_text_hf']('hello', model='huggingface/Qwen/Qwen3-4B')
        self.causal.from_pretrained.assert_called_once()
        self.conditional.from_pretrained.assert_not_called()

    def test_structured_context_reaches_template_and_is_logged(self):
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "Action: Run", "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "Run", "arguments": {}}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "done"}]
        self.tokenizer.apply_chat_template.return_value = "rendered context"
        self.ns['log_to_file'] = Mock()
        with tempfile.TemporaryDirectory() as directory:
            log = str(Path(directory) / 'step.log')
            self.ns['complete_text_hf'](messages, model='huggingface/Qwen/Qwen3.5-4B', log_file=log)
            record = json.loads(Path(log + '.context.jsonl').read_text())
        self.assertEqual(self.tokenizer.apply_chat_template.call_args.args[0], messages)
        self.assertEqual(record['messages'], messages)
        self.assertEqual(record['rendered_prompt'], 'rendered context')
        tokenization = next(c for c in self.tokenizer.call_args_list if c.args[0] == 'rendered context')
        self.assertFalse(tokenization.kwargs['add_special_tokens'])

    def test_context_limit_reserves_generation_tokens(self):
        self.tokenizer.model_max_length = 10
        self.tokenizer.apply_chat_template.side_effect = lambda items, **kw: 'x' * len(items)
        self.tokenizer.side_effect = lambda value, **kw: (
            SimpleNamespace(input_ids=[]) if isinstance(value, list) else
            Inputs(input_ids=[[1, 2]]) if kw.get('return_tensors') else
            SimpleNamespace(input_ids=list(value)))
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
        for n in range(2):
            messages.extend([{"role": "assistant", "content": str(n)},
                             {"role": "tool", "content": str(n)}])
        self.ns['complete_text_hf'](messages, model='huggingface/Qwen/Qwen3.5-4B', max_tokens_to_sample=6)
        rendered = self.tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(len(rendered), 4)
        self.assertEqual(rendered[-1]['content'], '1')

    def test_native_tools_go_to_template_not_generate_and_override_legacy_stop(self):
        from MLAgentBench.qwen_tools import ToolCallStop
        definitions = [{'type': 'function', 'function': {'name': 'list_files', 'parameters': {}}}]
        self.ns['complete_text_hf']([{'role': 'user', 'content': 'task'}],
            model='huggingface/Qwen/Qwen3.5-4B', tools=definitions, stop_sequences=['Observation:'])
        self.assertEqual(self.tokenizer.apply_chat_template.call_args.kwargs['tools'], definitions)
        self.assertNotIn('tools', self.model.generate.call_args.kwargs)
        criteria = self.model.generate.call_args.kwargs['stopping_criteria']
        self.assertEqual(len(criteria), 1)
        self.assertIsInstance(criteria[0], ToolCallStop)
        self.assertEqual(self.tokenizer.call_args.args[0], [])


if __name__ == '__main__':
    unittest.main()
