"""Issue #56: exercise actual installed Chat producers on synthetic engine output.

Run alongside the original run_qa.py suite with the same tokenizer/image mounts.
No model inference, HTTP client or generated tool execution is performed.
"""

import asyncio
import json
import unittest

from chat_fixture import CLASS, TOKENIZER, TOOLS, call, full, run
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.qwen3 import Qwen3Parser


class ChatCompatibility(unittest.TestCase):
    def variants(self, text, thinking=False):
        for size in (1, 7, 100000):
            with self.subTest(size=size):
                result = asyncio.run(run(text, thinking, size))
                self.assertTrue(result['done'])
                self.assertEqual(result['errors'], [])
                yield result

    def test_unknown_call_reaches_chat_client_instead_of_empty_stop(self):
        for result in self.variants(call('unknown_tool')):
            self.assertEqual(result['finishes'], ['tool_calls'])
            self.assertEqual(result['calls'][0]['name'], 'unknown_tool')
            self.assertEqual(json.loads(result['calls'][0]['arguments']),
                             {'cmd': 'synthetic value'})

    def test_sequential_chat_calls_have_contiguous_indices(self):
        text = call('unknown_tool') + call('exec_command') + call('another_tool')
        for result in self.variants(text):
            self.assertEqual(list(result['calls']), [0, 1, 2])
            self.assertEqual([c['name'] for c in result['calls'].values()],
                             ['unknown_tool', 'exec_command', 'another_tool'])

    def test_offered_namespace_alias_still_resolves(self):
        for name in ('functions.exec_command', 'tools.exec_command'):
            for result in self.variants(call(name)):
                self.assertEqual(result['calls'][0]['name'], 'exec_command')

    def test_unknown_name_is_never_remapped_to_an_offered_tool(self):
        for name in ('write_file', 'other.exec_command'):
            for result in self.variants(call(name)):
                self.assertEqual(result['calls'][0]['name'], name)

    def test_closed_reasoning_example_stays_reasoning(self):
        example = 'Example:\n' + call('unknown_tool') + '\nStill thinking.'
        for result in self.variants(example + '</think>Final answer.', True):
            self.assertEqual(result['reasoning'], example)
            self.assertEqual(result['content'], 'Final answer.')
            self.assertEqual(result['calls'], {})
            self.assertEqual(result['finishes'], ['stop'])

    def test_implicit_reasoning_end_keeps_tool_transition(self):
        for result in self.variants('Use a tool.\n' + call('unknown_tool'), True):
            self.assertEqual(result['calls'][0]['name'], 'unknown_tool')
            self.assertEqual(result['finishes'], ['tool_calls'])

    def test_full_chat_producer_matches_streaming_tool_name_contract(self):
        for name in ('unknown_tool', 'functions.exec_command'):
            response = asyncio.run(full(call(name)))
            choice = response.choices[0]
            self.assertEqual(choice.finish_reason, 'tool_calls')
            expected = 'exec_command' if name.startswith('functions.') else name
            self.assertEqual(choice.message.tool_calls[0].function.name, expected)

    def test_concurrent_chat_streams_are_independent(self):
        async def collect():
            return await asyncio.gather(
                run(call('unknown_tool'), False, 1),
                run('Unrelated answer.', False, 7),
            )
        tool, answer = asyncio.run(collect())
        self.assertEqual(tool['calls'][0]['name'], 'unknown_tool')
        self.assertEqual(answer['content'], 'Unrelated answer.')
        for result in (tool, answer):
            self.assertTrue(result['done'])
            self.assertEqual(result['errors'], [])

    def test_responses_validation_stays_strict_and_per_request(self):
        response_tools = [dict(type='function', **tool['function']) for tool in TOOLS]
        responses = ResponsesRequest(model='qwen', input='fixture', tools=response_tools)
        chat = ChatCompletionRequest(model='qwen', messages=[], tools=TOOLS)
        for request, expected in ((chat, True), (responses, False), (chat, True)):
            parser = Qwen3Parser(TOKENIZER, tools=request.tools,
                                 chat_template_kwargs={'enable_thinking': False})
            parsed = parser.extract_tool_calls(call('unknown_tool'), request)
            self.assertEqual(bool(parsed.tool_calls), expected)
        # Reuse one instance too: request policy must not stick or mutate the
        # cached parser config shared with unrelated/concurrent parsers.
        parser = Qwen3Parser(TOKENIZER, tools=responses.tools,
                             chat_template_kwargs={'enable_thinking': False})
        for request, expected in ((chat, True), (responses, False), (chat, True)):
            parsed = parser.extract_tool_calls(call('unknown_tool'), request)
            self.assertEqual(bool(parsed.tool_calls), expected)
        self.assertTrue(parser.parser_engine_config.validate_tool_names)

    def test_rejected_tool_warning_includes_name_without_arguments(self):
        tools = [dict(type='function', **tool['function']) for tool in TOOLS]
        request = ResponsesRequest(model='qwen', input='fixture', tools=tools)
        parser = Qwen3Parser(TOKENIZER, tools=request.tools,
                             chat_template_kwargs={'enable_thinking': False})
        with self.assertLogs('vllm.parser.engine.parser_engine', level='WARNING') as logs:
            parser.extract_tool_calls(call('unknown_tool'), request)
        warning = '\n'.join(logs.output)
        self.assertIn('unknown_tool', warning)
        self.assertNotIn('synthetic value', warning)


if __name__ == '__main__':
    unittest.main(verbosity=2)
