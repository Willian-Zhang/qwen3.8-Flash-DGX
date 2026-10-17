#!/usr/bin/env python3
"""CPU-only production ParserManager composition regression.

Run in the installed vLLM image with its local tokenizer. No overlays, model
inference, network, tool execution, or project access. Unknown-call tests must
fail on the pre-forwarding image and pass after both wrapper layers forward the
inner tool parser's rejection state. Only test statuses/structural values print.
"""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from pydantic import TypeAdapter
from transformers import AutoTokenizer
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest, ResponsesResponse, StreamingResponsesResponse,
)
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.exceptions import GenerationError
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.parser_manager import ParserManager
from vllm.sampling_params import SamplingParams

MODEL = os.environ['QWEN_TOKENIZER']
TOKENIZER = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
TOOLS = [{'type':'function','name':'exec_command','description':'CPU fixture; never execute',
    'parameters':{'type':'object','properties':{'cmd':{'type':'string'}},'required':['cmd']}}]


def production_parser_class():
    return ParserManager.get_parser(tool_parser_name='qwen3_coder',
        reasoning_parser_name='qwen3', enable_auto_tools=True, model_name='qwen')


def call(name='exec_command', outer_closer=True):
    # This command is only XML fixture data. Never executed by this test.
    return ('<tool_call>\n<function=' + name + '>\n'
            '<parameter=cmd>fixture value</parameter>\n</function>\n'
            + ('</tool_call>' if outer_closer else ''))


def request(stream):
    return ResponsesRequest(model='qwen', input='CPU parser fixture', tools=TOOLS,
        tool_choice='auto', stream=stream, store=False)


class StreamFixture(OpenAIServingResponses):
    async def responses_full_generator(self, req, params, outputs, context, model_name,
                                       tokenizer, metadata, created_time=None):
        # Deliberately only stubs final response aggregation, not parser/processor/
        # error event handling. Nonstream extraction is independently tested below.
        return ResponsesResponse.from_request(req, params, model_name, created_time,
            output=[], status='completed')


def service():
    obj = object.__new__(StreamFixture)
    obj.use_harmony = False
    obj.parser = production_parser_class()
    obj.enable_auto_tools = True
    obj.enable_per_request_metrics = False
    obj.enable_log_outputs = False
    obj.request_logger = None
    return obj


async def stream(text, chunk_size, thinking=False):
    req = request(True)
    cls = production_parser_class()
    parser = cls(TOKENIZER, tools=req.tools,
                 chat_template_kwargs={'enable_thinking':thinking})
    ctx = SimpleContext(response_parser=parser)
    prompt = TOKENIZER.encode('<|im_start|>assistant\n' + ('<think>\n' if thinking else ''), add_special_tokens=False)
    tokens = TOKENIZER.encode(text, add_special_tokens=False)
    async def outputs():
        previous = ''
        for index in range(0, len(tokens), chunk_size):
            ids = tokens[index:index + chunk_size]
            current = TOKENIZER.decode(tokens[:index + chunk_size], skip_special_tokens=False)
            assert current.startswith(previous)
            delta = current[len(previous):]
            previous = current
            output = NS(text=delta, token_ids=ids,
                finish_reason='stop' if index + chunk_size >= len(tokens) else None,
                logprobs=None)
            ctx.last_output = NS(outputs=[output], prompt_token_ids=prompt)
            yield ctx
    events = [event async for event in service().responses_stream_generator(
        req, SamplingParams(max_tokens=8192), outputs(), ctx, 'qwen', TOKENIZER, None)]
    for event in events:
        # The installed StreamingResponsesResponse union omits several normal
        # delta-event classes. Validate actual runtime-class JSON for every event
        # and the patched union specifically for the failure event under test.
        type(event).model_validate_json(event.model_dump_json())
        if event.type == 'response.failed':
            TypeAdapter(StreamingResponsesResponse).validate_python(event)
    return events, parser


def nonstream(text):
    req = request(False)
    parser = production_parser_class()(TOKENIZER, tools=req.tools,
        chat_template_kwargs={'enable_thinking':False})
    completion = NS(text=text,
        token_ids=TOKENIZER.encode(text, add_special_tokens=False),
        finish_reason='stop', logprobs=None)
    result = service()._make_response_output_items(req, completion, TOKENIZER, parser=parser)
    return result, parser


class ProductionComposition(unittest.TestCase):
    def test_actual_manager_returns_production_composition(self):
        cls = production_parser_class()
        self.assertTrue(issubclass(cls, DelegatingParser))
        parser = cls(TOKENIZER, tools=request(False).tools,
                     chat_template_kwargs={'enable_thinking':False})
        self.assertIsNotNone(parser.tool_parser)
        self.assertIsNotNone(parser.reasoning_parser)
        self.assertTrue(hasattr(parser.tool_parser, '_parser_engine'))

    def test_inner_rejection_reaches_outer_parser(self):
        req = request(False)
        parser = production_parser_class()(TOKENIZER, tools=req.tools,
            chat_template_kwargs={'enable_thinking':False})
        text = call('write_file')
        parser.parse(text, req, enable_auto_tools=True,
                     model_output_token_ids=TOKENIZER.encode(text, add_special_tokens=False))
        self.assertTrue(parser.tool_parser._parser_engine.has_rejected_tool_calls)
        self.assertTrue(getattr(parser.tool_parser, 'has_rejected_tool_calls', False))
        self.assertTrue(getattr(parser, 'has_rejected_tool_calls', False))

    def test_unknown_closed_stream_fails(self):
        for size in (1, 7, 100000):
            with self.subTest(chunk_size=size):
                events, _ = asyncio.run(stream(call('write_file'), size))
                types = [v.type for v in events]
                self.assertIn('response.failed', types)
                self.assertNotIn('response.completed', types)
                self.assertFalse(any(v.type == 'response.output_item.added'
                    and v.item.type == 'function_call' for v in events))
                if size == 7:
                    path = Path(os.environ.get('CODEX_TEST_EXPORT_EVENTS',
                        str(Path(__file__).resolve().parent / 'manager-rejected-events.json')))
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, 'w') as out:
                        json.dump([v.model_dump(mode='json', by_alias=True) for v in events], out, indent=2)

    def test_unknown_function_closed_outer_eof_stream_fails(self):
        for size in (1, 7, 100000):
            with self.subTest(chunk_size=size):
                events, _ = asyncio.run(stream(call('write_file', False), size))
                self.assertIn('response.failed', [v.type for v in events])
                self.assertNotIn('response.completed', [v.type for v in events])

    def test_known_exact_and_namespace_stream_deliver(self):
        for name in ('exec_command','functions.exec_command','tools.exec_command'):
            for size in (1, 7, 100000):
                with self.subTest(name=name, chunk_size=size):
                    events, parser = asyncio.run(stream(call(name), size))
                    self.assertNotIn('response.failed', [v.type for v in events])
                    self.assertEqual(events[-1].type, 'response.completed')
                    calls = [v.item for v in events if v.type == 'response.output_item.added'
                             and v.item.type == 'function_call']
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].name, 'exec_command')
                    args = ''.join(v.delta for v in events
                        if v.type == 'response.function_call_arguments.delta')
                    self.assertEqual(json.loads(args), {'cmd':'fixture value'})
                    self.assertFalse(getattr(parser, 'has_rejected_tool_calls', False))

    def test_unknown_nonstream_raises(self):
        for closed in (True, False):
            with self.subTest(outer_closer=closed):
                with self.assertRaises(GenerationError):
                    nonstream(call('write_file', closed))

    def test_known_exact_and_namespace_nonstream_deliver(self):
        for name in ('exec_command','functions.exec_command','tools.exec_command'):
            with self.subTest(name=name):
                output, parser = nonstream(call(name))
                calls = [v for v in output if v.type == 'function_call']
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].name, 'exec_command')
                self.assertEqual(json.loads(calls[0].arguments), {'cmd':'fixture value'})
                self.assertFalse(getattr(parser, 'has_rejected_tool_calls', False))

    def test_plain_answers_pass_but_reasoning_only_stop_fails(self):
        for text, thinking in [('Done.',False), ('CPU fixture reasoning.</think>Done.',True),
                               ('CPU fixture reasoning only.',True)]:
            for size in (1,7,100000):
                with self.subTest(thinking=thinking, chunk_size=size):
                    events, parser = asyncio.run(stream(text, size, thinking))
                    if text == 'CPU fixture reasoning only.':
                        self.assertIn('response.failed', [v.type for v in events])
                        self.assertNotIn('response.completed', [v.type for v in events])
                    else:
                        self.assertNotIn('response.failed', [v.type for v in events])
                        self.assertEqual(events[-1].type, 'response.completed')
                    self.assertFalse(getattr(parser, 'has_rejected_tool_calls', False))

    def test_incomplete_unknown_opener_stop_without_output_fails(self):
        text = '<tool_call>\n<function=write_file>\n<parameter=cmd>unfinished fixture'
        for size in (1,7,100000):
            with self.subTest(chunk_size=size):
                events, parser = asyncio.run(stream(text, size))
                self.assertIn('response.failed', [v.type for v in events])
                self.assertNotIn('response.completed', [v.type for v in events])
                self.assertFalse(getattr(parser, 'has_rejected_tool_calls', False))

    def test_malformed_opener_then_known_call_recovers(self):
        text = '<tool_call><function=bad\n<tool_call>' + call()
        for size in (1,7,100000):
            with self.subTest(chunk_size=size):
                events, parser = asyncio.run(stream(text, size))
                self.assertNotIn('response.failed', [v.type for v in events])
                self.assertEqual(events[-1].type, 'response.completed')
                calls = [v.item for v in events if v.type == 'response.output_item.added'
                         and v.item.type == 'function_call']
                self.assertEqual(len(calls),1)
                self.assertEqual(calls[0].name,'exec_command')
                self.assertFalse(getattr(parser, 'has_rejected_tool_calls', False))

    def test_reasoning_then_unknown_tool_fails(self):
        text = 'CPU fixture reasoning.</think>Announcing fixture.\n' + call('write_file')
        for size in (1,7,100000):
            with self.subTest(chunk_size=size):
                events, parser = asyncio.run(stream(text, size, thinking=True))
                self.assertIn('response.failed', [v.type for v in events])
                self.assertNotIn('response.completed', [v.type for v in events])
                self.assertTrue(getattr(parser, 'has_rejected_tool_calls', False))

    def test_reasoning_then_known_qualified_tool_delivers(self):
        text = 'CPU fixture reasoning.</think>Announcing fixture.\n' + call('functions.exec_command')
        for size in (1,7,100000):
            with self.subTest(chunk_size=size):
                events, parser = asyncio.run(stream(text, size, thinking=True))
                self.assertNotIn('response.failed', [v.type for v in events])
                self.assertEqual(events[-1].type,'response.completed')
                calls = [v.item for v in events if v.type == 'response.output_item.added'
                         and v.item.type == 'function_call']
                self.assertEqual(len(calls),1)
                self.assertEqual(calls[0].name,'exec_command')
                self.assertFalse(getattr(parser, 'has_rejected_tool_calls', False))


if __name__ == '__main__':
    unittest.main(verbosity=2)
