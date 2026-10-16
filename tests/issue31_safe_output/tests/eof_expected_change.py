"""Synthetic EOF fixture through actual ParserManager/Responses composition."""
import asyncio
import json
import os
from pathlib import Path
import unittest
from types import SimpleNamespace as NS
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import GenerationError
from vllm.sampling_params import SamplingParams
import parser_manager_integration as manager

TOKENIZER = manager.TOKENIZER
FIXTURE = 'Fixture note: `<tool_call>` control token — model emits'
SIGNAL = 'has_unresolved_reasoning_tool_candidate'


def req(stream=False, include_reasoning=True):
    return ResponsesRequest(model='qwen', input='Synthetic CPU fixture only',
        tools=manager.TOOLS, tool_choice='auto', stream=stream, store=False,
        include_reasoning=include_reasoning)


def make_parser(request, thinking=True):
    return manager.production_parser_class()(TOKENIZER, tools=request.tools,
        chat_template_kwargs={'enable_thinking':thinking})


def extract(text, finish='stop', include_reasoning=True):
    request = req(include_reasoning=include_reasoning)
    parser = make_parser(request)
    output = NS(text=text, token_ids=TOKENIZER.encode(text,add_special_tokens=False),
        finish_reason=finish, logprobs=None)
    return manager.service()._make_response_output_items(request, output, TOKENIZER, parser=parser)


async def stream(text=FIXTURE, size=7, finish='stop', include_reasoning=True, thinking=True):
    request = req(True, include_reasoning)
    parser = make_parser(request, thinking)
    context = SimpleContext(response_parser=parser)
    prompt = TOKENIZER.encode('<|im_start|>assistant\n' + ('<think>\n' if thinking else ''),
        add_special_tokens=False)
    tokens = TOKENIZER.encode(text,add_special_tokens=False)
    async def outputs():
        previous = ''
        for index in range(0,len(tokens),size):
            current = TOKENIZER.decode(tokens[:index+size],skip_special_tokens=False)
            assert current.startswith(previous)
            fragment=current[len(previous):]; previous=current
            output=NS(text=fragment,token_ids=tokens[index:index+size],
                finish_reason=finish if index+size>=len(tokens) else None,logprobs=None)
            context.last_output=NS(outputs=[output],prompt_token_ids=prompt)
            yield context
    events=[e async for e in manager.service().responses_stream_generator(
        request,SamplingParams(max_tokens=8192),outputs(),context,'qwen',TOKENIZER,None)]
    for event in events:
        type(event).model_validate_json(event.model_dump_json())
    return events, parser


class EOFIntegration(unittest.TestCase):
    def test_manager_stream_eof_preserves_reasoning_channel(self):
        for size in (1,2,7,100000):
            with self.subTest(size=size):
                request=req(); parser=make_parser(request)
                ids=TOKENIZER.encode(FIXTURE,add_special_tokens=False)
                prompt=TOKENIZER.encode('<|im_start|>assistant\n<think>\n',add_special_tokens=False)
                content=[];reasoning=[];calls=[]; previous=''
                for index in range(0,len(ids),size):
                    current=TOKENIZER.decode(ids[:index+size],skip_special_tokens=False)
                    fragment=current[len(previous):];previous=current
                    delta=parser.parse_delta(fragment,ids[index:index+size],request,
                        prompt_token_ids=prompt,finished=index+size>=len(ids))
                    if delta:
                        content.append(delta.content or '');reasoning.append(delta.reasoning or '')
                        calls.extend(delta.tool_calls or [])
                self.assertEqual(''.join(content),'')
                self.assertEqual(''.join(reasoning),FIXTURE)
                self.assertEqual(calls,[])
                self.assertTrue(getattr(parser,SIGNAL,False))

    def test_actual_stream_stop_fails_no_completion_or_content(self):
        for size in (1,2,7,100000):
            with self.subTest(size=size):
                events,parser=asyncio.run(stream(size=size))
                kinds=[e.type for e in events]
                self.assertIn('response.failed',kinds)
                self.assertNotIn('response.completed',kinds)
                self.assertNotIn('response.output_text.delta',kinds)
                self.assertTrue(getattr(parser,SIGNAL,False))
                if size==7:
                    path=Path(__file__).resolve().parent/'eof-failed-events.json'
                    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
                    with os.fdopen(fd,'w') as out:
                        json.dump([e.model_dump(mode='json',by_alias=True) for e in events
                            if e.type in ('response.created','response.in_progress','response.failed')],out,indent=2)

    def test_stream_hidden_reasoning_also_fails(self):
        events,_=asyncio.run(stream(include_reasoning=False))
        self.assertIn('response.failed',[e.type for e in events])
        self.assertNotIn('response.completed',[e.type for e in events])

    def test_stop_with_terminal_token_also_fails(self):
        events,_=asyncio.run(stream(FIXTURE+'<|im_end|>'))
        self.assertIn('response.failed',[e.type for e in events])
        self.assertNotIn('response.output_text.delta',[e.type for e in events])

    def test_nonstream_stop_also_fails(self):
        for include in (True,False):
            with self.subTest(include_reasoning=include):
                with self.assertRaises(GenerationError):
                    extract(FIXTURE,include_reasoning=include)

    def test_length_finish_keeps_existing_nonfailure_behavior(self):
        events,parser=asyncio.run(stream(finish='length'))
        self.assertNotIn('response.failed',[e.type for e in events])
        self.assertNotIn('response.output_text.delta',[e.type for e in events])
        self.assertTrue(getattr(parser,SIGNAL,False))
        # Only assert extraction/nonfailure; the fixture stubs final aggregation,
        # so it cannot establish production incomplete-status serialization.
        output=extract(FIXTURE,finish='length')
        self.assertFalse(any(v.type=='message' for v in output))

    def test_explicit_think_end_preserves_real_final_answer(self):
        text=FIXTURE+'</think>Done.'
        for size in (1,7,100000):
            with self.subTest(size=size):
                events,parser=asyncio.run(stream(text,size=size))
                self.assertNotIn('response.failed',[e.type for e in events])
                self.assertEqual(''.join(e.delta for e in events if e.type=='response.output_text.delta'),'Done.')
                self.assertFalse(getattr(parser,SIGNAL,False))

    def test_real_implicit_tool_call_still_delivered(self):
        for name in ('exec_command',):
            events,parser=asyncio.run(stream('Fixture reasoning.\n'+manager.call(name)))
            calls=[e.item for e in events if e.type=='response.output_item.added' and e.item.type=='function_call']
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0].name,'exec_command')
            self.assertNotIn('response.failed',[e.type for e in events])
            self.assertFalse(getattr(parser,SIGNAL,False))

    def test_closed_unknown_tool_remains_failed(self):
        events,_=asyncio.run(stream('Fixture reasoning.</think>\n'+manager.call('write_file')))
        self.assertIn('response.failed',[e.type for e in events])
        self.assertNotIn('response.completed',[e.type for e in events])

    def test_plain_reasoning_only_now_explicitly_failed(self):
        events,parser=asyncio.run(stream('Fixture reasoning only.'))
        self.assertIn('response.failed',[e.type for e in events])
        self.assertNotIn('response.completed',[e.type for e in events])
        self.assertFalse(getattr(parser,SIGNAL,False))

    def test_open_reasoning_fence_fails_without_tool_or_content(self):
        text='```xml\n'+manager.call()
        events,parser=asyncio.run(stream(text))
        self.assertNotIn('response.output_text.delta',[e.type for e in events])
        self.assertFalse(any(e.type=='response.output_item.added' and e.item.type=='function_call' for e in events))
        self.assertIn('response.failed',[e.type for e in events])
        self.assertTrue(getattr(parser,SIGNAL,False))

    def test_fenced_marker_length_finish_does_not_fail(self):
        events,parser=asyncio.run(stream('```xml\n'+manager.call(),finish='length'))
        self.assertNotIn('response.failed',[e.type for e in events])
        self.assertNotIn('response.output_text.delta',[e.type for e in events])
        self.assertTrue(getattr(parser,SIGNAL,False))

    def test_closed_fence_and_explicit_think_end_preserve_answer(self):
        text='```xml\n'+manager.call()+'\n```\n</think>Done.'
        events,parser=asyncio.run(stream(text))
        self.assertNotIn('response.failed',[e.type for e in events])
        self.assertEqual(''.join(e.delta for e in events if e.type=='response.output_text.delta'),'Done.')
        self.assertFalse(getattr(parser,SIGNAL,False))

    def test_bare_and_partial_function_headers_fail(self):
        for suffix in ('<tool_call>','<tool_call>\n<function=exec_command'):
            with self.subTest(partial_header='function' in suffix):
                events,parser=asyncio.run(stream('CPU fixture reasoning. '+suffix))
                self.assertIn('response.failed',[e.type for e in events])
                self.assertNotIn('response.output_text.delta',[e.type for e in events])
                self.assertTrue(getattr(parser,SIGNAL,False))

    def test_nonreasoning_output_not_newly_failed(self):
        events,_=asyncio.run(stream('Visible fixture text.',thinking=False))
        self.assertNotIn('response.failed',[e.type for e in events])
        self.assertEqual(''.join(e.delta for e in events if e.type=='response.output_text.delta'),'Visible fixture text.')
