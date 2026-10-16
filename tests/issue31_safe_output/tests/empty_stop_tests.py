"""New structural empty-stop contract through actual ParserManager/Responses."""
import asyncio
import unittest
from types import SimpleNamespace as NS
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.sampling_params import SamplingParams
from vllm.exceptions import GenerationError
import eof_original as eof
import parser_manager_integration as manager

async def events(text, size, include, finish='stop', thinking=True, parser_enabled=True):
    request=eof.req(True,include)
    parser=eof.make_parser(request,thinking) if parser_enabled else None
    ctx=SimpleContext(response_parser=parser)
    prompt=manager.TOKENIZER.encode('<|im_start|>assistant\n'+('<think>\n' if thinking else ''),add_special_tokens=False)
    ids=manager.TOKENIZER.encode(text,add_special_tokens=False)
    async def outputs():
        previous=''
        for index in range(0,max(1,len(ids)),size):
            current=manager.TOKENIZER.decode(ids[:index+size],skip_special_tokens=False)
            assert current.startswith(previous)
            output=NS(text=current[len(previous):],token_ids=ids[index:index+size],finish_reason=finish if index+size>=len(ids) else None,logprobs=None)
            previous=current;ctx.last_output=NS(outputs=[output],prompt_token_ids=prompt)
            yield ctx
    return [e async for e in manager.service().responses_stream_generator(request,SamplingParams(max_tokens=8192),outputs(),ctx,'qwen',manager.TOKENIZER,None)]

def extract(text, include, finish='stop', parser_enabled=True):
    request=eof.req(False,include);parser=eof.make_parser(request) if parser_enabled else None
    output=NS(text=text,token_ids=manager.TOKENIZER.encode(text,add_special_tokens=False),finish_reason=finish,logprobs=None)
    return manager.service()._make_response_output_items(request,output,manager.TOKENIZER,parser=parser)

class EmptyStop(unittest.TestCase):
    def test_stream_empty_reasoning_only_and_closed_think_fail(self):
        for text in ('','Fixture reasoning only.','Fixture reasoning.</think>','</think><|im_end|>','   '):
            for size in (1,7,100000):
                for include in (True,False):
                    with self.subTest(text_kind=len(text),size=size,include=include):
                        kinds=[e.type for e in asyncio.run(events(text,size,include))]
                        self.assertIn('response.failed',kinds);self.assertNotIn('response.completed',kinds)
    def test_nonstream_empty_reasoning_only_and_closed_think_fail(self):
        for text in ('','Fixture reasoning only.','Fixture reasoning.</think>','</think><|im_end|>','   '):
            for include in (True,False):
                with self.subTest(text_kind=len(text),include=include):
                    with self.assertRaises(GenerationError):extract(text,include)
    def test_length_is_not_failed(self):
        for text in ('','Fixture reasoning only.','Fixture reasoning.</think>'):
            for include in (True,False):
                kinds=[e.type for e in asyncio.run(events(text,1,include,'length'))]
                self.assertNotIn('response.failed',kinds)
                extract(text,include,'length')
    def test_real_answer_and_valid_call_are_unchanged(self):
        for text in ('Fixture reasoning.</think>Done.','Fixture reasoning.</think>'+manager.call()):
            for include in (True,False):
                for size in (1,7,100000):
                    kinds=[e.type for e in asyncio.run(events(text,size,include))]
                    self.assertNotIn('response.failed',kinds)
                output=extract(text,include);self.assertTrue(any(x.type in ('message','function_call') for x in output))
    def test_unparsed_empty_stop_is_also_failed(self):
        for text in ('','  '):
            with self.assertRaises(GenerationError):extract(text,False,parser_enabled=False)
            kinds=[e.type for e in asyncio.run(events(text,1,False,parser_enabled=False))]
            self.assertIn('response.failed',kinds);self.assertNotIn('response.completed',kinds)
