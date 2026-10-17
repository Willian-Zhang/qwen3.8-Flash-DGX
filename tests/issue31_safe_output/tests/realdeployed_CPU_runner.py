"""Actual installed vLLM/Pydantic SSE synthetic tests; no inference."""
import asyncio, importlib, json, os, sys, unittest
from pathlib import Path
from http import HTTPStatus
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest,ResponsesResponse,StreamingResponsesResponse
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.exceptions import GenerationError
from vllm.sampling_params import SamplingParams
from vllm.parser.qwen3 import Qwen3Parser
from pydantic import TypeAdapter
from base_regression import TOKENIZER,TOOLS,call
import deployed_regression

class Fixture(OpenAIServingResponses):
    async def responses_full_generator(self,request,sampling_params,result_generator,context,model_name,tokenizer,metadata,created_time=None):
        if getattr(self,'force_final_error',False):
            return self.create_error_response('Fixture final tool rejection',status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        return ResponsesResponse.from_request(request,sampling_params,model_name,created_time,output=[],status='completed')

def fixture():
    s=object.__new__(Fixture)
    s.use_harmony=False;s.parser=Qwen3Parser;s.enable_auto_tools=True
    s.enable_per_request_metrics=False;s.enable_log_outputs=False;s.request_logger=None
    return s

async def capture(text,final_error=False,chunk_size=7):
    req=ResponsesRequest(model='qwen',input='CPU fixture',tools=TOOLS,stream=True,store=False)
    params=SamplingParams(max_tokens=8192)
    p=Qwen3Parser(TOKENIZER,tools=req.tools,chat_template_kwargs={'enable_thinking':False})
    ctx=SimpleContext(response_parser=p)
    prompt_ids=TOKENIZER.encode('<|im_start|>assistant\n',add_special_tokens=False)
    ids=TOKENIZER.encode(text,add_special_tokens=False)
    async def outputs():
        previous=''
        for i in range(0,len(ids),chunk_size):
            current=TOKENIZER.decode(ids[:i+chunk_size],skip_special_tokens=False)
            chunk=current[len(previous):];previous=current
            completion=NS(text=chunk,token_ids=ids[i:i+chunk_size],finish_reason='stop' if i+chunk_size>=len(ids) else None,logprobs=None)
            ctx.last_output=NS(outputs=[completion],prompt_token_ids=prompt_ids)
            yield ctx
    s=fixture();s.force_final_error=final_error
    return [e async for e in s.responses_stream_generator(req,params,outputs(),ctx,'qwen',TOKENIZER,None)]

class ActualSSE(unittest.TestCase):
    def test_actual_error_shape(self):
        s=fixture();raw=s._convert_generation_error_to_streaming_response(GenerationError('CPU fixture rejection'))
        e=TypeAdapter(StreamingResponsesResponse).validate_json(raw)
        self.assertEqual(e.type,'error')
        print('Validated actual installed SSE error shape; fields='+','.join(sorted(json.loads(raw))))
    def test_unknown_write_file_no_completed(self):
        events=asyncio.run(capture(call('fixture').replace('<function=exec_command>','<function=write_file>')))
        types=[e.type for e in events]
        self.assertIn('response.failed',types);self.assertNotIn('response.completed',types)
        for e in events:TypeAdapter(StreamingResponsesResponse).validate_python(e)
        failed=next(e for e in events if e.type=='response.failed')
        self.assertEqual(failed.response.status,'failed')
        self.assertEqual(failed.response.error.code,'server_error')
        self.assertIn('absent from offered tools',failed.response.error.message)
    def test_unknown_whitespace_no_completed(self):
        events=asyncio.run(capture(call('fixture').replace('<function=exec_command>','<function= write_file >')))
        types=[e.type for e in events]
        self.assertIn('response.failed',types);self.assertNotIn('response.completed',types)
    def test_normal_namespace_delivered(self):
        events=asyncio.run(capture(call('fixture').replace('<function=exec_command>','<function=functions.exec_command>')))
        types=[e.type for e in events]
        self.assertIn('response.function_call_arguments.delta',types);self.assertEqual(types[-1],'response.completed');self.assertNotIn('response.failed',types)
        added=[e for e in events if e.type=='response.output_item.added' and e.item.type=='function_call']
        self.assertEqual(added[0].item.name,'exec_command')
    def test_malformed_opener_recovers(self):
        text='<tool_call><function=bad\n<tool_call>'+call('fixture')
        events=asyncio.run(capture(text));types=[e.type for e in events]
        self.assertIn('response.function_call_arguments.delta',types);self.assertEqual(types[-1],'response.completed');self.assertNotIn('response.failed',types)
    def test_final_error_response_no_completed(self):
        events=asyncio.run(capture('Done.',True));types=[e.type for e in events]
        self.assertIn('response.failed',types);self.assertNotIn('response.completed',types)
    def test_unknown_closed_function_eof_fails_across_chunks(self):
        text=call('fixture').replace('<function=exec_command>','<function=write_file>').removesuffix('</tool_call>')
        for chunk_size in (1,7,100000):
            with self.subTest(chunk_size=chunk_size):
                events=asyncio.run(capture(text,chunk_size=chunk_size))
                types=[e.type for e in events]
                self.assertIn('response.failed',types)
                self.assertNotIn('response.completed',types)
                for e in events:TypeAdapter(StreamingResponsesResponse).validate_python(e)

    def test_known_closed_function_eof_delivers_across_chunks(self):
        text=call('fixture').replace('<function=exec_command>','<function=functions.exec_command>').removesuffix('</tool_call>')
        for chunk_size in (1,7,100000):
            with self.subTest(chunk_size=chunk_size):
                events=asyncio.run(capture(text,chunk_size=chunk_size))
                types=[e.type for e in events]
                self.assertIn('response.function_call_arguments.delta',types)
                self.assertNotIn('response.failed',types)
                self.assertEqual(types[-1],'response.completed')

    def test_malformed_recovery_across_chunks(self):
        text='<tool_call><function=bad\n<tool_call>'+call('fixture')
        for chunk_size in (1,7,100000):
            with self.subTest(chunk_size=chunk_size):
                events=asyncio.run(capture(text,chunk_size=chunk_size))
                types=[e.type for e in events]
                self.assertIn('response.function_call_arguments.delta',types)
                self.assertNotIn('response.failed',types)
                self.assertEqual(types[-1],'response.completed')

    def test_nonstream_unknown_closed_function_eof_raises(self):
        s=fixture();req=ResponsesRequest(model='qwen',input='fixture',tools=TOOLS)
        p=Qwen3Parser(TOKENIZER,tools=req.tools,chat_template_kwargs={'enable_thinking':False})
        text=call('fixture').replace('<function=exec_command>','<function=write_file>').removesuffix('</tool_call>')
        with self.assertRaises(GenerationError):
            s._make_response_output_items(req,NS(text=text,token_ids=TOKENIZER.encode(text,add_special_tokens=False),finish_reason='stop',logprobs=None),TOKENIZER,parser=p)

    def test_final_reparse_unknown_raises(self):
        s=fixture();req=ResponsesRequest(model='qwen',input='fixture',tools=TOOLS)
        p=Qwen3Parser(TOKENIZER,tools=req.tools,chat_template_kwargs={'enable_thinking':False})
        text=call('fixture').replace('<function=exec_command>','<function=write_file>')
        with self.assertRaises(GenerationError):
            s._make_response_output_items(req,NS(text=text,token_ids=TOKENIZER.encode(text,add_special_tokens=False),finish_reason='stop',logprobs=None),TOKENIZER,parser=p)

if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(deployed_regression),unittest.defaultTestLoader.loadTestsFromTestCase(ActualSSE)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
