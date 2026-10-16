"""Actual Responses generator+aggregation, synthetic engine outputs, CPU only."""
import asyncio
import json
import os
from pathlib import Path
import unittest
from pydantic import TypeAdapter
from transformers import AutoTokenizer
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest, StreamingResponsesResponse
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.parser.parser_manager import ParserManager
from vllm.sampling_params import SamplingParams

TOKENIZER=AutoTokenizer.from_pretrained(os.environ['QWEN_TOKENIZER'],local_files_only=True)
TOOLS=[{'type':'function','name':'exec_command','parameters':{'type':'object','properties':{'cmd':{'type':'string'}},'required':['cmd']}}]
CLASS=ParserManager.get_parser(tool_parser_name='qwen3_coder',reasoning_parser_name='qwen3',enable_auto_tools=True)

def fixture(include=True,stream=True,thinking=True):
    req=ResponsesRequest(model='qwen',input='Synthetic CPU fixture',tools=TOOLS,stream=stream,store=False,include_reasoning=include)
    parser=CLASS(TOKENIZER,tools=req.tools,chat_template_kwargs={'enable_thinking':thinking})
    ctx=SimpleContext(response_parser=parser)
    service=object.__new__(OpenAIServingResponses)
    service.use_harmony=False;service.parser=CLASS;service.enable_auto_tools=True
    service.enable_per_request_metrics=False;service.enable_log_outputs=False;service.request_logger=None;service.tool_server=None
    return req,SamplingParams(max_tokens=8192),ctx,service

def engine_outputs(ctx,text,finish,size,thinking=True):
    ids=TOKENIZER.encode(text,add_special_tokens=False)
    prompt='<|im_start|>assistant\n'+('<think>\n' if thinking else '')
    promptids=TOKENIZER.encode(prompt,add_special_tokens=False)
    async def outputs():
        previous=''
        for index in range(0,max(1,len(ids)),size):
            current=TOKENIZER.decode(ids[:index+size],skip_special_tokens=False)
            assert current.startswith(previous)
            last=index+size>=len(ids)
            completion=CompletionOutput(index=0,text=current[len(previous):],token_ids=ids[index:index+size],cumulative_logprob=None,logprobs=None,finish_reason=finish if last else None)
            output=RequestOutput(request_id='synthetic-cpu',prompt=prompt,prompt_token_ids=promptids,prompt_logprobs=None,outputs=[completion],finished=last)
            previous=current;ctx.append_output(output);yield ctx
    return outputs(),len(ids)

async def actual_stream(text,finish,size=7,include=True,thinking=True):
    request,params,ctx,service=fixture(include,True,thinking)
    outputs,total=engine_outputs(ctx,text,finish,size,thinking)
    events=[event async for event in service.responses_stream_generator(request,params,outputs,ctx,'qwen',TOKENIZER,None,created_time=1)]
    # No responses_full_generator override: final body/usage is the real producer.
    for event in events:type(event).model_validate_json(event.model_dump_json())
    terminal=events[-1]
    TypeAdapter(StreamingResponsesResponse).validate_json(terminal.model_dump_json())
    assert [e.sequence_number for e in events]==list(range(len(events)))
    return events,total

async def actual_full(text,finish,size=7,include=True):
    request,params,ctx,service=fixture(include,False)
    outputs,total=engine_outputs(ctx,text,finish,size)
    response=await service.responses_full_generator(request,params,outputs,ctx,'qwen',TOKENIZER,None,created_time=1)
    return response,total

class LengthProducer(unittest.TestCase):
    def test_length_visible_real_aggregation_incomplete_event(self):
        for size in (1,7,100000):
            events,total=asyncio.run(actual_stream('<think>Fixture.</think>Partial', 'length',size))
            kinds=[e.type for e in events];last=events[-1]
            self.assertEqual(last.type,'response.incomplete');self.assertNotIn('response.completed',kinds)
            self.assertEqual(last.response.status,'incomplete');self.assertEqual(last.response.incomplete_details.reason,'max_output_tokens')
            self.assertEqual(last.response.usage.output_tokens,total)
            self.assertEqual(''.join(e.delta for e in events if e.type=='response.output_text.delta'),'Partial')
    def test_length_reasoning_empty_and_partial_tool_are_incomplete(self):
        for text in ('','<think>Fixture only.','<think>Fixture.</think><tool_call>\n<function=exec_command>\n<parameter=cmd>partial'):
            for include in (True,False):
                events,total=asyncio.run(actual_stream(text,'length',7,include))
                self.assertEqual(events[-1].type,'response.incomplete');self.assertEqual(events[-1].response.status,'incomplete')
                self.assertEqual(events[-1].response.usage.output_tokens,total)
                self.assertNotIn('response.failed',[e.type for e in events])
                self.assertNotIn('response.completed',[e.type for e in events])
    def test_nonstream_length_preserves_aggregation_status_and_usage(self):
        for text in ('','<think>Fixture only.','<think>Fixture.</think>Partial'):
            for include in (True,False):
                response,total=asyncio.run(actual_full(text,'length',1,include))
                self.assertEqual(response.status,'incomplete');self.assertEqual(response.incomplete_details.reason,'max_output_tokens')
                self.assertEqual(response.usage.output_tokens,total)
    def test_stop_visible_and_valid_call_complete(self):
        call='<think>Fixture.</think><tool_call>\n<function=exec_command>\n<parameter=cmd>synthetic value</parameter>\n</function>\n</tool_call>'
        for text in ('<think>Fixture.</think>Done.',call):
            events,total=asyncio.run(actual_stream(text,'stop',1))
            self.assertEqual(events[-1].type,'response.completed');self.assertEqual(events[-1].response.status,'completed')
            self.assertEqual(events[-1].response.usage.output_tokens,total)
            self.assertNotIn('response.incomplete',[e.type for e in events])
    def test_stop_empty_remains_failed(self):
        for text in ('','<think>Fixture only.','<think>Fixture.</think>'):
            events,_=asyncio.run(actual_stream(text,'stop'))
            self.assertEqual(events[-1].type,'response.failed');self.assertEqual(events[-1].response.status,'failed')
            self.assertNotIn('response.completed',[e.type for e in events])
    def test_background_replay_stops_at_each_actual_terminal(self):
        from collections import deque
        async def replay(text, finish):
            events,_=await actual_stream(text,finish)
            _,_,_,service=fixture()
            signal=asyncio.Event()
            service.event_store={'synthetic-background':(deque(events),signal)}
            async def collect():
                return [e async for e in service.responses_background_stream_generator('synthetic-background')]
            replayed=await asyncio.wait_for(collect(),timeout=1)
            self.assertEqual([e.type for e in replayed],[e.type for e in events])
            return replayed[-1].type
        for text,finish,expected in (('<think>Fixture.</think>Partial','length','response.incomplete'),('<think>Fixture.</think>Done.','stop','response.completed'),('','stop','response.failed')):
            self.assertEqual(asyncio.run(replay(text,finish)),expected)

    def test_background_waits_after_nonterminal_then_stops_at_incomplete(self):
        from collections import deque
        async def check():
            events,_=await actual_stream('<think>Fixture only.','length',include=False)
            _,_,_,service=fixture()
            signal=asyncio.Event();queue=deque(events[:2])
            service.event_store={'synthetic-pending':(queue,signal)}
            generator=service.responses_background_stream_generator('synthetic-pending')
            self.assertEqual((await anext(generator)).type,'response.created')
            self.assertEqual((await anext(generator)).type,'response.in_progress')
            pending=asyncio.create_task(anext(generator))
            await asyncio.sleep(0.01)
            self.assertFalse(pending.done())
            queue.append(events[-1]);signal.set()
            self.assertEqual((await asyncio.wait_for(pending,1)).type,'response.incomplete')
            with self.assertRaises(StopAsyncIteration):await anext(generator)
        asyncio.run(check())

    def test_stop_nonstream_real_aggregation_completed(self):
        response,total=asyncio.run(actual_full('<think>Fixture.</think>Done.','stop'))
        self.assertEqual(response.status,'completed');self.assertEqual(response.usage.output_tokens,total)


def export():
    root=Path(__file__).resolve().parent/'fixtures';root.mkdir(exist_ok=True)
    cases={'length-visible':('<think>Fixture.</think>Partial','length',True),
           'length-reasoning-hidden':('<think>Fixture only.','length',False),
           'stop-visible':('<think>Fixture.</think>Done.','stop',True),
           'stop-empty':('','stop',False)}
    reports=[]
    for name,(text,finish,include) in cases.items():
        events,_=asyncio.run(actual_stream(text,finish,7,include))
        path=root/(name+'.json');path.write_text(json.dumps([e.model_dump(mode='json',by_alias=True) for e in events],indent=2)+'\n')
        reports.append({'case':name,'terminal_event':events[-1].type,'response_status':events[-1].response.status,'event_count':len(events)})
    (root/'SUMMARY.json').write_text(json.dumps({'producer':'actual responses_stream_generator + actual responses_full_generator + SimpleContext.append_output + actual RequestOutput/CompletionOutput','model_inference':False,'cases':reports},indent=2)+'\n')
    return reports

if __name__=='__main__':
    import io
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(LengthProducer);log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    report={'tests':result.testsRun,'passed':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'log':log.getvalue(),'source_overlay':False,'model_inference':False}
    if result.wasSuccessful():report['fixtures']=export()
    Path(__file__).with_name('producer-result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));raise SystemExit(0 if result.wasSuccessful() else 1)
