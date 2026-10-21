"""Installed v0.31 producer integration; synthetic engine output, zero tool execution."""
import asyncio,json,sys,unittest
from unittest.mock import patch
from pathlib import Path
QA = Path(__file__).resolve().parents[1] / 'issue31_safe_output'
sys.path[:0] = [str(QA), str(QA / 'tests')]
import producer_tests as r
import chat_fixture as c
SCHEMA={'type':'object','properties':{'cmd':{'type':'string'},'count':{'type':'integer'}},'required':['cmd'],'additionalProperties':False}
TOOLS=[{'type':'function','name':'exec_command','parameters':SCHEMA}]
CHAT_TOOLS=[{'type':'function','function':{'name':'exec_command','parameters':SCHEMA}}]
def xml(value,name='exec_command',param='arguments'):
 return '<tool_call>\n<function='+name+'>\n<parameter='+param+'>'+value+'</parameter>\n</function>\n</tool_call>'
def body(value,name='exec_command'):return 'Fixture.</think>'+xml(value,name)
class Integration(unittest.TestCase):
 def test_responses_stream_full_and_persisted_json_agree(self):
  value={'cmd':'printf \'C:\\tmp 日本語\'\nnext\n','count':2}
  with patch.object(r,'TOOLS',TOOLS):
   for size in (1,7,100000):
    events,_=asyncio.run(r.actual_stream(body(json.dumps(value,ensure_ascii=False)),'stop',size))
    self.assertEqual(events[-1].type,'response.completed')
    final=events[-1].response
    self.assertEqual([json.loads(i.arguments) for i in final.output if i.type=='function_call'],[value])
    self.assertEqual(json.loads(''.join(e.delta for e in events if e.type=='response.function_call_arguments.delta')),value)
    self.assertEqual([json.loads(i['arguments']) for i in final.model_dump(mode='json')['output'] if i['type']=='function_call'],[value])
   full,_=asyncio.run(r.actual_full(body(json.dumps(value,ensure_ascii=False)),'stop',1))
   self.assertEqual([json.loads(i.arguments) for i in full.output if i.type=='function_call'],[value])
 def test_responses_invalid_fails_before_tool_item(self):
  from vllm.entrypoints.serve.engine.protocol import ErrorResponse
  with patch.object(r,'TOOLS',TOOLS):
   for value in ('not JSON','{}','{"cmd":17}','{"cmd":"x","extra":1}'):
    events,_=asyncio.run(r.actual_stream(body(value),'stop',1))
    self.assertEqual(events[-1].type,'response.failed')
    self.assertIn('arguments',events[-1].response.error.message)
    self.assertNotIn('absent from offered tools',events[-1].response.error.message)
    self.assertFalse(any(e.type=='response.output_item.added' and e.item.type=='function_call' for e in events))
    full,_=asyncio.run(r.actual_full(body(value),'stop',1));self.assertIsInstance(full,ErrorResponse)
 def test_length_without_closer_remains_incomplete(self):
  text='Fixture.</think><tool_call><function=exec_command><parameter=arguments>{"cmd":"partial'
  with patch.object(r,'TOOLS',TOOLS):
   events,_=asyncio.run(r.actual_stream(text,'length',1))
   self.assertEqual(events[-1].type,'response.incomplete')
   self.assertFalse(any(e.type=='response.output_item.added' and e.item.type=='function_call' for e in events))
 def test_chat_wrapped_args_normalized_stream_and_full(self):
  value={'cmd':'C:\\tmp\\日本語\nnext','count':3}
  with patch.object(c,'TOOLS',CHAT_TOOLS):
   for size in (1,7,100000):
    result=asyncio.run(c.run(body(json.dumps(value,ensure_ascii=False)),True,size))
    self.assertEqual(result['errors'],[]);self.assertTrue(result['done'])
    self.assertEqual(json.loads(result['calls'][0]['arguments']),value)
   full=asyncio.run(c.full(body(json.dumps(value,ensure_ascii=False)),True))
   self.assertEqual(json.loads(full.choices[0].message.tool_calls[0].function.arguments),value)
 def test_chat_invalid_wrapper_retains_client_error_path(self):
  with patch.object(c,'TOOLS',CHAT_TOOLS):
   result=asyncio.run(c.run(body('{}'),True,1))
   self.assertEqual(result['errors'],[])
   self.assertEqual(json.loads(result['calls'][0]['arguments']),{'arguments':'{}'})
   self.assertEqual(result['finishes'],['tool_calls'])
 def test_unknown_chat_name_not_authorized_by_wrapper(self):
  with patch.object(c,'TOOLS',CHAT_TOOLS):
   result=asyncio.run(c.run(body('{"cmd":"fixture"}','unknown_tool'),True,1))
   self.assertEqual(result['calls'][0]['name'],'unknown_tool')
   self.assertEqual(json.loads(result['calls'][0]['arguments']),{'arguments':'{"cmd":"fixture"}'})
 def test_implicit_reasoning_calls_use_schema_repair(self):
  text='Fixture.\n'+xml('{"cmd":"fixture"}')
  with patch.object(r,'TOOLS',TOOLS):
   for size in (1,7,100000):
    events,_=asyncio.run(r.actual_stream(text,'stop',size,thinking=True))
    self.assertEqual(events[-1].type,'response.completed')
    self.assertEqual([json.loads(i.arguments) for i in events[-1].response.output if i.type=='function_call'],[{'cmd':'fixture'}])
 def test_implicit_bad_args_fail_without_executable_item(self):
  text='Fixture.\n'+xml('{}')
  with patch.object(r,'TOOLS',TOOLS):
   events,_=asyncio.run(r.actual_stream(text,'stop',1,thinking=True))
   self.assertEqual(events[-1].type,'response.failed')
   self.assertIn('arguments',events[-1].response.error.message)
   self.assertFalse(any(e.type=='response.output_item.added' and e.item.type=='function_call' for e in events))
 def test_ambiguous_schema_keeps_legacy_shape(self):
  with patch.object(r,'TOOLS',TOOLS+TOOLS):
   events,_=asyncio.run(r.actual_stream(body('{"cmd":"fixture"}'),'stop',1))
   self.assertEqual(events[-1].type,'response.completed')
   self.assertEqual([json.loads(i.arguments) for i in events[-1].response.output if i.type=='function_call'],[{'arguments':'{"cmd":"fixture"}'}])
 def test_regular_chat_arguments_still_stream_before_close(self):
  from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
  req=ChatCompletionRequest(model='qwen',messages=[{'role':'user','content':'fixture'}],tools=CHAT_TOOLS)
  parser=c.CLASS(c.TOKENIZER,tools=req.tools,chat_template_kwargs={'enable_thinking':False})
  text=xml('a'*256,param='cmd');ids=c.TOKENIZER.encode(text,add_special_tokens=False)
  prompt=c.TOKENIZER.encode('<|im_start|>assistant\n',add_special_tokens=False)
  previous='';early=False
  for i,token in enumerate(ids):
   current=c.TOKENIZER.decode(ids[:i+1],skip_special_tokens=False)
   delta=parser.parse_delta(current[len(previous):],[token],req,prompt_token_ids=prompt if i==0 else None,finished=i==len(ids)-1);previous=current
   if delta and any(tc.function and tc.function.arguments for tc in delta.tool_calls or []):
    if '</parameter>' not in current:early=True
  self.assertTrue(early)
 def test_schema_changes_between_requests_do_not_leak(self):
  from vllm.parser.qwen3 import Qwen3Parser
  from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
  alternate=[{'type':'function','name':'exec_command','parameters':{'type':'object','properties':{'arguments':{'type':'string'}},'required':['arguments'],'additionalProperties':False}}]
  a=ResponsesRequest(model='qwen',input='fixture',tools=TOOLS)
  b=ResponsesRequest(model='qwen',input='fixture',tools=alternate)
  parser=Qwen3Parser(c.TOKENIZER,tools=a.tools,chat_template_kwargs={'enable_thinking':False})
  for request,expected in [(a,{'cmd':'fixture'}),(b,{'arguments':'{"cmd":"fixture"}'}),(a,{'cmd':'fixture'})]:
   result=parser.extract_tool_calls(xml('{"cmd":"fixture"}'),request)
   self.assertEqual(json.loads(result.tool_calls[0].function.arguments),expected)
if __name__=='__main__':unittest.main(verbosity=2)
