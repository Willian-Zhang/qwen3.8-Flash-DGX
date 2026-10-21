"""Synthetic engine frames through the installed Chat Completions producer; no tools execute."""
import asyncio,json,logging,os
from types import SimpleNamespace as NS
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.outputs import RequestOutput,CompletionOutput
from vllm.parser import ParserManager

TOKENIZER=AutoTokenizer.from_pretrained(os.environ['QWEN_TOKENIZER'],local_files_only=True)
CLASS=ParserManager.get_parser(tool_parser_name='qwen3_coder',reasoning_parser_name='qwen3',enable_auto_tools=True)
TOOLS=[{'type':'function','function':{'name':'exec_command','parameters':{'type':'object','properties':{'cmd':{'type':'string'}},'required':['cmd']}}}]
def call(name):return '<tool_call>\n<function='+name+'>\n<parameter=cmd>synthetic value</parameter>\n</function>\n</tool_call>'

def fixture():
 s=object.__new__(OpenAIServingChat)
 for k,v in dict(parser_cls=CLASS,model_config=NS(),enable_auto_tools=True,response_role='assistant',default_chat_template_kwargs={},enable_log_outputs=False,enable_log_deltas=False,request_logger=None,_include_reasoning_tokens_details=True,enable_force_include_usage=False,enable_prompt_tokens_details=False,enable_per_request_metrics=False,system_fingerprint=None).items():setattr(s,k,v)
 return s
async def run(text,thinking,size,prompt_open=None):
 req=ChatCompletionRequest(model='qwen',messages=[{'role':'user','content':'Synthetic fixture'}],tools=TOOLS,tool_choice="auto",stream=True,include_reasoning=True)
 ids=TOKENIZER.encode(text,add_special_tokens=False)
 prompt=TOKENIZER.encode('<|im_start|>assistant\n'+('<think>\n' if (thinking if prompt_open is None else prompt_open) else ''),add_special_tokens=False)
 async def outputs():
  previous=''
  for i in range(0,len(ids),size):
   current=TOKENIZER.decode(ids[:i+size],skip_special_tokens=False);delta=current[len(previous):];previous=current
   done=i+size>=len(ids)
   co=CompletionOutput(index=0,text=delta,token_ids=ids[i:i+size],cumulative_logprob=0.,logprobs=None,finish_reason='stop' if done else None)
   yield RequestOutput(request_id='fixture',prompt=None,prompt_token_ids=prompt,prompt_logprobs=None,outputs=[co],finished=done,num_cached_tokens=0)
 service=fixture();metadata=NS(final_usage_info=None)
 raw=[x async for x in service.chat_completion_stream_generator(req,outputs(),'chatcmpl-fixture','qwen',[],TOKENIZER,metadata,{'enable_thinking':thinking})]
 content=[];reason=[];calls={};errors=[];finishes=[]
 for chunk in raw:
  value=chunk.removeprefix('data: ').strip()
  if value=='[DONE]':continue
  row=json.loads(value)
  if 'error' in row:errors.append(row['error'])
  for choice in row.get('choices',[]):
   delta=choice.get('delta',{});content.append(delta.get('content') or '');reason.append(delta.get('reasoning') or delta.get('reasoning_content') or '')
   if choice.get('finish_reason'):finishes.append(choice['finish_reason'])
   for tc in delta.get('tool_calls') or []:
    f=tc.get('function',{});c=calls.setdefault(tc['index'],{'name':'','arguments':''});c['name']+=f.get('name') or '';c['arguments']+=f.get('arguments') or ''
 return dict(content=''.join(content),reasoning=''.join(reason),calls=calls,errors=errors,finishes=finishes,done=raw[-1]=='data: [DONE]\n\n')

async def full(text, thinking=False):
 req=ChatCompletionRequest(model='qwen',messages=[{'role':'user','content':'Synthetic fixture'}],tools=TOOLS,tool_choice='auto',stream=False,include_reasoning=True)
 ids=TOKENIZER.encode(text,add_special_tokens=False)
 prompt=TOKENIZER.encode('<|im_start|>assistant\n'+('<think>\n' if thinking else ''),add_special_tokens=False)
 async def outputs():
  co=CompletionOutput(index=0,text=text,token_ids=ids,cumulative_logprob=0.,logprobs=None,finish_reason='stop')
  yield RequestOutput(request_id='fixture',prompt=None,prompt_token_ids=prompt,prompt_logprobs=None,outputs=[co],finished=True,num_cached_tokens=0)
 service=fixture();parser=service._make_parser(req,TOKENIZER,{'enable_thinking':thinking})
 response=await service.chat_completion_full_generator(req,outputs(),'chatcmpl-fixture','qwen',[],TOKENIZER,NS(final_usage_info=None),parser)
 return response
