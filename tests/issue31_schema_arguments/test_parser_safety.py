# v0.31 fixture adapter only; assertions remain in the author test module.
import json,os
from transformers import AutoTokenizer
from vllm.parser.parser_manager import ParserManager
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
TOK=AutoTokenizer.from_pretrained(os.environ['QWEN_TOKENIZER'],local_files_only=True)
CLS=ParserManager.get_parser(tool_parser_name='qwen3_coder',reasoning_parser_name='qwen3',enable_auto_tools=True)
TOOLS=[{'type':'function','name':'exec_command','parameters':{'type':'object','properties':{'cmd':{'type':'string'}},'required':['cmd']}}]
def call(name='exec_command',value='fixture',last_close=True):
 return '<tool_call>\n<function='+name+'>\n<parameter=cmd>\n'+value+('\n</parameter>' if last_close else '')+'\n</function>\n</tool_call>'
def make(tools=None,thinking=False):
 req=ResponsesRequest(model='qwen',input='CPU fixture',tools=TOOLS if tools is None else tools,tool_choice='auto',store=False)
 p=CLS(TOK,tools=req.tools,chat_template_kwargs={'enable_thinking':thinking})
 prompt=TOK.encode('<|im_start|>assistant\n'+('<think>\n' if thinking else ''),add_special_tokens=False)
 return req,p,prompt
def ids(text):return TOK.encode(text,add_special_tokens=False)
def full(text,thinking=False,tools=None,tokens=None):
 req,p,prompt=make(tools,thinking)
 return p.parse(text,req,enable_auto_tools=True,model_output_token_ids=ids(text) if tokens is None else tokens),p
def streaming(text,size=7,thinking=False,tools=None,tokens=None):
 req,p,prompt=make(tools,thinking);tokens=ids(text) if tokens is None else tokens
 content=[];reasoning=[];calls={};previous=''
 for index in range(0,max(1,len(tokens)),size):
  current=TOK.decode(tokens[:index+size],skip_special_tokens=False)
  if not text.startswith(current) or not current.startswith(previous):continue
  delta=p.parse_delta(current[len(previous):],tokens[index:index+size],req,prompt_token_ids=prompt if index==0 else None,finished=index+size>=len(tokens));previous=current
  if delta:
   content.append(delta.content or '');reasoning.append(delta.reasoning or '')
   for tc in delta.tool_calls or []:
    c=calls.setdefault(tc.index,{'name':'','arguments':''})
    if tc.function:c['name']+=tc.function.name or '';c['arguments']+=tc.function.arguments or ''
 return (''.join(reasoning),''.join(content),list(calls.values())),p
