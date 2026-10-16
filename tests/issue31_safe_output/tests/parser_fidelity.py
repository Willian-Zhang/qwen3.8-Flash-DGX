"""Actual ParserManager tokenizer-only proof. No model weights or inference."""
import hashlib, importlib, importlib.util, json, sys, os
from pathlib import Path
from transformers import AutoTokenizer
from vllm.parser.parser_manager import ParserManager
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

ROOT=Path(__file__).resolve().parent
MODEL=os.environ['QWEN_TOKENIZER']
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
tools=[{'type':'function','name':'exec_command','parameters':{'type':'object','properties':{'cmd':{'type':'string'}},'required':['cmd']}}]
req=ResponsesRequest(model='qwen',input='public synthetic CPU fixture',tools=tools)
markers=['<|im_end|>','<|im_start|>','<|endoftext|>','<tool_call>','</tool_call>','<think>','</think>']
protocol_ids={tok.convert_tokens_to_ids(m) for m in markers}
def base_bpe(text):
    ids=[]
    for part,_ in tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text):
        ids.extend(t.id for t in tok.backend_tokenizer.model.tokenize(part))
    assert tok.decode(ids,skip_special_tokens=False)==text
    assert not (set(ids)&protocol_ids)
    return ids

def parser():
    cls=ParserManager.get_parser(tool_parser_name='qwen3_coder',reasoning_parser_name='qwen3',enable_auto_tools=True)
    return cls(tok,tools=req.tools,chat_template_kwargs={'enable_thinking':True})
def stream(text,ids,size):
    p=parser();parts=[];reason=[];calls={};prior=''
    for i in range(0,len(ids),size):
        current=tok.decode(ids[:i+size],skip_special_tokens=False)
        d=p.parse_delta(current[len(prior):],ids[i:i+size],req,prompt_token_ids=[] if i==0 else None,finished=i+size>=len(ids));prior=current
        if not d:continue
        parts.append(d.content or '');reason.append(d.reasoning or '')
        for t in d.tool_calls or []:
            x=calls.setdefault(t.index,{'name':'','arguments':''})
            if t.function:x['name']+=t.function.name or '';x['arguments']+=t.function.arguments or ''
    return ''.join(reason),''.join(parts),list(calls.values())
def final(text,ids):
    r,c,ts=parser().parse(text,req,enable_auto_tools=True,model_output_token_ids=ids)
    return r or '',c or '',[{'name':t.name,'arguments':t.arguments} for t in ts or []]
def record(label,text,ids):
    expected=stream(text,ids,len(ids));actual=final(text,ids)
    return {'case':label,'stream_equals_final':expected==actual,'all_stream_chunk_sizes_match':all(stream(text,ids,n)==expected for n in (1,2,7)), 'final_digest':hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest(),'tool_count':len(actual[2])}

fixtures=[]
for marker in markers:
    prefix='<think>CPU fixture.</think>Literal `';suffix='` remains data.'
    fixtures.append((f'final_literal_{marker}',prefix+marker+suffix,tok.encode(prefix,add_special_tokens=False)+base_bpe(marker)+tok.encode(suffix,add_special_tokens=False)))
    prefix='<think>Quote `';suffix='` as data.</think>Done.'
    fixtures.append((f'reasoning_literal_{marker}',prefix+marker+suffix,tok.encode(prefix,add_special_tokens=False)+base_bpe(marker)+tok.encode(suffix,add_special_tokens=False)))
    prefix='<think>CPU fixture.</think><tool_call>\n<function=exec_command>\n<parameter=cmd>write ';suffix=' exactly</parameter>\n</function>\n</tool_call>'
    fixtures.append((f'argument_literal_{marker}',prefix+marker+suffix,tok.encode(prefix,add_special_tokens=False)+base_bpe(marker)+tok.encode(suffix,add_special_tokens=False)))
candidate=[record(*x) for x in fixtures]
for label,text,ids in fixtures:
    r,c,ts=final(text,ids)
    marker=label.split('_literal_',1)[1]
    if label.startswith('final_'):assert c=='Literal `'+marker+'` remains data.' and not ts
    elif label.startswith('reasoning_'):assert r=='Quote `'+marker+'` as data.' and c=='Done.' and not ts
    else:assert len(ts)==1 and ts[0]['name']=='exec_command' and json.loads(ts[0]['arguments'])['cmd']=='write '+marker+' exactly'
p=parser();text='<think>CPU fixture.</think>Done.'
normal=final(text,tok.encode(text,add_special_tokens=False));assert normal[1]=='Done.' and not normal[2]
print(json.dumps({'installed_cases':candidate,'stream_final_mismatches':sum(not x['stream_equals_final'] for x in candidate),'chunk_mismatches':sum(not x['all_stream_chunk_sizes_match'] for x in candidate),'ordinary_bpe_protocol_ids_absent':True,'normal_final_answer_passed':True,'literal_byte_fidelity_cases':len(fixtures),'no_model_inference':True,'source_overlay':False},indent=2))
assert all(x['stream_equals_final'] and x['all_stream_chunk_sizes_match'] for x in candidate)
